"""The service as it runs in production, as real processes on this machine:
API nodes (uvicorn) and model workers (Redis Streams consumers), all reaching
Postgres and Redis through the fault proxies.

Each process can be killed (SIGKILL: no drain, no goodbye), paused (SIGSTOP)
and resumed (SIGCONT), and started again under the same name.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

from .infra import free_port

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # holds werewolf_harness/

# Fast clocks so a fault plays out in seconds, not minutes.
TIMING = {
    "LEASE_TTL_MS": "3000", "LEASE_RENEW_S": "0.5", "RECOVERY_SCAN_S": "0.5",
    "AGENT_TIMEOUT_S": "25", "CLAIM_IDLE_MS": "3000",
}


class Proc:
    def __init__(self, name: str, kind: str, argv: list[str], env: dict, log_dir: str, port: int | None = None):
        self.name, self.kind, self.argv, self.env, self.port = name, kind, argv, env, port
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self.p: subprocess.Popen | None = None
        self.starts = 0

    def start(self) -> None:
        self.starts += 1
        log = open(self.log_path, "ab")
        log.write(f"\n===== start #{self.starts} at {time.strftime('%H:%M:%S')} =====\n".encode())
        self.p = subprocess.Popen(self.argv, env=self.env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.close()

    @property
    def alive(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def kill(self) -> None:
        if self.alive:
            self.p.send_signal(signal.SIGKILL)
            self.p.wait()

    def pause(self) -> None:
        self.p.send_signal(signal.SIGSTOP)

    def resume(self) -> None:
        self.p.send_signal(signal.SIGCONT)

    def stop(self) -> None:
        if self.alive:
            self.p.send_signal(signal.SIGCONT)
            self.p.terminate()
            try:
                self.p.wait(8)
            except subprocess.TimeoutExpired:
                self.kill()


class Cluster:
    def __init__(self, db_url: str, redis_url: str, run_dir: str, nodes: int = 2, workers: int = 2):
        self.run_dir = run_dir
        self.control = os.path.join(run_dir, "control.json")
        self.call_log = os.path.join(run_dir, "calls.jsonl")
        self.set_control({})
        base = {**os.environ, "PYTHONPATH": ROOT, "DATABASE_URL": db_url, "REDIS_URL": redis_url,
                "AGENT_EXECUTOR": "streams", "AGENT_WORKERS": "8", "ADMISSION": "false",
                "RATE_LIMIT_PER_MIN": "1000000", "CREATE_LIMIT_PER_HOUR": "100000",
                "JWT_SECRET": "faults-run-secret-faults-run-secret!!", "LOG_LEVEL": "WARNING",
                "FAULTS_CONTROL": self.control, "FAULTS_CALL_LOG": self.call_log,
                "OTEL_SDK_DISABLED": "true", **TIMING}
        self.nodes: list[Proc] = []
        for i in range(1, nodes + 1):
            port = free_port()
            self.nodes.append(Proc(f"api{i}", "api", [sys.executable, "-m", "uvicorn",
                                                       "werewolf_harness.app.api.main:app", "--port", str(port),
                                                       "--log-level", "warning"],
                                   {**base, "NODE_ID": f"api{i}"}, run_dir, port))
        self.workers = [Proc(f"worker{i}", "worker", [sys.executable, "-m", "werewolf_harness.app.llm.worker"],
                             {**base, "NODE_ID": f"worker{i}"}, run_dir) for i in range(1, workers + 1)]

    @property
    def procs(self) -> list[Proc]:
        return self.nodes + self.workers

    def set_control(self, c: dict) -> None:
        tmp = self.control + ".tmp"
        with open(tmp, "w") as f:
            json.dump(c, f)
        os.replace(tmp, self.control)                      # atomic: a reader sees old or new, never half

    async def start(self) -> None:
        for p in self.procs:
            p.start()
        await asyncio.gather(*(self.ready(n) for n in self.nodes))

    async def ready(self, n: Proc, timeout: float = 40) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not n.alive:
                raise RuntimeError(f"{n.name} exited; see {n.log_path}")
            try:
                if (await self.get(n, "/readyz")).get("ok"):
                    return
            except Exception:                              # noqa: BLE001
                pass
            await asyncio.sleep(0.2)
        raise RuntimeError(f"{n.name} not ready after {timeout}s; see {n.log_path}")

    async def restart(self, p: Proc) -> None:
        p.kill()
        p.start()
        if p.kind == "api":
            await self.ready(p)

    def stop(self) -> None:
        """SIGTERM to all at once, then SIGKILL whatever is still up after 3 s."""
        for p in self.procs:
            if p.alive:
                p.p.send_signal(signal.SIGCONT)
                p.p.terminate()
        t0 = time.monotonic()
        for p in self.procs:
            try:
                p.p.wait(max(0.1, 3 - (time.monotonic() - t0)))
            except subprocess.TimeoutExpired:
                p.kill()

    # ------------------------------------------------------------ HTTP (blocking calls, run in a thread)
    async def get(self, n: Proc, path: str) -> dict:
        return await asyncio.to_thread(_http, n.port, path, None, None)

    async def post(self, n: Proc, path: str, body: dict, token: str | None = None) -> dict:
        return await asyncio.to_thread(_http, n.port, path, body, token)

    def up_nodes(self) -> list[Proc]:
        return [n for n in self.nodes if n.alive]


def _http(port: int, path: str, body: dict | None, token: str | None) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())
