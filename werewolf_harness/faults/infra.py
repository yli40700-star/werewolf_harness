"""Postgres and Redis for a fault run, each behind a FaultProxy.

Own (default when the binaries are on this machine): a throwaway Redis
(`redis-server`, no RDB, no AOF) and a throwaway Postgres cluster (`initdb` in a
temp dir, fsync off) on free ports. "Restart" is real: Redis is SIGKILLed and
started empty; Postgres is stopped with `-m immediate` (a crash: in-flight
transactions die, the server recovers from WAL on start).

Shared (CI, where Postgres / Redis are service containers this job cannot
restart): the given URLs. "Restart" is emulated at the proxy: refuse + reset
every connection for the outage, and for Redis also FLUSHDB -- what a restart
without persistence leaves behind.
"""
from __future__ import annotations

import asyncio
import glob
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlparse

from .proxy import FaultProxy


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pg_bin(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for pat in ("/opt/homebrew/opt/postgresql@*/bin", "/usr/local/opt/postgresql@*/bin", "/usr/lib/postgresql/*/bin"):
        for d in sorted(glob.glob(pat), reverse=True):
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
    return None


def can_own_redis() -> bool:
    return shutil.which("redis-server") is not None


def can_own_pg() -> bool:
    return all(_pg_bin(n) for n in ("initdb", "pg_ctl", "postgres"))


async def _wait_port(port: int, timeout: float = 20) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            _r, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise RuntimeError(f"nothing listening on {port}")


# ---------------------------------------------------------------- Redis
class Redis:
    def __init__(self, own: bool, url: str | None = None):
        self.own = own
        self.proc: subprocess.Popen | None = None
        self.port = free_port() if own else urlparse(url).port or 6379
        self.host = "127.0.0.1" if own else (urlparse(url).hostname or "127.0.0.1")
        self.db = 0 if own else int((urlparse(url).path or "/0").strip("/") or 0)
        self.proxy: FaultProxy | None = None

    @property
    def direct_url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"

    @property
    def url(self) -> str:                                  # what the services use
        return f"redis://127.0.0.1:{self.proxy.port}/{self.db}"

    async def start(self) -> None:
        if self.own:
            self._spawn()
            await _wait_port(self.port)
        else:
            await self.flush()
        self.proxy = await FaultProxy(self.host, self.port, name="redis").start()

    def _spawn(self) -> None:
        self.proc = subprocess.Popen(["redis-server", "--port", str(self.port), "--bind", "127.0.0.1", "--save", "",
                                      "--appendonly", "no", "--protected-mode", "no"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    async def flush(self) -> None:
        from redis.asyncio import Redis as R
        r = R.from_url(self.direct_url)
        try:
            await r.flushdb()
        finally:
            await r.aclose()

    async def restart(self, down_s: float = 1.0) -> str:
        """Gone and back, empty. Returns how it was done."""
        assert self.proxy is not None
        if self.own:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait()
            self.proxy.cut()
            await asyncio.sleep(down_s)
            self._spawn()
            await _wait_port(self.port)
            return "SIGKILL redis-server, start an empty one"
        self.proxy.refuse(True)
        self.proxy.cut()
        await self.flush()
        await asyncio.sleep(down_s)
        self.proxy.refuse(False)
        return "proxy: refuse + reset connections, FLUSHDB (shared Redis)"

    async def stop(self) -> None:
        if self.proxy:
            await self.proxy.close()
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


# ---------------------------------------------------------------- Postgres
class Postgres:
    def __init__(self, own: bool, url: str | None = None):
        self.own = own
        self.dir: str | None = None
        self.proxy: FaultProxy | None = None
        if own:
            self.port, self.host, self.user, self.password, self.dbname = free_port(), "127.0.0.1", "postgres", "", "wh"
        else:
            u = urlparse(url.replace("postgresql+asyncpg", "postgresql"))
            self.port, self.host = u.port or 5432, u.hostname or "127.0.0.1"
            self.user, self.password, self.dbname = u.username or "postgres", u.password or "", u.path.strip("/")

    def _url(self, host: str, port: int) -> str:
        auth = self.user + (f":{self.password}" if self.password else "")
        return f"postgresql+asyncpg://{auth}@{host}:{port}/{self.dbname}"

    @property
    def direct_url(self) -> str:
        return self._url(self.host, self.port)

    @property
    def url(self) -> str:
        return self._url("127.0.0.1", self.proxy.port)

    async def start(self) -> None:
        if self.own:
            self.dir = tempfile.mkdtemp(prefix="wh-faults-pg-")
            subprocess.run([_pg_bin("initdb"), "-D", self.dir, "-U", "postgres", "--auth=trust", "-E", "UTF8",
                            "--no-sync"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._pg_ctl("start")
            await _wait_port(self.port)
            await self._createdb()
        await self.reset_schema()
        self.proxy = await FaultProxy(self.host, self.port, name="postgres").start()

    def _pg_ctl(self, what: str, *extra: str) -> None:
        opts = f"-p {self.port} -k {self.dir} -c listen_addresses=127.0.0.1 -c fsync=off -c max_connections=300"
        cmd = [_pg_bin("pg_ctl"), "-D", self.dir, "-l", os.path.join(self.dir, "log"), "-w", *extra, what]
        if what == "start":
            cmd[1:1] = ["-o", opts]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    async def _createdb(self) -> None:
        import asyncpg
        for _ in range(50):
            try:
                c = await asyncpg.connect(host=self.host, port=self.port, user="postgres", database="postgres")
                break
            except (OSError, asyncpg.PostgresError):
                await asyncio.sleep(0.1)
        try:
            await c.execute(f"CREATE DATABASE {self.dbname}")
        finally:
            await c.close()

    async def reset_schema(self) -> None:
        from ..app.storage import migrate
        await migrate.adowngrade(self.direct_url)
        await migrate.aupgrade(self.direct_url)

    async def restart(self, down_s: float = 1.0) -> str:
        assert self.proxy is not None
        if self.own:
            self._pg_ctl("stop", "-m", "immediate")
            self.proxy.cut()
            await asyncio.sleep(down_s)
            self._pg_ctl("start")
            await _wait_port(self.port)
            return "pg_ctl stop -m immediate (crash, WAL recovery on start)"
        self.proxy.refuse(True)
        self.proxy.cut()
        await asyncio.sleep(down_s)
        self.proxy.refuse(False)
        return "proxy: refuse + reset connections (shared Postgres)"

    async def stop(self) -> None:
        if self.proxy:
            await self.proxy.close()
        if self.own and self.dir:
            try:
                self._pg_ctl("stop", "-m", "immediate")
            except subprocess.CalledProcessError:
                pass
            shutil.rmtree(self.dir, ignore_errors=True)
