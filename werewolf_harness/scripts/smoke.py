"""End-to-end check of a running stack: one mock game through the front door,
then every piece around it (metrics, scrape targets, dashboard, replay).

    make smoke                                         # brings the stack up, runs this inside api1
    python -m werewolf_harness.scripts.smoke --base http://caddy:80 ...

Standard library only, so it runs in the API image and on a bare host alike.
It drives the public HTTP API exactly as a browser would -- guest login,
create a game whose seats are all `mock`, poll until it is over -- so a green
smoke means the proxy, both API nodes' health, the Redis-Streams workers (with
AGENT_EXECUTOR=streams every model turn crosses a worker) and Postgres are wired.
Exit code 0 = all checks passed; each check prints one line.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request


def _http(method: str, url: str, body: dict | None = None, token: str | None = None, timeout: float = 10) -> tuple[int, str]:
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _json(method: str, url: str, **kw) -> dict:
    code, text = _http(method, url, **kw)
    if code >= 300:
        raise AssertionError(f"{method} {url} -> {code}: {text[:300]}")
    return json.loads(text)


def _wait(what: str, fn, timeout: float, every: float = 1.0):
    """Poll fn() until it returns something truthy; the last value is reported on timeout."""
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            last = fn()
            if last:
                return last
        except (OSError, AssertionError, ValueError) as e:      # not up yet
            last = e
        time.sleep(every)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what} (last: {last!r})"[:500])


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'通过' if ok else '失败'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        raise SystemExit(1)


def play_one(base: str, variant: str, seed: int, timeout: float) -> tuple[str, dict, float]:
    tok = _json("POST", f"{base}/api/auth/guest", body={"name": "smoke"})["token"]
    t0 = time.monotonic()
    g = _json("POST", f"{base}/api/games", token=tok, body={"variant": variant, "models": ["mock"], "seed": seed})
    gid = g["id"]

    def over():
        s = _json("GET", f"{base}/api/games/{gid}")["game"]
        return s if s["state"] in ("over", "error") else None
    s = _wait(f"game {gid} to finish", over, timeout)
    return gid, s, time.monotonic() - t0


def prom_targets(prom: str) -> list[dict]:
    return _json("GET", f"{prom}/api/v1/targets")["data"]["activeTargets"]


async def replay_all(db_url: str) -> dict:
    from ..app.storage import Store
    from .verify_replay import verify
    store = Store.connect(db_url)
    try:
        return await verify(store)
    finally:
        await store.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://caddy:80", help="the public entry point (Caddy)")
    ap.add_argument("--api", default="http://api1:8402", help="one API node, for /metrics")
    ap.add_argument("--worker", default="http://worker:9100", help="a worker's metrics endpoint ('' = skip)")
    ap.add_argument("--prometheus", default="http://prometheus:9090", help="'' = skip")
    ap.add_argument("--grafana", default="http://grafana:3000", help="'' = skip")
    ap.add_argument("--variant", default="8p")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=240)
    ap.add_argument("--no-replay", action="store_true", help="skip the (seed, journal) replay check (needs DATABASE_URL)")
    a = ap.parse_args()

    print("冒烟测试：", a.base, flush=True)
    _wait("the API behind the proxy", lambda: _http("GET", f"{a.base}/readyz")[0] == 200, 60)
    check("Caddy -> API /readyz", True)

    gid, s, secs = play_one(a.base, a.variant, a.seed, a.timeout)
    check("一局 mock 对局打完", s["state"] == "over",
          f"game={gid} winner={(s.get('outcome') or {}).get('winner')} rounds={s.get('round')} "
          f"turns={s.get('version')} {secs:.1f}s")
    check("没有规则兜底替代的回合", (s.get("health") or {}).get("forced", 0) == 0, json.dumps(s.get("health"))[:200])

    code, text = _http("GET", f"{a.api}/metrics")
    check("API /metrics", code == 200 and "wh_games_finished_total" in text and "wh_games_created_total" in text)
    if a.worker:
        code, text = _http("GET", f"{a.worker}/metrics")
        check("worker /metrics", code == 200 and "wh_llm_queue_wait_ms" in text)

    if a.prometheus:
        want = {"api", "worker"}

        def all_up():
            ts = prom_targets(a.prometheus)
            jobs = {t["labels"]["job"] for t in ts}
            return ts if want <= jobs and all(t["health"] == "up" for t in ts) else None
        ts = _wait("every Prometheus target to be up", all_up, 60, every=2)
        check("Prometheus 抓取目标全部 up", True, ", ".join(sorted(t["labels"]["instance"] for t in ts)))
        # the game may have ended after the last scrape: wait a scrape interval or two
        q = _wait("wh_games_finished_total in Prometheus", lambda: _json(
            "GET", f"{a.prometheus}/api/v1/query?query=sum(wh_games_finished_total)")["data"]["result"], 30, every=2)
        check("Prometheus 能查到对局指标", True, f"finished={q[0]['value'][1]}")

        def worker_jobs():
            q = _json("GET", f"{a.prometheus}/api/v1/query?query=sum+by+(instance)(wh_llm_queue_wait_ms_count)")
            per = {r["metric"]["instance"]: int(float(r["value"][1])) for r in q["data"]["result"]}
            # every committed turn of an all-model table was one worker job (asks ahead may add more)
            return per if sum(per.values()) >= (s.get("version") or 1) else None
        per = _wait("worker job counts in Prometheus to cover every turn", worker_jobs, 30, every=2)
        check("模型回合经 Redis Streams 由 worker 执行", True, " ".join(f"{k}={v}" for k, v in sorted(per.items())))

    if a.grafana:
        def dash():
            hits = _json("GET", f"{a.grafana}/api/search?query=werewolf")
            return [h for h in hits if h.get("uid") == "werewolf"]
        hits = _wait("the provisioned Grafana dashboard", dash, 60, every=2)
        check("Grafana 看板已自动加载", True, hits[0].get("url", ""))

    if not a.no_replay:
        db = os.environ.get("DATABASE_URL")
        check("有 DATABASE_URL 可做回放校验", bool(db))
        r = asyncio.run(replay_all(db))
        check("回放校验 (seed, journal) -> 同一结局", r["checked"] >= 1 and not r["mismatches"],
              f"checked={r['checked']} mismatches={len(r['mismatches'])}")
    print("全部通过", flush=True)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"  失败  {e}", flush=True)
        sys.exit(1)
