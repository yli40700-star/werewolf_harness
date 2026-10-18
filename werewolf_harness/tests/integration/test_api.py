"""P3: the HTTP/WebSocket surface -- auth, seat privacy, idempotent acts, resume, limits."""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from .conftest import DB_URL, REDIS_URL


async def _clean():
    from redis.asyncio import Redis
    from sqlalchemy import text

    from werewolf_harness.app.storage import Store
    s = Store.connect(DB_URL, pool_size=2)
    async with s.engine.begin() as c:
        await c.execute(text("TRUNCATE llm_calls, agent_state, journal, game_seats, games, users CASCADE"))
    await s.close()
    r = Redis.from_url(REDIS_URL)
    await r.flushdb()
    await r.aclose()


@pytest.fixture
def client_factory(_migrated, settings):
    made = []

    def make(**over):
        asyncio.run(_clean())
        from werewolf_harness.app.api.main import create_app
        app = create_app(settings.model_copy(update=over), migrate_on_start=False, env_keys={})
        c = TestClient(app)
        c.__enter__()
        made.append(c)
        return c

    yield make
    for c in made:
        c.__exit__(None, None, None)


def _guest(c):
    r = c.post("/api/auth/guest", json={"name": "t"}).json()
    return {"Authorization": f"Bearer {r['token']}"}, r["user"]


def _wait(c, gid, pred, headers=None, seat=None, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        q = f"?seat={seat}" if seat else ""
        s = c.get(f"/api/games/{gid}{q}", headers=headers or {}).json()
        if pred(s):
            return s
        time.sleep(0.05)
    raise AssertionError("timeout")


def _new_game(c, h, **kw):
    body = {"variant": "8p", "models": ["mock"], "human_seats": [3], "turn_seconds": 30, "seed": 7, **kw}
    r = c.post("/api/games", json=body, headers=h)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_creating_needs_a_login(client_factory):
    c = client_factory()
    assert c.post("/api/games", json={"variant": "8p"}).status_code == 401


def test_a_model_nobody_can_serve_is_refused_at_creation(client_factory):
    # a seat with no key would be played by the rule fallback for the whole game
    c = client_factory()
    h, _ = _guest(c)
    r = c.post("/api/games", json={"variant": "8p", "models": ["deepseek-v4-pro", "mock"], "human_seats": [3]}, headers=h)
    assert r.status_code == 400 and "deepseek-v4-pro" in r.text
    assert "mock" in c.get("/api/meta").json()["models"]


def test_a_seat_is_private_to_whoever_claimed_it(client_factory):
    c = client_factory()
    h, _ = _guest(c)
    gid = _new_game(c, h)
    # nobody can read seat 3 -- or an AI seat -- by just putting it in the URL (the old server allowed both)
    for seat in (3, 5):
        s = c.get(f"/api/games/{gid}?seat={seat}").json()
        assert s["need_join"] is True and s["seat"] is None
        assert s["view"].get("role") is None and "private" not in s["view"]
    j = c.post(f"/api/games/{gid}/join", json={"seat": 3}, headers=h).json()
    tok = {"Authorization": f"Bearer {j['seat_token']}"}
    s = c.get(f"/api/games/{gid}?seat=3", headers=tok).json()
    assert s["seat"] == 3 and s["view"]["role"] and "me" in s
    # the token is for seat 3 only
    assert c.get(f"/api/games/{gid}?seat=4", headers=tok).json()["seat"] is None
    # someone else cannot take the seat
    h2, _ = _guest(c)
    assert c.post(f"/api/games/{gid}/join", json={"seat": 3}, headers=h2).status_code == 409
    assert c.post(f"/api/games/{gid}/join", json={"seat": 4}, headers=h2).status_code == 409   # an AI seat


def test_start_and_act_need_the_seat_token_and_a_retry_lands_once(client_factory):
    c = client_factory()
    h, _ = _guest(c)
    gid = _new_game(c, h)
    assert c.post(f"/api/games/{gid}/start", headers=h).status_code == 403          # a user token is not a seat
    tok = {"Authorization": "Bearer " + c.post(f"/api/games/{gid}/join", json={"seat": 3}, headers=h).json()["seat_token"]}
    assert c.post(f"/api/games/{gid}/start", headers=tok).status_code == 200
    s = _wait(c, gid, lambda s: s.get("me", {}).get("waiting"), tok, seat=3)
    t = s["me"]["turn"]
    body = {"seat": 3, "seq": t["seq"], "action": t["fallback"], "request_id": "req-1"}
    assert c.post(f"/api/games/{gid}/act", json=body).status_code == 403              # no token
    assert c.post(f"/api/games/{gid}/act", json=body, headers=tok).json()["status"] == "accepted"
    assert c.post(f"/api/games/{gid}/act", json=body, headers=tok).json()["status"] == "duplicate"


def test_websocket_act_resume_and_no_duplicate_events(client_factory):
    c = client_factory()
    h, _ = _guest(c)
    gid = _new_game(c, h)
    seat_token = c.post(f"/api/games/{gid}/join", json={"seat": 3}, headers=h).json()["seat_token"]
    c.post(f"/api/games/{gid}/start", headers={"Authorization": f"Bearer {seat_token}"})
    seen: list[int] = []
    with c.websocket_connect(f"/ws/games/{gid}?token={seat_token}") as ws:
        hello = ws.receive_json()
        assert hello["op"] == "hello" and hello["seat"] == 3
        acted = False
        while not acted:
            m = ws.receive_json()
            if m["op"] == "event":
                seen.append(m["seq"])
            s = c.get(f"/api/games/{gid}?seat=3", headers={"Authorization": f"Bearer {seat_token}"}).json()
            if s["me"]["waiting"]:
                t = s["me"]["turn"]
                ws.send_json({"op": "act", "seq": t["seq"], "action": t["fallback"], "request_id": "ws-1"})
                while True:
                    m = ws.receive_json()
                    if m["op"] == "ack":
                        assert m["result"]["status"] == "accepted"
                        break
                    if m["op"] == "event":
                        seen.append(m["seq"])
                acted = True
    assert seen == list(range(len(seen)))                      # in order, nothing skipped
    last = seen[-1]
    # reconnect with last_seq: the stream continues exactly after it
    with c.websocket_connect(f"/ws/games/{gid}?token={seat_token}&last_seq={last}") as ws:
        ws.receive_json()                                      # hello
        m = ws.receive_json()
        while m["op"] != "event":
            m = ws.receive_json()
        assert m["seq"] == last + 1


def test_a_silent_websocket_is_closed_after_two_missed_heartbeats(client_factory):
    c = client_factory(heartbeat_s=0.3)
    h, _ = _guest(c)
    gid = _new_game(c, h, turn_seconds=None)
    with c.websocket_connect(f"/ws/games/{gid}") as ws:
        ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as e:
            for _ in range(200):
                ws.receive_json()
        assert e.value.code == 4000


def test_rate_limit_per_ip(client_factory):
    c = client_factory(rate_limit_per_min=5)
    codes = [c.get("/api/meta").status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]


def test_leaderboard_and_metrics_after_a_game(client_factory):
    c = client_factory()
    h, user = _guest(c)
    gid = _new_game(c, h, turn_seconds=0.2)           # the person never answers: every turn is the clock's
    tok = {"Authorization": "Bearer " + c.post(f"/api/games/{gid}/join", json={"seat": 3}, headers=h).json()["seat_token"]}
    c.post(f"/api/games/{gid}/start", headers=tok)
    _wait(c, gid, lambda s: s["game"]["state"] == "over", timeout=90)
    time.sleep(0.2)
    users = c.get("/api/leaderboard").json()["users"]
    assert any(u["user"] == user["id"] for u in users)             # ranked even with 0 wins
    models = c.get("/api/leaderboard?kind=models").json()["models"]
    assert {m["model"] for m in models} == {"mock"} and sum(m["games"] for m in models) == 7
    me = c.get("/api/me", headers=h).json()
    assert me["games"][0]["id"] == gid
    body = c.get("/metrics").text
    assert "wh_commit_ms_count" in body and "wh_games_finished_total" in body
    # after the game every seat is open: the log carries the full journal
    log = c.get(f"/api/games/{gid}/log").json()
    assert log["revealed"] and len(log["journal"]) == log["game"]["version"]


def test_sse_for_spectators_carries_only_public_events(client_factory):
    c = client_factory()
    h, _ = _guest(c)
    gid = _new_game(c, h, human_seats=[], turn_seconds=None)
    got = []
    with c.stream("GET", f"/api/games/{gid}/events?after=0") as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                import json
                got.append(json.loads(line[6:]))
            if len(got) >= 30:
                break
    assert [e["seq"] for e in got] == list(range(len(got)))
    for e in got:
        if e["kind"] in ("phase", "turn") and e.get("task") in ("night", "pack_chat", "pack_plan"):
            raise AssertionError(f"a private task leaked: {e}")


def test_a_cascade_game_is_created_over_http_and_counts_its_escalations(client_factory):
    # docs/adr/0014: every AI seat starts on the cheap tier and moves to the strong one in-turn
    c = client_factory()
    h, _ = _guest(c)
    bad = [({"cheap": "mock-flaky"}, "cheap 和 strong"),                         # one tier missing
           ({"cheap": "mock-flaky", "strong": "mock-strong", "mode": "sometimes"}, "sometimes"),
           ({"cheap": "deepseek-v4-pro", "strong": "mock-strong"}, "deepseek-v4-pro")]   # a tier nobody serves
    for cascade, why in bad:
        r = c.post("/api/games", json={"variant": "8p", "models": ["mock"], "human_seats": [], "cascade": cascade},
                   headers=h)
        assert r.status_code == 400 and why in r.text, r.text
    r = c.post("/api/games", json={"variant": "8p", "models": ["mock"], "human_seats": [], "seed": 11,
                                   "cascade": {"cheap": "mock-flaky", "strong": "mock-strong", "mode": "both"}},
               headers=h)
    assert r.status_code == 201, r.text
    game = r.json()
    assert game["cascade"]["mode"] == "both" and "级联" in game["model"]
    assert game["models"] == ["mock-flaky", "mock-strong"]
    # the counts live on the live table: read them while the game is still being played
    s = _wait(c, game["id"], lambda s: s["game"].get("cascade", {}).get("turns", 0) >= 12)
    st = s["game"]["cascade"]
    assert 0 < st["strong_turns"] <= st["turns"] and st["strong_calls"] <= st["calls"]
    assert st["escalations"].get("decisive")                      # votes and night actions start strong
    assert set(st["escalations"]) - {"decisive"}, st              # and the flaky cheap tier failed somewhere
