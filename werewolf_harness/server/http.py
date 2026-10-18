"""The service over HTTP, on the standard library only.

No framework: the harness's whole claim is that the contract is small, and a
web server that needs three packages to expose five methods undercuts it. One
thread per connection is the right shape here anyway -- a table is a handful of
watchers, and a long-poll that blocks a thread is exactly what a turn is.
"""
from __future__ import annotations

import json
import mimetypes
import os
import pathlib
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..envs import VARIANTS
from ..players.providers import PROFILES, offered

# Measured, not guessed: seed 101 (12p, static, six models) spent 2 367 608 tokens
# over 153 turns. Headroom for a dynamic pack and a longer game.
DEFAULT_MAX_TOKENS = 6_000_000
from .service import GameService

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"


def read_env(path: str = ".env") -> dict:
    out = {}
    for base in (pathlib.Path(path), pathlib.Path.cwd() / path):
        if base.exists():
            for line in base.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
            break
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_KEY_OPENAI_GROUP", "LLM_KEY_CLAUDE_GROUP", "LLM_KEY_GEMINI_GROUP"):
        if os.environ.get(k):
            out[k] = os.environ[k]
    return out


class Handler(BaseHTTPRequestHandler):
    service: GameService = None                     # set by serve()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):              # one line per request is noise here
        pass

    # ------------------------------------------------------------ plumbing
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------ routes
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        p = url.path
        if p in ("/", "/index.html"):
            return self._file("index.html")
        if p.startswith("/static/"):
            return self._file(p[len("/static/"):])
        if p == "/api/meta":
            return self._json({"variants": {k: {"seats": v.num_players, "sheriff": v.sheriff,
                                                "roles": v.role_counts} for k, v in VARIANTS.items()},
                               "models": offered(self.service.env_keys) + ["mock", "script"],
                               "profiles": PROFILES,
                               "guards": ["无", "L1", "L1+L2", "L1+L2+L3"]})
        if p == "/api/games":
            return self._json({"games": self.service.list()})
        parts = [x for x in p.split("/") if x]
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "games":
            g = self.service.get(parts[2])
            if g is None:
                return self._json({"error": "no such game"}, 404)
            seat = int(q["seat"][0]) if q.get("seat") else None
            if len(parts) == 3:
                return self._json(g.seat_state(seat))
            if parts[3] == "events":
                return self._events(g, int(q.get("after", ["0"])[0]))
            if parts[3] == "log":
                return self._json(g.full_log(seat))
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        parts = [x for x in url.path.split("/") if x]
        body = self._body()
        if url.path == "/api/games/resume":
            try:
                g = self.service.resume_saved(body.get("log") or {},
                                              max_tokens=body.get("max_tokens") or None)
            except (ValueError, KeyError) as e:
                return self._json({"error": str(e)}, 400)
            return self._json(g.summary(), 201)
        if url.path == "/api/games":
            variant = body.get("variant", "12p")
            if variant not in VARIANTS:
                return self._json({"error": "unknown variant"}, 400)
            try:
                g = self.service.create(
                    variant=variant,
                    model=body.get("models") or body.get("model", "mock"),
                    guard=body.get("guard", "L1+L2+L3"),
                    playbook=body.get("playbook", "static"),
                    human_seats=[int(s) for s in body.get("human_seats", [])],
                    turn_seconds=(None if body.get("turn_seconds") in (None, 0, "0") else float(body["turn_seconds"])),
                    seed=int(body.get("seed") or 0) or int.from_bytes(os.urandom(2), "big"),
                    # A 12p game measured 2.37M tokens end to end, and a dynamic
                    # pack talks every night on top of that. A cap under what the
                    # default variant actually costs does not protect anyone -- it
                    # just kills the game at round 4 and blames the models.
                    max_tokens=int(body.get("max_tokens") or DEFAULT_MAX_TOKENS),
                    xray=bool(body.get("xray")),
                )
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            return self._json(g.summary(), 201)
        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "games":
            g = self.service.get(parts[2])
            if g is None:
                return self._json({"error": "no such game"}, 404)
            if parts[3] == "resume":
                try:
                    return self._json(self.service.resume(
                        parts[2], max_tokens=int(body["max_tokens"]) if body.get("max_tokens") else None
                    ).summary(), 201)
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)
            if parts[3] == "start":
                g.start()
                return self._json(g.summary())
            if parts[3] == "act":
                return self._json(g.submit(int(body["seat"]), int(body["seq"]), body["action"]))
        return self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------ SSE
    def _events(self, game, after: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cursor = after
        try:
            while True:
                events = game.wait(cursor, timeout=20.0)
                for e in events:
                    self.wfile.write(f"data: {json.dumps(e.to_dict(), ensure_ascii=False, default=str)}\n\n".encode())
                    cursor = e.seq + 1
                self.wfile.write(b": ping\n\n")       # keeps proxies and the tab awake
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    # ------------------------------------------------------------ static
    def _file(self, name: str) -> None:
        path = (WEB / name).resolve()
        if not str(path).startswith(str(WEB.resolve())) or not path.exists():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript") or ctype.endswith("json"):
            ctype += "; charset=utf-8"
        self._send(200, path.read_bytes(), ctype)


def serve(host: str = "127.0.0.1", port: int = 8402, env_keys: dict | None = None) -> None:
    Handler.service = GameService(env_keys if env_keys is not None else read_env())
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"狼人杀 Arena — http://{host}:{port}   (Ctrl-C 停止)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")
