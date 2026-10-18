// The same scenario for k6 (https://k6.io): k6 run -e URL=http://127.0.0.1:8402 -e TABLES=200 loadtest/ws_bots.k6.js
import http from "k6/http";
import ws from "k6/ws";
import { Trend, Counter } from "k6/metrics";

const push = new Trend("push_ms"), ack = new Trend("ack_ms"), acts = new Counter("acts");
const URL = __ENV.URL || "http://127.0.0.1:8402";
export const options = { scenarios: { tables: { executor: "per-vu-iterations", vus: +(__ENV.TABLES || 100), iterations: 1, maxDuration: "3m" } } };

export default function () {
  const j = (r) => JSON.parse(r.body);
  const tok = j(http.post(`${URL}/api/auth/guest`, "{}", { headers: { "Content-Type": "application/json" } })).token;
  const H = { headers: { "Content-Type": "application/json", Authorization: `Bearer ${tok}` } };
  const g = j(http.post(`${URL}/api/games`, JSON.stringify({ variant: "8p", models: ["sim-cheap"], human_seats: [1], turn_seconds: 60 }), H));
  const seat = j(http.post(`${URL}/api/games/${g.id}/join`, JSON.stringify({ seat: 1 }), H)).seat_token;
  const S = { headers: { "Content-Type": "application/json", Authorization: `Bearer ${seat}` } };
  http.post(`${URL}/api/games/${g.id}/start`, "{}", S);
  ws.connect(`${URL.replace("http", "ws")}/ws/games/${g.id}?token=${seat}`, {}, (sock) => {
    const sent = {};
    sock.setInterval(() => sock.send(JSON.stringify({ op: "ping" })), 25000);
    sock.on("message", (raw) => {
      const m = JSON.parse(raw);
      if (m.op === "event") {
        push.add(Date.now() - m.ts * 1000);
        if (m.kind === "end") { sock.close(); return; }
        const s = j(http.get(`${URL}/api/games/${g.id}?seat=1`, S));
        if (s.me && s.me.waiting) {
          const rid = `${__VU}-${Date.now()}`; sent[rid] = Date.now();
          sock.send(JSON.stringify({ op: "act", seq: s.me.turn.seq, action: s.me.turn.fallback, request_id: rid }));
        }
      } else if (m.op === "ack") { ack.add(Date.now() - sent[m.request_id]); acts.add(1); }
    });
    sock.setTimeout(() => sock.close(), 170000);
  });
}
