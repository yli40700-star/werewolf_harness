"""Leaderboards in Redis sorted sets.

    lb:users              score = wins * 1e10 + (1e10 - time of the latest win)
                          -> more wins first; on equal wins, whoever got there first
    lb:models:{side}      wins per model on one side (village / werewolf)
    lb:models:games:{side}  games per model on that side (a hash), for win rates

A ZSET is a skip list plus a hash: ZINCRBY / ZREVRANK are O(log n), which is
why "my rank" and "the ten around me" stay cheap however many players there are.
"""
from __future__ import annotations

import time

from redis.asyncio import Redis

SCALE = 1e10


def _side(role: str) -> str:
    return "werewolf" if role in ("werewolf", "wolf", "white_wolf_king", "wolf_king") else "village"


class Leaderboard:
    def __init__(self, redis: Redis):
        self.r = redis

    async def record(self, *, winner: str, roles: dict, seat_users: dict[int, str], seat_models: dict[int, str]) -> None:
        now = time.time()
        pipe = self.r.pipeline(transaction=True)
        for seat, role in roles.items():
            seat = int(seat)
            side = _side(role)
            won = side == winner or (winner == "wolf" and side == "werewolf")
            if seat in seat_users:
                uid = seat_users[seat]
                pipe.hincrby("lb:users:games", uid, 1)
                if won:
                    pipe.hincrby("lb:users:wins", uid, 1)
            elif seat in seat_models:
                m = seat_models[seat]
                pipe.hincrby(f"lb:models:games:{side}", m, 1)
                pipe.zincrby(f"lb:models:{side}", 1 if won else 0, m)
        await pipe.execute()
        # user scores need the fresh win count, so a second round trip
        for uid in seat_users.values():
            wins = int(await self.r.hget("lb:users:wins", uid) or 0)
            await self.r.zadd("lb:users", {uid: wins * SCALE + (SCALE - now)})

    async def top_users(self, n: int = 20) -> list[dict]:
        rows = await self.r.zrevrange("lb:users", 0, n - 1, withscores=True)
        return [{"user": _s(u), "wins": int(sc // SCALE), "rank": i + 1} for i, (u, sc) in enumerate(rows)]

    async def around(self, uid: str, k: int = 5) -> dict:
        rank = await self.r.zrevrank("lb:users", uid)
        if rank is None:
            return {"rank": None, "around": []}
        lo = max(0, rank - k)
        rows = await self.r.zrevrange("lb:users", lo, rank + k, withscores=True)
        return {"rank": rank + 1, "around": [{"user": _s(u), "wins": int(sc // SCALE), "rank": lo + i + 1}
                                            for i, (u, sc) in enumerate(rows)]}

    async def models(self) -> list[dict]:
        out = []
        for side in ("village", "werewolf"):
            games = {_s(k): int(v) for k, v in (await self.r.hgetall(f"lb:models:games:{side}")).items()}
            wins = {_s(m): int(s) for m, s in await self.r.zrevrange(f"lb:models:{side}", 0, -1, withscores=True)}
            for m, g in games.items():
                out.append({"model": m, "side": side, "games": g, "wins": wins.get(m, 0),
                            "win_rate": round(wins.get(m, 0) / g, 3) if g else 0.0})
        return sorted(out, key=lambda r: (r["side"], -r["win_rate"], -r["games"]))


def _s(v) -> str:
    return v.decode() if isinstance(v, bytes) else v
