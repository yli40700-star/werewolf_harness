"""Who drives a game: a Redis lease, backed by a Postgres fencing token.

    acquire   SET lease:{game} {node} NX PX ttl       -- only if nobody holds it
    renew     if GET == me then PEXPIRE               -- Lua: check-and-extend is one step
    release   if GET == me then DEL                   -- Lua: never delete someone else's lease

A lease alone is not enough. A node can be paused (GC, SIGSTOP, a stalled VM)
past its TTL, lose the lease without knowing, wake up and write. So winning the
lease is followed by `store.bump_fencing()`, and every journal write carries the
token it got; the database refuses a stale one (storage/db.py). The lease decides
who *should* write; the token decides who *can*.
"""
from __future__ import annotations

from redis.asyncio import Redis

_RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""
_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class Leases:
    def __init__(self, redis: Redis, node: str, ttl_ms: int):
        self.r = redis
        self.node = node
        self.ttl_ms = ttl_ms
        self._renew = redis.register_script(_RENEW)
        self._release = redis.register_script(_RELEASE)

    @staticmethod
    def key(game_id: str) -> str:
        return f"lease:{game_id}"

    async def acquire(self, game_id: str) -> bool:
        return bool(await self.r.set(self.key(game_id), self.node, nx=True, px=self.ttl_ms))

    async def acquire_many(self, game_ids: list[str]) -> list[str]:
        """acquire() for a batch in one pipeline (not a transaction: each SET NX stands
        alone, exactly as if sent one by one). Returns the ids we now hold."""
        if not game_ids:
            return []
        pipe = self.r.pipeline(transaction=False)
        for g in game_ids:
            pipe.set(self.key(g), self.node, nx=True, px=self.ttl_ms)
        return [g for g, ok in zip(game_ids, await pipe.execute(), strict=True) if ok]

    async def renew(self, game_id: str) -> bool:
        return bool(await self._renew(keys=[self.key(game_id)], args=[self.node, self.ttl_ms]))

    async def release(self, game_id: str) -> bool:
        return bool(await self._release(keys=[self.key(game_id)], args=[self.node]))

    async def holder(self, game_id: str) -> str | None:
        v = await self.r.get(self.key(game_id))
        return v.decode() if isinstance(v, bytes) else v

    async def holders(self, game_ids: list[str]) -> dict[str, str | None]:
        """Who holds each lease, in one MGET (the recovery scan asks about every live game)."""
        if not game_ids:
            return {}
        vals = await self.r.mget([self.key(g) for g in game_ids])
        return {g: (v.decode() if isinstance(v, bytes) else v) for g, v in zip(game_ids, vals, strict=True)}
