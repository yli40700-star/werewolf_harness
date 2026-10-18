"""Token-aware rate limits and per-game budgets, in Redis (shared by every worker).

Request counts are the wrong unit for a model API: one call is 2k tokens, the next
30k. So the bucket holds tokens:

  reserve   before the call: take (estimated prompt + max_tokens) from the bucket
            of the provider key (its TPM). Not enough -> wait (bounded) or give up.
  settle    after the call: put back what was reserved but not used
            (reserved - actual). A stream cut halfway settles with what arrived.

Reserving max_tokens is pessimistic -- it can block a call that would have fit --
but charging only afterwards lets a burst overspend the provider's quota and
turns into 429s for everyone. Settling afterwards recovers most of the slack.
"""
from __future__ import annotations

import time

import redis as redis_sync

# Refill continuously at rate tokens/s up to capacity; take n if available.
_TAKE = """
local cap = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local n = tonumber(ARGV[4])
local s = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(s[1]) or cap
local ts = tonumber(s[2]) or now
tokens = math.min(cap, tokens + (now - ts) / 1000 * rate)
if tokens < n then
  redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
  redis.call('PEXPIRE', KEYS[1], 120000)
  return math.ceil((n - tokens) / rate * 1000)
end
redis.call('HMSET', KEYS[1], 'tokens', tokens - n, 'ts', now)
redis.call('PEXPIRE', KEYS[1], 120000)
return 0
"""
_GIVE = """
local cap = tonumber(ARGV[1])
local s = redis.call('HGET', KEYS[1], 'tokens')
if s then redis.call('HSET', KEYS[1], 'tokens', math.min(cap, tonumber(s) + tonumber(ARGV[2]))) end
return 1
"""


class TokenLimiter:
    def __init__(self, r: redis_sync.Redis, tpm: dict[str, int], default_tpm: int = 2_000_000):
        self.r = r
        self.tpm = tpm
        self.default_tpm = default_tpm
        self._take = r.register_script(_TAKE)
        self._give = r.register_script(_GIVE)

    def _cap(self, key: str) -> int:
        return self.tpm.get(key, self.default_tpm)

    def reserve(self, key: str, n: int, max_wait_s: float = 10.0) -> bool:
        cap = self._cap(key)
        n = min(n, cap)
        deadline = time.monotonic() + max_wait_s
        while True:
            wait_ms = int(self._take(keys=[f"tpm:{key}"], args=[cap, cap / 60.0, int(time.time() * 1000), n]))
            if wait_ms == 0:
                return True
            if time.monotonic() + wait_ms / 1000 > deadline:
                return False
            time.sleep(min(wait_ms / 1000, 1.0))

    def settle(self, key: str, reserved: int, used: int, pipe=None) -> None:
        if reserved > used:
            self._give(keys=[f"tpm:{key}"], args=[self._cap(key), reserved - used], client=pipe)


class GameBudget:
    """Tokens spent per game, across every worker. INCRBY is atomic."""

    def __init__(self, r: redis_sync.Redis):
        self.r = r

    def spent(self, game_id: str) -> int:
        return int(self.r.get(f"budget:{game_id}") or 0)

    def add(self, game_id: str, tokens: int, pipe=None) -> None:
        p = pipe if pipe is not None else self.r.pipeline()
        p.incrby(f"budget:{game_id}", tokens)
        p.expire(f"budget:{game_id}", 7 * 24 * 3600)
        if pipe is None:
            p.execute()
