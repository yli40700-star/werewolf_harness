"""The online service: a recoverable runtime around the Arena core.

Layering (outer depends on inner, never the reverse):

    api/      HTTP, WebSocket, SSE, auth          -- talks to runtime only
    runtime/  one actor per game, journal-first commit, leases, recovery
    llm/      the model-call layer shared by every game (queue, limits, breakers)
    storage/  Postgres (the source of truth) and Redis (leases, presence, limits)
    obs/      logs, metrics, traces

arena/, envs/ and players/ are the original library and are not modified in
meaning: world == f(seed, journal) is the property everything here leans on.
"""
