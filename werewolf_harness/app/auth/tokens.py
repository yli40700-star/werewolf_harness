"""JWTs, two kinds.

    user token   who you are (a guest id), long-lived, sent as `Authorization: Bearer`
    seat token   which seat of which game you may act for; issued by /join after the
                 database accepted your claim, short-lived, required to read a seat's
                 private view or to act

Stateless on purpose: verifying a request is a signature check (HS256) plus an
expiry check, no session lookup. The cost is that a token cannot be revoked
before it expires -- which is why seat tokens are short.
"""
from __future__ import annotations

import time

import jwt

ALG = "HS256"


class AuthError(Exception):
    pass


def issue_user(secret: str, user_id: str, ttl_s: int) -> str:
    now = int(time.time())
    return jwt.encode({"sub": user_id, "kind": "user", "iat": now, "exp": now + ttl_s}, secret, algorithm=ALG)


def issue_seat(secret: str, user_id: str, game_id: str, seat: int, ttl_s: int) -> str:
    now = int(time.time())
    return jwt.encode({"sub": user_id, "kind": "seat", "game": game_id, "seat": seat, "iat": now, "exp": now + ttl_s},
                      secret, algorithm=ALG)


def verify(secret: str, token: str, kind: str | None = None) -> dict:
    try:
        claims = jwt.decode(token, secret, algorithms=[ALG], options={"require": ["exp", "sub"]})
    except jwt.PyJWTError as e:
        raise AuthError(str(e)) from e
    if kind and claims.get("kind") != kind:
        raise AuthError(f"expected a {kind} token")
    return claims


def seat_of(secret: str, token: str | None, game_id: str) -> int | None:
    """The seat this token may act for in this game, or None."""
    if not token:
        return None
    try:
        c = verify(secret, token, "seat")
    except AuthError:
        return None
    return int(c["seat"]) if c.get("game") == game_id else None
