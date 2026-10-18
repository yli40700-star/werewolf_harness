"""Price per million tokens (input, cached input, output), USD.

Estimates for accounting and budgets, not invoices: gateways add margins and
prices change. Unknown models cost the default row. Update from the providers'
pages when it matters.
"""
from __future__ import annotations

DEFAULT = (1.0, 0.1, 4.0)
PRICES: dict[str, tuple[float, float, float]] = {
    "mock": (0.0, 0.0, 0.0), "script": (0.0, 0.0, 0.0),
    "sim": (1.0, 0.1, 4.0), "sim-strong": (3.0, 0.3, 15.0), "sim-cheap": (0.3, 0.03, 1.2),
}


def cost(model: str, input_tokens: int, cached: int, output: int) -> float:
    p_in, p_cached, p_out = PRICES.get(model, DEFAULT)
    return ((input_tokens - cached) * p_in + cached * p_cached + output * p_out) / 1_000_000
