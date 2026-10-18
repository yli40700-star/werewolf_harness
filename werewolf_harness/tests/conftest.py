"""Test-wide guards.

A game writes its journal to `games/resume/` so a dead one can be resumed. That
is right in production and wrong in a test run: the suite would litter the
project's own data directory with throwaway games. Every test gets its own.
"""
from __future__ import annotations

import pytest

from ..server import service


@pytest.fixture(autouse=True)
def _save_somewhere_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "SAVE_DIR", tmp_path / "resume")
