"""Every recorded case in tests/regressions/ still plays the same game.

A case is a game log, the model calls it made (cassette.jsonl.gz) and a case.json of
expectations; scripts/make_regression.py writes them. Replaying needs no model and no
network. A failure names the first thing that changed: a request the harness now
builds differently (a prompt change -- what to do then is in scripts/make_regression.py), a
journal step, a turn record, or an expectation.
"""
from __future__ import annotations

import os

import pytest

from werewolf_harness.scripts.make_regression import DEFAULT_DEST, run_case

CASES = sorted(d for d in (os.listdir(DEFAULT_DEST) if os.path.isdir(DEFAULT_DEST) else [])
               if os.path.exists(os.path.join(DEFAULT_DEST, d, "case.json")))


@pytest.mark.parametrize("name", CASES)
def test_regression_case_replays(name):
    problems = run_case(os.path.join(DEFAULT_DEST, name))
    assert not problems, "\n".join(problems)


def test_there_are_cases():
    assert len(CASES) >= 2
