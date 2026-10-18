"""Parallel decide, serial commit: several seats think at once, the journal stays the one
the sequential loop would have written."""
from __future__ import annotations

import time

import pytest

from werewolf_harness.arena import Terminal, Turn, replay, run
from werewolf_harness.arena.driver import fingerprint, spec_key
from werewolf_harness.envs import VARIANTS, Werewolf
from werewolf_harness.players import LLMPlayer, MockClient, RandomPlayer
from werewolf_harness.players.providers.base import LLMResponse


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_an_open_turn_sees_at_its_turn_what_it_saw_when_it_opened(variant):
    """The property the driver relies on, checked on every step of many random games:
    from the moment a turn is declared open until it is played, nothing its actor sees
    or may do changes -- unless the turn stops being open (a wolf blew the day up)."""
    checked = 0
    for seed in range(1, 41):
        env = Werewolf(variant)
        players = {s: RandomPlayer(seed * 100 + s) for s in range(1, VARIANTS[variant].num_players + 1)}
        turn = env.reset(seed)
        first_seen: dict[Turn, tuple[str, str]] = {}
        while isinstance(turn, Turn):
            opened = env.open_turns()
            assert not opened or opened[0] == turn, (opened, turn)
            assert len({t.actor for t in opened}) == len(opened)
            for t in list(first_seen):
                if t not in opened and t != turn:
                    del first_seen[t]                       # no longer open: its answer would be dropped
            for t in opened:
                first_seen.setdefault(t, (fingerprint(env.observe(t.actor)), spec_key(env.action_space(t.actor, t.task))))
            if turn in first_seen:
                now = (fingerprint(env.observe(turn.actor)), spec_key(env.action_space(turn.actor, turn.task)))
                assert now == first_seen.pop(turn), f"seed {seed}: {turn} changed while it was open"
                checked += 1
            spec = env.action_space(turn.actor, turn.task)
            action = players[turn.actor].act(env.observe(turn.actor), turn.task, spec, None)
            while spec.check(action):
                action = players[turn.actor].act(env.observe(turn.actor), turn.task, spec, spec.check(action))
            turn = env.step(turn.actor, turn.task, action)
    assert checked > 100


def test_ballots_are_sealed_until_the_vote_is_over():
    env = Werewolf("8p")
    players = {s: RandomPlayer(s) for s in range(1, 9)}
    turn = env.reset(5)
    seen_open_vote = False
    while isinstance(turn, Turn):
        if turn.task == "vote" and env._votes.get(env._round):
            seen_open_vote = True
            assert env._round not in env.observe(turn.actor)["public"]["votes"]
        spec = env.action_space(turn.actor, turn.task)
        action = players[turn.actor].act(env.observe(turn.actor), turn.task, spec, None)
        if spec.check(action):
            action = spec.fallback
        turn = env.step(turn.actor, turn.task, action)
    assert seen_open_vote
    assert env.observe(1)["public"]["votes"]                   # and public once it is over


def test_the_night_opens_seer_guard_and_the_first_wolf_together():
    env = Werewolf("12p")
    turn = env.reset(2)
    while turn.task != "night":
        spec = env.action_space(turn.actor, turn.task)
        turn = env.step(turn.actor, turn.task, spec.fallback)
    roles = [env._roles[t.actor] for t in env.open_turns()]
    assert roles == ["seer", "guard", "werewolf"]
    assert all("night" in env.observe(t.actor) for t in env.open_turns())


class SlowMock(MockClient):
    """The mock seat with a real wait in it, so overlap shows up on the clock."""

    def __init__(self, seed, delay):
        super().__init__(seed=seed)
        self.delay = delay

    def chat(self, *a, **k) -> LLMResponse:
        time.sleep(self.delay)
        return super().chat(*a, **k)


def _llm_players(n, seed, delay, async_note):
    return {s: LLMPlayer(SlowMock(seed * 100 + s, delay), async_note=async_note) for s in range(1, n + 1)}


def test_a_parallel_game_replays_and_is_faster():
    seed, delay = 7, 0.03
    t = time.monotonic()
    serial = run(Werewolf("8p"), _llm_players(8, seed, delay, False), seed)
    t_serial = time.monotonic() - t
    t = time.monotonic()
    par = run(Werewolf("8p"), _llm_players(8, seed, delay, True), seed, parallel=8)
    t_par = time.monotonic() - t
    for ep in (serial, par):
        end = replay(Werewolf("8p"), seed, ep.journal)
        assert isinstance(end, Terminal) and end.outcome == ep.outcome
        assert len(ep.timings) == len(ep.journal)
    assert any(r["ahead"] for r in par.timings)
    assert t_par < t_serial * 0.6, (t_par, t_serial)


@pytest.mark.parametrize("seed", range(1, 16))
def test_random_seats_in_parallel_replay_exactly(seed):
    """Random seats blow days up mid-vote, which is what makes answers stale."""
    env = Werewolf("12p")
    ep = run(env, {s: RandomPlayer(seed * 100 + s) for s in range(1, 13)}, seed, parallel=6)
    end = replay(Werewolf("12p"), seed, ep.journal)
    assert isinstance(end, Terminal) and end.outcome == ep.outcome


def test_a_discarded_turn_leaves_the_seat_as_it_was():
    env = Werewolf("8p")
    turn = env.reset(3)
    p = LLMPlayer(MockClient(seed=1), async_note=True)
    spec = env.action_space(turn.actor, turn.task)
    p.act(env.observe(turn.actor), turn.task, spec, None)
    p.flush()
    before = (p.note, len(p.history), p.belief.snapshot())
    p.act(env.observe(turn.actor), turn.task, spec, None)
    p.discard_last()
    assert (p.note, len(p.history), p.belief.snapshot()) == before
