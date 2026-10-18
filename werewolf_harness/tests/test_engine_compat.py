"""Engine versions: every stored journal replays under the rules it was written with.

The corpus (tests/corpus/, built by scripts/build_compat_corpus.py) is the CI hook: a
change to any rule branch an existing version uses fails here, per game, with the step
it bites at. A deliberate rule change goes behind a new version (RULE_CHANGES) instead,
and then the checker says exactly which old games it would have changed.
"""
from __future__ import annotations

import functools
import json
import pathlib

import pytest

from .. import cli
from ..envs import werewolf as ww
from ..envs import (BASELINE_ENGINE_VERSION, ENGINE_VERSION, MIN_ENGINE_VERSION, RULE_CHANGES, VARIANTS, Rules,
                    UnsupportedEngineVersion, Werewolf)
from ..scripts import compat_check as cc

CORPUS = pathlib.Path(__file__).parent / "corpus"
FILES = sorted(CORPUS.glob("*.json"))


@functools.cache
def _games() -> tuple[cc.StoredGame, ...]:
    return tuple(cc.load_logs([str(p) for p in FILES]))


# ---------------------------------------------------------------- the CI gate
@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_every_stored_journal_replays_identically_under_its_recorded_version(path):
    game = cc.load_logs([str(path)])[0]
    rep = cc.check_game(game, against=game.engine_version, fingerprint=True)
    assert rep.recorded == "identical", f"{path.name} under v{game.engine_version}: {rep.detail}"


def test_the_corpus_covers_every_supported_version_and_variant():
    games = _games()
    assert len(games) >= 50
    assert {g.engine_version for g in games} == set(range(MIN_ENGINE_VERSION, ENGINE_VERSION + 1))
    assert {g.variant for g in games} == set(VARIANTS)
    real = [g for g in games if g.id.startswith("real-")]
    assert len(real) == 34 and all(g.engine_version == BASELINE_ENGINE_VERSION for g in real)
    assert all(g.fingerprints and len(g.fingerprints) == len(g.journal) + 1 for g in games)


def test_real_logs_without_a_version_are_read_as_the_baseline():
    raw = json.loads(next(p for p in FILES if p.stem.startswith("real-")).read_text())
    assert "engine_version" not in raw
    assert cc.game_from_log(raw, "x").engine_version == BASELINE_ENGINE_VERSION == 1


# ---------------------------------------------------------------- the registry
def test_rules_flags_are_exactly_the_registered_changes():
    fields = set(Rules.__dataclass_fields__) - {"version"}
    assert fields == {c.flag for c in RULE_CHANGES}
    assert ENGINE_VERSION == max(c.introduced for c in RULE_CHANGES)
    for c in RULE_CHANGES:
        assert getattr(Rules.at(c.introduced - 1), c.flag) is False
        assert getattr(Rules.at(c.introduced), c.flag) is True
    assert Rules.at(2).changed_since(1) == [c for c in RULE_CHANGES if c.flag == "tie_revote"]


def test_a_version_outside_the_window_is_refused():
    for bad in (0, ENGINE_VERSION + 1, "2", True):
        with pytest.raises(UnsupportedEngineVersion):
            Werewolf("8p", engine_version=bad)
    assert Werewolf("8p").engine_version == ENGINE_VERSION
    assert Werewolf("8p", engine_version=1).rules.tie_revote is False


# ---------------------------------------------------------------- the checker finds exactly the affected games
def _ties_with_voters_left(game: cc.StoredGame) -> int | None:
    """Independent oracle, from the v1 world only: the journal index of the ballot that
    closed the first tied day vote with somebody left to revote -- the one place where
    v2 (tie_revote) asks for a different turn. None if the game never had one."""
    env = Werewolf(game.variant, playbook_mode=game.playbook, engine_version=1)
    env.reset(game.seed)
    for i, rec in enumerate(game.journal):
        rnd = env._round
        env.step(rec.actor, rec.task, rec.action)
        if rec.task != "vote" or rnd not in env._exiles or env._exiles[rnd] is not None or env._detonations and \
                env._detonations[-1]["round"] == rnd:
            continue
        ballots, weights = env._votes[rnd], env._vote_weights
        tally: dict[int, float] = {}
        for voter, t in ballots.items():
            tally[t] = tally.get(t, 0.0) + weights.get(voter, 1.0)
        top = max(tally.values())
        tied = [t for t, n in tally.items() if n == top]
        alive_then = set(env._alive)
        if len(tied) > 1 and alive_then - set(tied):
            return i
    return None


def test_the_checker_lists_exactly_the_games_the_tie_rule_changes_and_where():
    games = [g for g in _games() if g.engine_version == 1]
    reports = {r.id: r for r in cc.check(list(games), against=2)}
    oracle = {g.id: _ties_with_voters_left(g) for g in games}
    affected = {gid for gid, i in oracle.items() if i is not None}
    assert affected and len(affected) < len(games)                  # some, not all
    assert {gid for gid, r in reports.items() if r.current != "identical"} == affected
    for gid in affected:
        d = reports[gid].detail["current"]
        assert reports[gid].current == "diverged"
        assert d["step"] == oracle[gid] + 1                         # the row after the tied ballot
        assert d["rule_changes"] == ["tie_revote"]
        assert d["got"].endswith("/ vote")                          # v2 wants the revote
        assert "revote" in d["first_state_diff"]["keys"]
    assert all(r.recorded == "identical" for r in reports.values())  # and v1 still recovers them all


def test_v2_games_with_a_revote_do_not_replay_under_v1():
    games = [g for g in _games() if g.engine_version == 2]
    revoted = [g for g in games if cc.replay_under(g, 2).states[-1].get("tie_votes")]
    assert revoted, "the corpus should pin the v2 branch with at least one revote"
    for g in revoted:
        assert cc.check_game(g, against=1).current == "diverged"


# ---------------------------------------------------------------- fingerprint mode
def test_fingerprint_mode_catches_a_change_that_keeps_every_turn_and_the_winner(monkeypatch):
    """An accidental edit of shared code: the review stores a shortened remark. Every
    journal row is still accepted and every outcome is the same -- only the per-step
    public state shows that the world moved."""
    games = [g for g in _games() if any(r.task == "review" and r.action["name"] == "review_say" for r in g.journal)][:3]
    assert games
    real = ww.Werewolf._apply_review_say

    def clipped(self, actor, args):
        real(self, actor, {**args, "content": args["content"][:5]})
    monkeypatch.setattr(ww.Werewolf, "_apply_review_say", clipped)
    for g in games:
        assert cc.check_game(g, against=g.engine_version).recorded == "identical"       # outcome only: blind
        rep = cc.check_game(g, against=g.engine_version, fingerprint=True)
        first_say = next(i for i, r in enumerate(g.journal) if r.task == "review" and r.action["name"] == "review_say")
        assert rep.recorded == "diverged" and rep.detail["recorded"]["step"] == first_say
        assert rep.broken


def test_cli_reports_and_exit_code(tmp_path, capsys):
    out = tmp_path / "r.json"
    code = cc.main([str(p) for p in FILES if p.stem.startswith("real-")] + ["--json", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "当前 v2 回放：一致 30 / 分叉 4 / 崩溃 0" in text
    assert json.loads(out.read_text())["summary"]["recorded"] == {"identical": 34, "diverged": 0, "crashed": 0}
    assert cc.main([str(p) for p in FILES if p.stem.startswith("real-")] + ["--strict"]) == 1


# ---------------------------------------------------------------- cli play / replay record and honour the version
def test_cli_play_records_the_version_and_replay_uses_it(tmp_path, capsys):
    log = tmp_path / "g.json"
    assert cli.main(["play", "--variant", "8p", "--seed", "3", "--model", "mock", "--parallel", "1",
                     "--engine-version", "1", "--out", str(log)]) == 0
    saved = json.loads(log.read_text())
    assert saved["engine_version"] == 1 and saved["view_revision"] == ww.VIEW_REVISION
    assert cli.main(["replay", str(log)]) == 0
    assert "engine v1" in capsys.readouterr().out


def test_cli_replay_of_an_unversioned_log_with_a_tie_uses_the_baseline(tmp_path, capsys):
    """real-base-g1007 ties on day one; under v2 it would diverge, under its baseline it must not."""
    raw = json.loads((CORPUS / "real-base-g1007.json").read_text())
    p = tmp_path / "old.json"
    p.write_text(json.dumps(raw))
    assert cli.main(["replay", str(p)]) == 0
    assert "engine v1" in capsys.readouterr().out
    end = cc.replay_under(cc.game_from_log(raw, "g1007"), 1)
    assert end.status == "ok" and end.outcome["winner"] == raw["outcome"]["winner"]


# ---------------------------------------------------------------- table config
def test_table_config_pins_the_version_and_old_configs_read_as_baseline():
    from ..app.runtime.table import Table, TableConfig
    t = Table("g1", 1, TableConfig(variant="8p"))
    assert t.cfg.engine_version == ENGINE_VERSION == t.env.engine_version
    assert t.cfg.to_dict()["engine_version"] == ENGINE_VERSION
    assert TableConfig.from_dict({"variant": "8p"}).engine_version == BASELINE_ENGINE_VERSION
    old = Table("g2", 1, TableConfig.from_dict({"variant": "8p"}))
    assert old.env.rules.tie_revote is False and old.summary()["engine_version"] == 1


def test_world_state_and_public_state_agree_across_versions_until_the_rule_fires():
    g = next(x for x in _games() if x.id == "real-base-g1007.json")
    a, b = cc.replay_under(g, 1), cc.replay_under(g, 2)
    assert a.status == "ok" and b.status == "diverged" and b.step == 18
    assert a.hashes[:18] == b.hashes[:18] and a.worlds[:18] == b.worlds[:18]
