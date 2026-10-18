import json
import subprocess
import sys


def _play(tmp_path, *extra) -> dict:
    out = tmp_path / "g.json"
    subprocess.run([sys.executable, "-m", "werewolf_harness.cli", "play", "--variant", "8p", "--seed", "2001",
                    "--model", "mock", "--out", str(out), *extra], check=True, capture_output=True)
    return json.loads(out.read_text())


def test_wolves_get_their_own_model(tmp_path):
    d = _play(tmp_path, "--wolf-model", "mock-strong")
    roles = d["ground_truth"]["roles"]
    seen = {(roles[str(t["player_id"])] == "werewolf", t["model"]) for t in d["turns"]}
    assert seen == {(True, "mock-strong"), (False, "mock")}
    assert d["settings"]["wolf_model"] == "mock-strong"


def test_without_memory_no_note_is_kept(tmp_path):
    d = _play(tmp_path, "--no-memory")
    assert d["settings"]["memory"] is False
    assert not any(t.get("note_after") or t.get("note_before") for t in d["turns"])


def test_a_plain_game_keeps_its_old_settings(tmp_path):
    assert not {"wolf_model", "wolf_hint", "memory"} & set(_play(tmp_path)["settings"])


def test_the_wolf_hint_reaches_wolves_only():
    from werewolf_harness.players import GuardStack
    from werewolf_harness.players.llm.context import ContextBuilder
    from werewolf_harness.replay_llm import WOLF_HINTS
    cb = ContextBuilder(GuardStack())
    cb.wolf_hint = WOLF_HINTS["hantiao"]
    view = {"you": 1, "role": "werewolf", "team": "werewolf", "private": {}}
    assert "悍跳" in cb._identity(view)
    assert "悍跳" not in cb._identity({**view, "role": "villager", "team": "village"})
