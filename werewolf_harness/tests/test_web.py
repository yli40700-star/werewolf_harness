"""The browser half, checked the cheap way.

There is no JS runtime in this suite, so these are not behaviour tests. They
catch the one failure that has actually shipped twice: a helper that is called
but never defined, because a patch's anchor silently did not match. The symptom
in the browser is a tab that does nothing, with the data sitting right there in
the response.
"""
from __future__ import annotations

import pathlib
import re

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"

# Everything a page may call without declaring it.
GLOBALS = {
    "$", "el", "api", "fetch", "setTimeout", "setInterval", "clearInterval", "clearTimeout",
    "parseInt", "parseFloat", "String", "Number", "Boolean", "Object", "Array", "JSON", "Math",
    "Date", "Promise", "Error", "Set", "Map", "RegExp", "EventSource", "URLSearchParams", "WebSocket",
    "console", "document", "window", "location", "encodeURIComponent", "decodeURIComponent",
    "isNaN", "alert", "requestAnimationFrame", "structuredClone", "if", "for", "while", "switch",
    "catch", "return", "typeof", "function", "await", "new", "of", "in", "do", "else",
}
_DECL = re.compile(r"(?:function|class)\s+([A-Za-z_$][\w$]*)|"
                   r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=")
_CALL = re.compile(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(")
_ARROW = re.compile(r"\(([^()]*)\)\s*=>")
_PARAMS = re.compile(r"function\s+[A-Za-z_$][\w$]*\s*\(([^()]*)\)")
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
KEYWORDS = {"async", "var", "let", "const", "yield", "delete", "void", "instanceof"}


def _js() -> str:
    return (WEB / "app.js").read_text()


def _code(src: str) -> str:
    """Comments are prose; a word in one is not a call."""
    return _COMMENT.sub(" ", src)


def test_every_helper_the_page_calls_is_a_helper_the_page_has():
    src = _code(_js())
    declared = {a or b for a, b in _DECL.findall(src)}
    declared |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*=>", src))          # x => ...
    for params in _ARROW.findall(src) + _PARAMS.findall(src):              # (a, b) => ... / function f(a, b)
        declared |= set(re.findall(r"[A-Za-z_$][\w$]*", params))
    missing = sorted({n for n in _CALL.findall(src)} - declared - GLOBALS - KEYWORDS)
    assert not missing, f"app.js calls but never defines: {missing}"


def test_the_tabs_the_page_offers_all_have_a_renderer():
    src = _js()
    for tab, fn in [("tab-say", "renderFeed"), ("tab-vote", "renderVotes"),
                    ("tab-all", "renderLog"), ("tab-review", "renderReview")]:
        assert f'"#{tab}"' in src, f"{tab} is not wired"
        assert f"function {fn}" in src, f"{fn} is missing"
    html = (WEB / "index.html").read_text()
    assert 'id="tab-vote"' in html and 'id="tab-review"' in html
    # 思考过程 moved onto the seat: clicking a player is how you read his trace,
    # so the rail no longer carries a tab for it.
    assert 'id="tab-think"' not in html
    assert "function renderPicked" in src and "thinkStep" in src


def test_no_action_reaches_a_player_as_a_bare_id():
    """A tactic id is what the rules check, not what a person is shown."""
    from ..envs.werewolf import Werewolf
    src = _js()
    env = Werewolf("8p")
    names = set()
    for task in ("night", "speak", "vote", "last_words", "hunter_shoot", "badge", "review"):
        for role_seat in range(1, 9):
            try:
                spec = env.action_space(role_seat, task)
            except Exception:
                continue
            names |= {t.name for t in spec.tools}
    env.reset(5)
    for task in ("night", "speak", "vote", "review"):
        for seat in range(1, 9):
            try:
                names |= {t.name for t in env.action_space(seat, task).tools}
            except Exception:
                pass
    assert names, "no tools were collected"
    for n in sorted(names):
        assert f"  {n}:" in src or f"\n  {n}:" in src, f"{n} has no Chinese name in TOOL_CN"


def test_a_seat_is_how_you_read_a_players_thinking():
    """Clicking a seat opens that player's trace -- and only when we may see it."""
    src = _js()
    assert "function renderPicked" in src
    picked = src[src.index("function renderPicked"):]
    picked = picked[:picked.index("\nfunction ", 1)]
    assert "state.picked" in picked, "the panel does not read the clicked seat"
    assert "state.s.traces" in picked, "the panel does not gate on the reveal"
    assert "thinkStep(" in picked, "the panel does not render the steps"
    # the gate is a real early return, not a styling choice
    assert "if (!traces) return;" in picked
    assert "state.picked = state.picked === id" in src, "clicking a seat does not toggle it"
    # the old rail-side view is gone, not merely unreachable
    assert "function renderThink" not in src and "thinkSeat" not in src


def test_you_choose_which_round_you_are_reading():
    """第 1 轮 is most of what you reason from on 第 3 轮; it must stay reachable."""
    src = _js()
    assert "function roundPicker" in src
    assert "sayRound" in src and "voteRound" in src        # speeches and ballots both
    assert "sayRound: null" in src and "voteRound: null" in src   # they live in state
    feed = src[src.index("function renderFeed"):]
    feed = feed[:feed.index("\n/* One row of round chips")]
    assert "state.sayRound" in feed, "the feed ignores the chosen round"
    # a chosen round starts at its first line, not scrolled past its last
    assert "state.sayRound == null ? scroller.scrollHeight : 0" in feed


def test_the_seat_panel_cannot_stretch_the_table_out_of_shape():
    """A long reasoning chain used to grow the column and squash the felt."""
    css = (WEB / "style.css").read_text()
    block = css[css.index(".picked {"):]
    block = block[:block.index("}")]
    assert "position: absolute" in block            # out of the column's flow
    assert "max-height" in block and "overflow: auto" in block
    assert ".felt-wrap { position: relative; }" in css   # anchored to the table area
    assert '"shut"' in _js(), "the floating panel has no way to close it"
