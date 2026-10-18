"""Reading and searching the playbook.

Retrieval here is keyword scoring over a small corpus, not embeddings. Nine
documents do not need a vector index, and an index would put a second model
between the agent and the thing being measured. What matters for the experiment
is that the agent has to *ask* for a strategy and gets back text it did not
write -- the ranking function is not the object of study.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).with_name("playbook.json")


@dataclass(frozen=True, eq=False)
class Strategy:
    id: str
    name: str
    en: str
    role: str
    summary: str
    when: tuple[str, ...] = ()
    how: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    needs_teammates: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    abort_signal: str = ""
    # Machine-checkable twins of the `when` clauses. See situation.py.
    preconditions: dict = field(default_factory=dict)

    def brief(self) -> str:
        """One line, for a list. The pack sees these before picking."""
        return f"{self.name}（{self.en}, {self.role}）— {self.summary}"

    def full(self) -> str:
        """The whole entry, for the seat that was assigned it.

        Deliberately verbose: this is the text the agent will be judged against
        on whether it actually followed the tactic it was handed.
        """
        lines = [f"### {self.name} · {self.en}", "", self.summary, ""]
        for label, items in (("适用时机", self.when), ("怎么打", self.how),
                             ("风险", self.risks)):
            if items:
                lines.append(f"**{label}**")
                lines += [f"- {x}" for x in items]
                lines.append("")
        if self.needs_teammates:
            lines.append(f"**需要队友配合**：{', '.join(self.needs_teammates)}")
        if self.conflicts_with:
            lines.append(f"**不能与之同时执行**：{', '.join(self.conflicts_with)}")
        if self.abort_signal:
            lines.append(f"**放弃信号**：{self.abort_signal}")
        return "\n".join(lines).strip()


@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    needs: int
    roles: tuple[str, ...]
    when: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "needs": self.needs,
                "roles": list(self.roles), "when": self.when}


@lru_cache(maxsize=1)
def load_playbook(path: str | None = None) -> tuple[dict[str, Strategy], tuple[Plan, ...]]:
    raw = json.loads(Path(path or _PATH).read_text(encoding="utf-8"))
    strategies = {}
    for s in raw["strategies"]:
        strategies[s["id"]] = Strategy(
            id=s["id"], name=s["name"], en=s["en"], role=s["role"],
            summary=s["summary"],
            when=tuple(s.get("when", ())), how=tuple(s.get("how", ())),
            risks=tuple(s.get("risks", ())),
            aliases=tuple(s.get("aliases", ())),
            needs_teammates=tuple(s.get("needs_teammates", ())),
            conflicts_with=tuple(s.get("conflicts_with", ())),
            abort_signal=s.get("abort_signal", ""),
            preconditions=s.get("preconditions", {}),
        )
    plans = tuple(
        Plan(id=p["id"], name=p["name"], needs=p["needs"],
             roles=tuple(p["roles"]), when=p.get("when", ""))
        for p in raw["assignments"]["plans"]
    )
    return strategies, plans


def get_strategy(sid: str) -> Strategy | None:
    return load_playbook()[0].get(sid)


def assignments() -> tuple[Plan, ...]:
    return load_playbook()[1]


def plan_for(pack_size: int) -> list[Plan]:
    """Plans a pack this size can actually staff.

    A plan that needs four seats is not an option for three wolves, and
    offering it would cost the pack a coordination turn discovering that.
    """
    return [p for p in assignments() if p.needs <= pack_size]


_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[a-zA-Z]{2,}")


def _tokens(text: str) -> list[str]:
    """Unigrams plus character bigrams for CJK.

    Single characters alone are not enough: 归票 (consolidate the vote against
    someone) and 抗推 (survive the vote landing on you) mean opposite things and
    share 票 through the surrounding text, so a unigram index ranks them almost
    identically. The bigram is what separates them, and it is the cheapest
    approximation of segmentation that does not pull in a tokeniser.
    """
    text = text.lower()
    out: list[str] = _LATIN.findall(text)
    for run in _CJK.findall(text):
        out.extend(run)                                   # unigrams
        out.extend(run[i:i + 2] for i in range(len(run) - 1))   # bigrams
    return out


def search_strategies(query: str, limit: int = 3) -> list[tuple[Strategy, float]]:
    """Rank the playbook against a situation description.

    Scored on the whole entry rather than the summary: a wolf asking "我被归票
    了怎么办" should reach 抗推 through its `when` clause, which is where that
    situation is actually described.
    """
    q = set(_tokens(query))
    if not q:
        return []

    # Field weights. `when` describes the situation a wolf is *in*, which is
    # what a query is, so it outranks the rest. Without this, "我被归票了怎么办"
    # retrieves 归票 -- the tactic of consolidating votes *against* someone --
    # over 抗推, which is what to do when the votes are landing on you. The two
    # share almost every character and mean opposite things, which is exactly
    # the failure mode a keyword index has and an embedding would not.
    FIELDS = ((lambda s: [s.name, s.en], 1.0),
              (lambda s: [s.summary], 1.0),
              # How a wolf describes its own spot. Weighted highest because a
              # query *is* a situation description, and because the tactic's
              # name is often not in it.
              (lambda s: list(s.aliases), 3.0),
              (lambda s: list(s.when), 2.5),
              (lambda s: list(s.how) + list(s.risks) + [s.abort_signal], 0.8))

    scored: list[tuple[Strategy, float]] = []
    for s in load_playbook()[0].values():
        score = 0.0
        hit_any = False
        for pick, weight in FIELDS:
            terms = _tokens(" ".join(pick(s)))
            if not terms:
                continue
            hits = sum(1 for t in terms if t in q)
            if hits:
                hit_any = True
                # Normalise by length so a long field does not win on volume.
                score += weight * hits / (len(terms) ** 0.5)
        if hit_any:
            scored.append((s, round(score, 4)))
    scored.sort(key=lambda x: (-x[1], x[0].id))
    return scored[:limit]


def strategy_ids() -> list[str]:
    """Every tactic id in the playbook, in file order."""
    return list(load_playbook()[0].keys())


def strategy_labels() -> dict[str, str]:
    """id -> what a person calls it, for anything a person has to read.

    The id is what the rules check and what the record stores; it is not a
    name. Wherever a human picks from these, they pick by this.
    """
    return {s.id: f"{s.name} — {s.summary.split('。')[0]}" for s in load_playbook()[0].values()}


def plan_labels() -> dict[str, str]:
    """Same, for the pack's whole-team assignments."""
    return {p.id: f"{p.name}（{p.needs} 狼）" for p in assignments()} | {"solo": "各自为战 — 不定分工"}
