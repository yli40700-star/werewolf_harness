# How prompts are written here

Three rules. They exist because the prompt is not scaffolding around the
measurement -- it *is* half of the measurement, and every sentence in it that
answers a question the experiment is supposed to ask has quietly replaced a
finding with an instruction.

---

## 1. Say what exists and what it costs. Never say which to pick.

Every sentence in a prompt has to be one of two things:

- **an affordance** -- what actions are open this turn, what the legal targets
  are, what tools can be called;
- **a consequence** -- what the *rules* guarantee will follow from each.

```
✅  self_detonate 立刻结束今天——今天不再投票、不放逐任何人，
    你当场出局、没有遗言，警徽若在你手上一并失效。
❌  局势不利时可以考虑自爆止损。
```

**The test is mechanical: read the sentence as a player on the other side. If
it becomes false, it is policy, not fact.** "Vote for the player you most
believe is a wolf" is true for a villager and an instruction to betray for a
wolf -- policy, cut it. "A tie exiles nobody" holds for everyone -- fact, keep
it.

That exact sentence shipped for the project's whole history and went to the
wolves verbatim, along with a last-words line asking for "the one thing the
living most need to hear" and a badge line warning that "a badge handed to a
wolf is worth more to them than your death cost them". Of 43 wolf last words,
seven owned up to being a wolf and five named a living teammate; one handed the
village three of them and signed off 村子加油. Whether a dying wolf betrays its
pack is a finding. It must not be an instruction.

## 2. A consequence is what the rules do, not what you think will happen.

The wolf's speak prompt used to say that claiming a god role costs you
"把自己变成焦点". That is a strategic guess wearing the clothes of a rule. What
the rules actually do is put two of that role on the table and force the village
to pick one, so that is what it says now.

If you cannot point at the line of engine code that makes a consequence
happen, it is not a consequence. It is advice.

## 3. Any sentence that states engine behaviour must be pinned by a test.

Hand-written claims about the implementation drift. This has now happened three
times:

- an 8-player rulebook -- "ids 1-8", two wolves, no sheriff, no guard -- was
  sent to every 12- and 15-player game ever run. A wolf that reasoned "there
  are only players 1-8" was written off as hallucinating; it was reading its
  own prompt.
- the output contract listed six information tools and followed them with
  "Only these tools exist. Do not invent tool names." A wolf is handed
  fourteen. `self_detonate`, `read_board`, `list_strategies` and `kill_options`
  were among the eight it was told did not exist -- and `self_detonate` has
  gone unused across 53 textbook openings.
- the rulebook said only that an exiled hunter shoots. The engine lets him
  shoot however he died, poison included -- a deliberate deviation, so that
  whether the shot lands does not publish the cause of death. The hunter was
  told a narrower rule than he held, and the witch was left believing that
  poisoning him was safe.

So: rules text is rendered from `Variant` and `rules.NIGHT_RULES`; the tool
list is rendered from the registry the turn actually hands to the model; and
`tests/test_rules_text.py` plays each night rule out against the real engine
and asserts the sentence is what happened. A caller that passes no registry
gets no tool enumeration at all -- a list that might be wrong is worse than no
list.

---

## Where the line is

**Pushed context must be neutral. Pulled context may have a view.**

The tactic catalogue in `strategies/playbook.json` is full of policy -- `when`,
`how`, `risks`, `abort_signal`. That is allowed, because a wolf *fetches* it
with `query_strategy` and the pack-chat prompt says in as many words that the
catalogue's timing may not match today's board and the board wins. Nothing from
it is injected into a turn the agent did not ask for.

The moment a tactic's advice is copied into a system message, it stops being
reference material and becomes an instruction, and the arm stops measuring
whether the model would have chosen it.

## What neutrality does and does not buy you

Removing the steering does not produce a neutral measurement. It produces a
measurement of the model's *disposition*, which is not the same as its ceiling.

This is not a guess. Shao et al. found GPT-4 naming random play as the optimal
rock-paper-scissors strategy in 100 trials out of 100 and then preferring rock
in all of them, and wolves systematically knifing player 0 -- reasoning
correct, action selection biased, and prompting does not move it
(arXiv:2310.18940). This harness reproduced the shape independently: stating
自爆's price on every single turn changed mentions from 0/234 to 1/227.

So a neutral prompt answers "what does this model do when nobody tells it what
to do". If you want the ceiling, write a steered version and report both, and
say which one every number came from.
