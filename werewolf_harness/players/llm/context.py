"""Prompt assembly for the LLM seat.

Everything the model reads is built here from three sources, in this order of
authority: the environment's rules text (pushed, neutral), the view (facts the
seat is entitled to), and the ActionSpec (what it may do right now). Nothing in
this file decides for the agent; see PROMPTS.md for the contract.
"""
from __future__ import annotations

from ..guard import GuardStack, verify_mod

MAX_OBSERVATION_CHARS = 2400
MAX_STEPS_IN_CONTEXT = 6

OUTPUT_CONTRACT = """\
Work in short steps. On each step, either call one information tool or end your
turn with a terminal action.
{tool_lines}
- You have at most {max_steps} steps. If you reach the last step without a
  terminal action, one will be forced from you.
- Do not invent tool names.

Speeches from earlier rounds are NOT shown to you. Use query_history to recall
them."""

OUTPUT_CONTRACT_STATIC = """\
Work in short steps. On each step, either call one information tool or end your
turn with a terminal action. The tools this turn offers are listed under THIS TURN.
- You have at most {max_steps} steps. If you reach the last step without a
  terminal action, one will be forced from you.
- Do not invent tool names.

Speeches from earlier rounds are NOT shown to you. Use query_history to recall
them."""

# What the agents speak at the table. The prompts themselves stay in English (they are
# pinned by tests and shared across seats for the prompt cache); only the output language moves.
LANG_LINES = {
    "zh": "Write everything you say at the table -- speeches, last words, pack chat, and every `thought` -- "
          "in Simplified Chinese (简体中文), the way a Chinese player talks. Tool names, argument names and "
          "the JSON envelope stay exactly as specified, in English.",
    "en": "",
}

_ACTION_CN = {"night_check": "验人", "night_kill": "刀人", "night_save": "用解药救", "night_poison": "用毒药毒",
              "night_protect": "守护", "night_skip": "夜里空过", "speak": "发言", "vote": "投票",
              "last_words": "留遗言", "campaign_run": "竞选警长", "campaign_pass": "退出竞选",
              "campaign_vote": "投警长", "hunter_shoot": "开枪", "hunter_hold": "不开枪",
              "badge_transfer": "移交警徽", "badge_tear": "撕警徽", "self_detonate": "自爆", "pack_plan": "定狼队分工",
              "review_say": "复盘发言", "review_pass": "复盘过"}

_ACTION_ROLE_CN = {"werewolf": "狼人", "seer": "预言家", "witch": "女巫", "hunter": "猎人",
                   "guard": "守卫", "villager": "村民"}
_TRUE_CAUSE = {"werewolf": "被狼刀", "witch": "被女巫毒", "vote": "被票出", "hunter": "被猎人枪带走", "detonate": "自爆"}

_SPEECH_LABEL = {"campaign": ", running for sheriff",
                 "last_words": ", DEAD -- these are their last words and they cannot be questioned about them"}


class ContextBuilder:
    def __init__(self, guard: GuardStack, max_steps: int = 8, max_observation_chars: int = MAX_OBSERVATION_CHARS,
                 max_steps_in_context: int = MAX_STEPS_IN_CONTEXT, anonymise_speakers: bool = False,
                 cache_layout: bool = True, lang: str = "zh"):
        self.guard = guard
        self.lang_line = LANG_LINES.get(lang, "")
        self.max_steps = max_steps
        self.max_observation_chars = max_observation_chars
        self.max_steps_in_context = max_steps_in_context
        self.anonymise_speakers = anonymise_speakers
        # Order the system message by how often each part changes, so the part every
        # seat shares (rules, guard preamble, output contract) is a byte-identical
        # prefix and hits the provider's prompt cache. The old order put the seat's
        # private identity second, which made the shared prefix end after the rules.
        self.cache_layout = cache_layout
        self.wolf_hint = ""              # an extra line for werewolves only (an experiment's directive)

    # ---------------------------------------------------------------- system
    def system_message(self, view: dict, terminal_names: list[str], info_names: list[str]) -> str:
        preamble = self.guard.system_preamble()
        if not self.cache_layout:
            parts = [view["rules"], "", self._identity(view), ""]
            if preamble:
                parts += [preamble, ""]
            parts.append(OUTPUT_CONTRACT.format(max_steps=self.max_steps, tool_lines=_tool_lines(info_names, terminal_names)))
            parts += ["", f"You are player {view['you']}. Speak in the first person, in at most 4 sentences, like a player at the table."]
            if self.lang_line:
                parts += [self.lang_line]
            return "\n".join(parts)
        # shared by every seat and every turn of this table
        parts = [view["rules"], ""]
        if preamble:
            parts += [preamble, ""]
        parts += [OUTPUT_CONTRACT_STATIC.format(max_steps=self.max_steps), "",
                  "Speak in the first person, in at most 4 sentences, like a player at the table.", ""]
        if self.lang_line:
            parts += [self.lang_line, ""]
        # this seat, this turn
        parts += [self._identity(view), "", "THIS TURN:", _tool_lines(info_names, terminal_names), "",
                  f"You are player {view['you']}."]
        return "\n".join(parts)

    def _identity(self, view: dict) -> str:
        priv = view["private"]
        lines = ["YOUR IDENTITY (private, never shown to anyone else):",
                 f"  You are player {view['you']}, role: {view['role']}, team: {view['team']}."]
        if "fellow_wolves" in priv:
            others = priv["fellow_wolves"]
            lines.append(f"  Your fellow wolves: {others or 'none left alive'}. Never reveal this.")
        if self.wolf_hint and view["role"] == "werewolf":
            lines.append("  " + self.wolf_hint)
        if priv.get("my_tactic"):
            lines.append(f"  The pack's plan is {priv.get('pack_plan')!r} and your assigned tactic is {priv['my_tactic']!r}. "
                         "Call my_assignment to read it in full, and read_board to see whether it still applies.")
        elif priv.get("pack_roles"):
            lines.append(f"  The pack's assignment: {priv['pack_roles']}.")
        if priv.get("pack_vote_target"):
            t = priv["pack_vote_target"]
            lines.append(f"  狼队今天约定的投票目标是 {t} 号。这是队里定下的约定，不是引擎的约束：跟或不跟都能提交。"
                         "每一轮的完整票面（谁投了谁）对所有人公开，白天任何人都可以用 query_votes 调出来。")
        if priv.get("wasted_knives"):
            k = "、".join(f"第{x['round']}夜刀{x['target']}号" for x in priv["wasted_knives"])
            lines.append(f"  这几刀落空了：{k}。刀空意味着守卫守了那个位置，或者女巫把解药花在了他身上。")
        if priv.get("pack_chat"):
            lines.append("  What the pack has said in the night channel:")
            for e in priv["pack_chat"]:
                wants = f" [wants {e['wants']}]" if e.get("wants") else ""
                lines.append(f"    night {e['round']}, player {e['player_id']}{wants}: {(e.get('content') or '（沉默）').replace(chr(10), ' ')}")
        if priv.get("my_actions"):
            lines.append("  Your own record — what you have done, which only you remember:")
            for a in priv["my_actions"][-14:]:
                lines.append("    " + _render_action(a))
        if "checks" in priv:
            if priv["checks"]:
                for c in priv["checks"]:
                    lines.append(f"  Night {c['round']}: you checked player {c['target']} -> {'WEREWOLF' if c['is_wolf'] else 'not a werewolf'}.")
            else:
                lines.append("  You have not checked anyone yet.")
        if "antidote_available" in priv:
            lines.append(f"  Antidote left: {priv['antidote_available']}, poison left: {priv['poison_available']}.")
        return "\n".join(lines)

    # ---------------------------------------------------------------- situation
    def situation_message(self, view: dict, belief, task: str, spec, note: str = "") -> tuple[str, set[int], list[dict]]:
        """(text, directive targets seen in others' speech, guard detections).

        `note` is the seat's own message to itself from its previous turn; it is
        rendered under the beliefs and comes from the player, never the view.
        """
        if task == "night":
            return self._night_message(view, belief, spec, note), set(), []
        if task == "review":
            return self._review_message(view, spec)
        pub = view["public"]
        rnd = view["round"]
        directives: set[int] = set()
        detections: list[dict] = []
        lines = [f"=== ROUND {rnd}, {view['phase'].replace('_', ' ')} ===", f"Alive: {pub['alive']}"]
        if pub["dead"]:
            lines.append("Dead: " + ", ".join(f"player {d['player_id']} (round {d['round']}, {_dead_label(d['cause'])})" for d in pub["dead"]))
        for r in sorted(pub["exiles"]):
            counts = pub["vote_counts"].get(r, {})
            ex = pub["exiles"][r]
            lines.append(f"Round {r} vote tally: {counts} -> " + (f"exiled player {ex}" if ex is not None else "nobody exiled"))
        lines += ["", "--- what has been said this round ---"]
        today = [s for s in pub["speeches"] if s["round"] == rnd]
        if not today:
            lines.append("(nobody has spoken yet this round)")
        seen: dict[str, int] = {}
        for s in today:
            seen[s["kind"]] = seen.get(s["kind"], 0) + 1
            label = _SPEECH_LABEL.get(s["kind"], "")
            if s["player_id"] == view["you"]:
                lines.append(f"[you, position {seen[s['kind']]}{label}] {s['content']}")
                continue
            who = "a player" if self.anonymise_speakers else f"player {s['player_id']}"
            directives |= verify_mod.directive_targets(s["content"])
            cleaned, dets = self.guard.clean_incoming(s["content"], source=f"player_{s['player_id']}", kind=s["kind"], round_no=rnd)
            detections.extend(dets)
            lines.append(f"[{who}, position {seen[s['kind']]}{label}]")
            lines.append(cleaned)
        lines += ["", "--- your recorded beliefs ---", belief.summarise(pub["alive"]), *_note_block(note), "", self._task_line(task, view, spec)]
        return "\n".join(lines), directives, detections

    def _night_message(self, view: dict, belief, spec, note: str = "") -> str:
        pub = view["public"]
        night = view.get("night", {})
        lines = [f"=== NIGHT {view['round']} ===", f"Alive: {pub['alive']}"]
        if pub["dead"]:
            lines.append("Dead: " + ", ".join(f"player {d['player_id']} (round {d['round']}, {_dead_label(d['cause'])})" for d in pub["dead"]))
        lines += ["", "--- your night turn ---"]
        if "victims_tonight" in night:
            lines.append(f"The wolves attacked: {night['victims_tonight'] or 'nobody yet'}. "
                         f"Antidote left: {night['antidote_available']}, poison left: {night['poison_available']}.")
        if night.get("pack_votes_so_far"):
            lines.append(f"Your packmates have already named: {night['pack_votes_so_far']}. The majority decides; a tie goes to whoever named first.")
        lines.append(f"You may call: {', '.join(t.name for t in spec.tools)}")
        lines += ["", "--- your recorded beliefs ---", belief.summarise(pub["alive"]), *_note_block(note), "",
                  "YOUR TASK [night]: decide your night action and call it. Nobody sees this turn, and nothing you do here "
                  "is announced -- the table learns only who died, never how."]
        return "\n".join(lines)

    def _review_message(self, view: dict, spec) -> tuple[str, set[int], list[dict]]:
        """The 复盘 turn: the game is over, the result is fixed, and the table is open.

        Everything here comes out of ``view["review"]`` -- the handout the environment
        only builds while the phase is ``review``. Nothing is reconstructed locally, so a
        seat can never be shown more here than ``observe`` decided to hand it.
        """
        r = view.get("review") or {}
        me = view["you"]
        directives: set[int] = set()
        detections: list[dict] = []
        won = "狼人阵营" if r.get("winner") == "werewolf" else "好人阵营"
        lines = [f"=== 复盘 (round {r.get('round')} of {r.get('rounds_max')}) ===",
                 f"这局打完了：{won}获胜，一共走了 {r.get('rounds_played')} 轮。结果已经定死，接下来说什么都不会改变它。",
                 "", "--- 全场底牌 ---"]
        roles = r.get("roles") or {}
        survivors = set(r.get("survivors") or [])
        for seat in sorted(roles, key=int):
            tag = "存活" if int(seat) in survivors else "出局"
            mine = "（你）" if int(seat) == me else ""
            lines.append(f"  {seat} 号：{_ACTION_ROLE_CN.get(roles[seat], roles[seat])}{mine} · {tag}")
        if r.get("deaths"):
            lines += ["", "--- 出局顺序与真实死因 ---"]
            for d in r["deaths"]:
                lines.append(f"  第 {d['round']} 轮：{d['player_id']} 号 {_TRUE_CAUSE.get(d['cause'], d['cause'])}")
        if r.get("night_actions"):
            lines += ["", "--- 每一夜谁做了什么 ---"]
            for a in r["night_actions"]:
                tgt = f" → {a['target']} 号" if a.get("target") is not None else ""
                lines.append(f"  第 {a['round']} 夜：{a['player_id']} 号 {_ACTION_CN.get(a['action'], a['action'])}{tgt}（{a.get('outcome')}）")
        if r.get("seer_checks"):
            checks = "，".join(f"第{c['round']}夜验 {c['target']} 号 → {'狼' if c['is_wolf'] else '好人'}" for c in r["seer_checks"])
            lines += ["", f"预言家的验人结果：{checks}"]
        if r.get("wasted_knives"):
            lines.append("落空的刀：" + "，".join(f"第{k['round']}夜刀 {k['target']} 号" for k in r["wasted_knives"]))
        if r.get("pack_plans"):
            lines += ["", "--- 狼队的分工 ---"]
            for h in r["pack_plans"]:
                lines.append(f"  第 {h['round']} 夜（{h['by']} 号提交）：方案 {h['plan']}，" +
                             "，".join(f"{w}号 {t}" for w, t in h["roles"].items()))
        if r.get("pack_chat"):
            lines += ["", "--- 狼人频道里说过的话 ---"]
            for e in r["pack_chat"]:
                cleaned, dets = self.guard.clean_incoming(e.get("content") or "", source=f"pack_chat_player_{e['player_id']}",
                                                          kind="pack_chat", round_no=e.get("round") or 0)
                detections.extend(dets)
                lines.append(f"  第 {e['round']} 夜，{e['player_id']} 号：{cleaned}")
        for rnd in sorted(r.get("exiles") or {}):
            ex = (r["exiles"] or {}).get(rnd)
            ballots = (r.get("votes") or {}).get(rnd) or {}
            tally = "，".join(f"{v}号→{t}号" for v, t in sorted(ballots.items())) or "无人投票"
            lines.append(f"第 {rnd} 轮票面：{tally} → " + (f"{ex} 号出局" if ex is not None else "平票，无人出局"))
        said = r.get("said") or []
        lines += ["", "--- 复盘已经说过的话 ---"]
        if not said:
            lines.append("（还没有人开口）")
        for e in said:
            if e["player_id"] == me:
                lines.append(f"[你] {e['content']}")
                continue
            who = "某位玩家" if self.anonymise_speakers else f"{e['player_id']} 号"
            directives |= verify_mod.directive_targets(e["content"])
            cleaned, dets = self.guard.clean_incoming(e["content"], source=f"review_player_{e['player_id']}",
                                                      kind="review", round_no=e.get("round") or 0)
            detections.extend(dets)
            lines.append(f"[{who}]")
            lines.append(cleaned)
        lines += ["", self._task_line("review", view, spec)]
        return "\n".join(lines), directives, detections

    # ---------------------------------------------------------------- task lines
    def _task_line(self, task: str, view: dict, spec) -> str:
        body = self._task_body(task, view, spec)
        return body.replace("YOUR TASK", f"YOUR TASK [{task}]", 1)

    def _task_body(self, task: str, view: dict, spec) -> str:
        pub, priv = view["public"], view["private"]
        badge = pub.get("sheriff")
        if task == "speak":
            pos, total = view.get("speech_position"), view.get("speakers_total")
            office = ""
            if badge == view["you"]:
                office = " You hold the badge: you speak first each day and your ballot counts 1.5."
            elif badge:
                office = f" Player {badge} holds the badge, speaks first and casts 1.5 votes."
            extra = ""
            if view["role"] == "werewolf":
                extra = (" 你这一轮可用的动作还包括：`self_detonate` 立刻结束今天——今天不再投票、不放逐任何人，你当场出局、没有遗言，"
                         "警徽若在你手上一并失效；跳一个已经有人跳过的神职，结果是场上同时存在两个该神职，全场必须二选一；"
                         "公开指认自己的队友，结果是他进入被投名单而你不在其中。")
            return (f"YOUR TASK: it is your turn to speak ({pos} of {total} this round). Everyone speaking after you will hear this; "
                    "everyone before you has already spoken. What you say enters the public record and can be pulled up by anyone "
                    "later with `query_history`. Investigate or update your beliefs if you want to, then call `speak`." + office + extra)
        if task == "pack_chat":
            first = not priv.get("pack_roles")
            head = "YOUR TASK: 现在是狼人时间，只有你的狼队友听得见，白天没有人能查到这里说过什么。这是你们今晚唯一能对齐的机会。"
            middle = "今晚是第一夜，明天的分工还没有定。" if first else "分工已经定过一次了，它会一直有效直到有人改它。"
            body = ("可以取用的材料：`read_board` 报场上的事实（谁跳了身份、谁在挨点名、狼队还剩几只），只报事实不给建议；"
                    "`list_strategies` 是完整战术清单，每条写了适用时机和放弃信号；`query_strategy` 看某条全文。"
                    "攻略写的时机未必对得上今天的局面，对不上时以你看到的局面为准。然后 `pack_say` 发言：说什么、要不要说、"
                    "说多少都由你决定，想要某个战术就把它填进 `tactic` 参数——填了队友就看得到，不填他们只看得到你的正文；"
                    "如果这一轮你没有要补充的，就 `pack_pass`。全队在同一轮里都 pass，狼人时间就结束；"
                    "pass 过之后队友又说了新东西，你还会再轮到，那时想说仍然可以说。")
            tail = "" if first else "改与不改都要有人开口：没有人提出，现有分工原样进入明天。"
            return head + middle + body + tail
        if task == "pack_plan":
            revising = bool(priv.get("pack_roles"))
            staffable = priv.get("staffable_plans") or []
            fits = f"当前人数能提交的方案 id：{staffable}。" if staffable else ""
            head = ("YOUR TASK: 你是狼队位置最靠前的活狼，今晚这一版分工由你提交。" if revising
                    else "YOUR TASK: 你是本局狼队里位置最靠前的活狼，由你提交第一版分工。")
            common = ("队友刚在狼人频道表过态，他们要什么你都看到了。`read_board` 给事实，`list_strategies` 给全部战术，"
                      f"`list_plans` 给现成的方案组合。{fits}`kill_options` 列出每个活着的非狼座位和场上对它的公开标记"
                      "（跳过什么身份、被谁发过金水或报过查杀、有没有领过票），并标出哪几刀落空过——落空表示那一夜守卫守在那里或女巫救了他。"
                      "提交时 `pack_plan` 要一个方案 id 和每只活狼的战术，你不必沿用方案自带的默认分工，可以逐个指派。"
                      "`vote_target` 是可选的：填了，队友白天会看到这个约定；不填，各自独立投。四狼同票在公开票面上是所有人都看得见的。")
            return head + common + ("维持原分工也需要提交一次——不提交，这一夜就没有分工。" if revising else "")
        if task == "review":
            r = view.get("review") or {}
            return ("YOUR TASK: 这一局已经结束，输赢已经定死，这一段说什么都不会改变结果。上面是这局的全部底牌和全部记录——"
                    "谁是什么身份、每一夜谁动了谁、预言家验了谁、狼队在私频里怎么商量的，全场同一份，活着的和出局的都看得到。"
                    "现在轮到你说：你当时是怎么想的、哪些判断对了、哪些判断错了、错在哪一步，换一次你会怎么打。"
                    "你也可以直接回应别人刚说的话。`review_say` 是发言，`review_pass` 是这一轮没有要补充的；"
                    f"全场在同一轮里都 pass，复盘就结束，最多 {r.get('rounds_max')} 轮。"
                    "别人 pass 之后你说了新东西，他还会再轮到。")
        if task == "campaign":
            return ("YOUR TASK: the sheriff election is open. `campaign_run` with a speech puts you on the ballot and costs you your own vote "
                    "in this election; `campaign_pass` keeps your vote and takes you off it. The sheriff speaks first every day, casts 1.5 votes, "
                    "and on death either hands the badge to a living player or destroys it." + self._detonate_line(view))
        if task == "campaign_vote":
            return (f"YOUR TASK: elect a sheriff. The candidates are {pub.get('sheriff_candidates', [])} and their campaign speeches are above. "
                    "Whoever wins speaks first each day and casts 1.5 votes for the rest of the game. Call `campaign_vote`; leaving out the target abstains.")
        if task == "last_words":
            return ("YOUR TASK: you are dead and this is your last turn. Nobody can question you about it, every living player hears it, "
                    "and it stays in the public record for anyone to pull up later. There is no turn after this one. Call `last_words`.")
        if task == "hunter_shoot":
            return ("YOUR TASK: you are the hunter and you have died. `hunter_shoot` with a target kills that player immediately and in the open; "
                    "`hunter_hold` fires at nobody and the game goes on with everyone still alive. The shot is yours however you died.")
        if task == "badge":
            return ("YOUR TASK: you held the badge and you have died. `badge_transfer` gives it to a living player, who then speaks first each day "
                    "and casts 1.5 votes; `badge_tear` destroys it, and nobody holds a badge for the rest of the game.")
        return ("YOUR TASK: the table is voting now. `vote` takes any living player except yourself, and there is no way to abstain -- "
                "if you end your steps without voting, one is filled in for you. The player with the most votes is exiled and gets last words; "
                "a tie exiles nobody and the day ends with everyone still alive. A sheriff's ballot counts 1.5." + self._detonate_line(view))

    @staticmethod
    def _detonate_line(view: dict) -> str:
        if view["role"] != "werewolf":
            return ""
        return (" 你这一轮还可以 `self_detonate`：立刻结束今天——今天不再投票、不放逐任何人，你当场出局、没有遗言，"
                "警徽若在你手上一并失效。")

    # ---------------------------------------------------------------- observations
    def observation_message(self, tool: str, observation: str, untrusted: bool, source: str | None = None):
        directives: set[int] = set()
        detections: list[dict] = []
        text = observation
        if untrusted:
            directives |= verify_mod.directive_targets(observation)
            text, detections = self.guard.clean_incoming(observation, source=source or f"tool:{tool}", kind="tool_result")
        truncated = len(text) > self.max_observation_chars
        if truncated:
            text = text[:self.max_observation_chars] + "\n[...truncated]"
        return text, directives, detections, truncated

    def trim_steps(self, step_messages: list[dict]) -> list[dict]:
        keep = max(2, self.max_steps_in_context - self.max_steps_in_context % 2)
        if len(step_messages) <= keep:
            return step_messages
        dropped = step_messages[:-keep]
        window = step_messages[-keep:]
        while window and window[0].get("role") == "tool":
            dropped.append(window.pop(0))
        summary = "; ".join(s for s in (_summarise(m) for m in dropped) if s) or "some lookups"
        return [{"role": "user", "content": f"[earlier this turn you already did: {summary}]"}] + window


def _tool_lines(info: list[str], term: list[str]) -> str:
    out = []
    if info:
        out.append(f"- Information tools ({', '.join(sorted(info))}) can be called several times.")
    if term:
        out.append(f"- One terminal action ends your turn immediately: {', '.join(sorted(term))}. Take it once you have what you need.")
    out.append("- The tools offered to you this turn are the only ones that exist.")
    return "\n" + "\n".join(out)


def _note_block(note: str) -> list[str]:
    return ["", "--- your note from last turn ---", (note or "").strip() or "(none)"]


def _summarise(m: dict) -> str:
    calls = m.get("tool_calls") or []
    if calls:
        return f"called {calls[0]['function']['name']}"
    return (m.get("content") or "").strip()[:60]


def _dead_label(cause: str) -> str:
    return {"exiled": "exiled", "shot": "shot", "detonated": "self-detonated", "night": "died at night"}.get(cause, cause)


def _render_action(a: dict) -> str:
    kind = _ACTION_CN.get(a.get("kind"), a.get("kind", ""))
    head = f"第{a.get('round')}轮" + ("夜" if a.get("phase") == "night" else "昼") + " " + kind
    bits = []
    if a.get("abstained"):
        bits.append("→ 弃票")
    elif a.get("target") is not None:
        bits.append(f"→ {a['target']} 号")
    if a.get("outcome"):
        bits.append(f"（结果：{a['outcome']}）")
    if a.get("plan"):
        bits.append(f"方案 {a['plan']}，我的分工 {a.get('my_tactic')}")
    if a.get("said"):
        bits.append(f"：{a['said'][:70]}")
    if a.get("reason"):
        bits.append(f"，理由：{a['reason'][:50]}")
    return head + (" " + " ".join(bits) if bits else "")
