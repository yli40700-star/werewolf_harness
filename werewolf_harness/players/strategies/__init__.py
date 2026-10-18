"""The wolf playbook, and retrieval over it.

Documented human tactics, kept as data. Three reasons it is a corpus rather
than prompt text:

**It does not fit.** Nine strategies with their preconditions, execution notes
and failure signals is several thousand tokens, and a wolf needs one of them at
a time. Putting the whole book in every system prompt would spend the context
budget on eight strategies the agent is not running.

**It makes the assignment real.** The guides are explicit that these are a
*division of labour* decided on night one -- 悍跳 needs a 冲锋 to work, and
倒钩 contradicts 悍跳 in the same seat. That structure is in the data, so the
coordination phase can read it rather than have it hard-coded.

**It is untrusted content an agent retrieves.** Which makes it the same shape
as a poisoned document in a RAG pipeline, and a channel this harness can
already measure.
"""

from .playbook import (
    Strategy,
    assignments,
    get_strategy,
    load_playbook,
    plan_for,
    search_strategies,
)
from .situation import Situation, describe, read_situation

__all__ = [
    "Strategy", "assignments", "get_strategy", "load_playbook",
    "plan_for", "search_strategies",
    "Situation", "describe", "read_situation",
]
