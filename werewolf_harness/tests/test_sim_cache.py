from ..app.llm.sim import BLOCK_TOKENS, CHARS_PER_TOKEN, PrefixCache


def test_a_repeated_prefix_is_cached_block_by_block():
    c = PrefixCache()
    shared = "x" * (BLOCK_TOKENS * CHARS_PER_TOKEN * 10)          # 10 blocks = 1280 tokens
    total, cached = c.lookup_and_fill(shared + "A" * 100, now=0)
    assert cached == 0
    total, cached = c.lookup_and_fill(shared + "B" * 100, now=1)
    assert cached == 10 * BLOCK_TOKENS                              # the shared blocks, not the tail


def test_a_change_early_in_the_prompt_breaks_everything_after_it():
    c = PrefixCache()
    body = "y" * (BLOCK_TOKENS * CHARS_PER_TOKEN * 12)
    c.lookup_and_fill("seat 1 " + body, now=0)
    _, cached = c.lookup_and_fill("seat 2 " + body, now=1)
    assert cached == 0


def test_short_prompts_and_expired_entries_do_not_count():
    c = PrefixCache(ttl_s=10)
    short = "z" * (BLOCK_TOKENS * CHARS_PER_TOKEN * 4)             # 512 tokens < 1024
    c.lookup_and_fill(short, now=0)
    assert c.lookup_and_fill(short, now=1)[1] == 0
    long = "w" * (BLOCK_TOKENS * CHARS_PER_TOKEN * 9)
    c.lookup_and_fill(long, now=0)
    assert c.lookup_and_fill(long, now=100)[1] == 0                 # past the TTL
