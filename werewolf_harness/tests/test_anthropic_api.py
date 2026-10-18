from werewolf_harness.players.providers.anthropic_api import AnthropicClient, system_blocks


def test_the_shared_part_gets_its_own_breakpoint():
    blocks = system_blocks("RULES\nCONTRACT\nYOUR IDENTITY (private, never shown): seat 3", True, True)
    assert [b["text"] for b in blocks] == ["RULES\nCONTRACT\n", "YOUR IDENTITY (private, never shown): seat 3"]
    assert all(b["cache_control"] == {"type": "ephemeral"} for b in blocks)


def test_the_old_layout_is_one_block():
    assert len(system_blocks("RULES\nYOUR IDENTITY (private): seat 3", True, False)) == 1
    assert "cache_control" not in system_blocks("RULES", False, True)[0]


def test_the_payload_marks_the_end_of_the_turn():
    c = AnthropicClient("claude-haiku-5-5", "k")
    p = c.payload([{"role": "system", "content": "R\nYOUR IDENTITY (private): 1"}, {"role": "user", "content": "go"}],
                  None, 0.7, 100)
    assert p["messages"][0]["content"][0] == {"type": "text", "text": "go", "cache_control": {"type": "ephemeral"}}
    assert "temperature" not in p and len(p["system"]) == 2
