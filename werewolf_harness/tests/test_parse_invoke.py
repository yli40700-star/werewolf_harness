from werewolf_harness.players.providers.base import invoke_call, parse_json_action
from werewolf_harness.players.providers.mock import MockClient

XML = ('<invoke name="night_kill">\n<parameter name="target_id">2</parameter>\n'
       '<parameter name="thought">第一晚没有信息，先刀 2 号。</parameter>\n</invoke>')


def test_a_reply_in_the_xml_tool_call_form_is_read_as_the_same_call():
    """sonnet 5.5 on the JSON protocol answered about one call in five in this form."""
    call = invoke_call(XML)
    assert call is not None and not call.malformed
    assert call.name == "night_kill"
    assert call.arguments == {"target_id": 2, "thought": "第一晚没有信息，先刀 2 号。"}
    assert invoke_call("<invoke>no name</invoke>") is None and invoke_call("just prose") is None


def test_the_json_parser_itself_is_unchanged_so_old_recordings_replay():
    assert parse_json_action(XML).malformed
    assert parse_json_action('{"thought": "t", "action": "vote", "args": {"target_id": 3}}').name == "vote"


def test_new_games_turn_it_on_and_old_settings_leave_it_off():
    from werewolf_harness.replay_llm import build_players
    base = {"guard": "", "evidence": False, "playbook": "static", "temperature": 0.7, "lang": "zh",
            "max_tokens": 10**6, "max_usd": None, "parallel": 1}
    old, _, _ = build_players(base, 2, lambda s: MockClient(seed=s))
    new, _, _ = build_players({**base, "invoke_calls": True}, 2, lambda s: MockClient(seed=s))
    assert not old[1].invoke_calls and new[1].invoke_calls
