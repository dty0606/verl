import json
import importlib.util
import sys
from pathlib import Path


_PARSER_PATH = Path(__file__).resolve().parents[2] / "verl" / "utils" / "tau3_action_parser.py"
_SPEC = importlib.util.spec_from_file_location("tau3_action_parser_under_test", _PARSER_PATH)
_PARSER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _PARSER
_SPEC.loader.exec_module(_PARSER)
parse_model_output_to_tau_action = _PARSER.parse_model_output_to_tau_action


def test_qwen_xml_preserves_unquoted_json_scalar_types():
    parsed = parse_model_output_to_tau_action(
        """<tool_call>
<function=update_reservation_baggages><parameter=reservation_id>
"FQ8APE"
</parameter><parameter=total_baggages>
3
</parameter><parameter=nonfree_baggages>
0
</parameter><parameter=payment_id>
"gift_card_8190333"
</parameter></function>
</tool_call>"""
    )

    action = json.loads(parsed.action_for_env)
    assert action["name"] == "update_reservation_baggages"
    assert action["arguments"]["reservation_id"] == "FQ8APE"
    assert action["arguments"]["total_baggages"] == 3
    assert action["arguments"]["nonfree_baggages"] == 0
    assert action["arguments"]["payment_id"] == "gift_card_8190333"


def test_qwen_xml_keeps_quoted_numeric_strings_as_strings():
    parsed = parse_model_output_to_tau_action(
        """<tool_call>
<function=get_reservation_details><parameter=reservation_id>
"12345"
</parameter></function>
</tool_call>"""
    )

    action = json.loads(parsed.action_for_env)
    assert action["arguments"]["reservation_id"] == "12345"


def test_qwen_xml_preserves_nested_json_containers():
    parsed = parse_model_output_to_tau_action(
        """<tool_call>
<function=book_reservation><parameter=flights>
[{"date": "2024-05-17", "flight_number": "HAT097"}]
</parameter><parameter=payment_methods>
[{"amount": 319, "payment_id": "credit_card_3563913"}]
</parameter></function>
</tool_call>"""
    )

    action = json.loads(parsed.action_for_env)
    assert action["arguments"]["flights"] == [{"date": "2024-05-17", "flight_number": "HAT097"}]
    assert action["arguments"]["payment_methods"] == [{"amount": 319, "payment_id": "credit_card_3563913"}]
