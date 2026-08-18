import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "qworld_cli", Path(__file__).resolve().parents[1] / "qworld" / "__main__.py"
)
qworld_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qworld_cli)

_item_count = qworld_cli._item_count
_limit_examples = qworld_cli._limit_examples


def test_limit_examples_accepts_single_json_object():
    item = {"id": "q1", "question": "What is AI?"}

    assert _limit_examples(item, 1) == item
    assert _limit_examples(item, 0) == []
    assert _item_count(item) == 1
