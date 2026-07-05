import json
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "qworld_cli", Path(__file__).resolve().parents[1] / "qworld" / "__main__.py"
)
qworld_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qworld_cli)

_limit_examples = qworld_cli._limit_examples
_load_input = qworld_cli._load_input
_item_count = qworld_cli._item_count
_write_output = qworld_cli._write_output


def test_load_jsonl_input(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "q1", "question": "What is AI?"}),
                json.dumps({"id": "q2", "question": "What is ML?"}),
            ]
        ),
        encoding="utf-8",
    )

    assert _load_input(str(path)) == [
        {"id": "q1", "question": "What is AI?"},
        {"id": "q2", "question": "What is ML?"},
    ]


def test_write_jsonl_output(tmp_path):
    path = tmp_path / "results.jsonl"
    _write_output(
        str(path),
        [{"id": "q1", "final_criteria": []}, {"id": "q2", "final_criteria": []}],
        output_format="jsonl",
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["q1", "q2"]


def test_limit_examples_accepts_single_json_object():
    item = {"id": "q1", "question": "What is AI?"}

    assert _limit_examples(item, 1) == item
    assert _limit_examples(item, 0) == []
    assert _item_count(item) == 1
