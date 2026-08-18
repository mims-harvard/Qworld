import json
import sys

import pytest

import qworld.__main__ as cli
import qworld.client as client


def test_normalize_input_data_accepts_supported_shapes():
    assert cli._normalize_input_data("What is AI?") == [{"id": "0", "question": "What is AI?"}]

    item = {"question": "What is AI?"}
    assert cli._normalize_input_data(item) == [item]

    items = [{"question": "A"}, {"question": "B"}]
    assert cli._normalize_input_data(items) is items


def test_normalize_input_data_rejects_unsupported_json():
    with pytest.raises(TypeError, match="input JSON"):
        cli._normalize_input_data(42)


def test_main_handles_single_object_input(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"id": "q1", "question": "What is AI?"}), encoding="utf-8")

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        def generate(self, data):
            assert data == [{"id": "q1", "question": "What is AI?"}]
            return [{"id": "q1", "final_criteria": []}]

    monkeypatch.setattr(client, "CriteriaGenerator", FakeGenerator)
    monkeypatch.setattr(
        sys,
        "argv",
        ["qworld", "-i", str(input_path), "-o", str(output_path), "--max-examples", "1"],
    )

    cli.main()

    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"id": "q1", "final_criteria": []}]
