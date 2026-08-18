import qworld.client as client
from qworld.client import CriteriaGenerator, normalize_questions


def _generator_without_clients(monkeypatch):
    gen = CriteriaGenerator.__new__(CriteriaGenerator)
    gen.n_scenario_expands = 0
    gen.n_perspective_expands = 0
    gen.n_criteria_expands = 0
    gen.dedup_threshold = 0.6
    gen.max_workers = 2
    gen._log_fn = None
    gen._call_llm = lambda *args, **kwargs: None
    gen._get_embeddings = lambda texts: []

    def fake_pipeline(item, **kwargs):
        return {
            "prompt_id": item.get("prompt_id", item.get("id")),
            "question": item.get("question"),
        }

    monkeypatch.setattr(client, "run_pipeline", fake_pipeline)
    return gen


def test_normalize_questions_does_not_mutate_single_dict():
    question = {"question": "What is AI?"}

    single_input, items = normalize_questions(question)

    assert single_input is True
    assert items == [{"id": "0", "question": "What is AI?"}]
    assert question == {"question": "What is AI?"}


def test_generate_does_not_mutate_batch_dicts(monkeypatch):
    gen = _generator_without_clients(monkeypatch)
    questions = [{"question": "A"}, {"id": "custom", "question": "B"}]

    results = gen.generate(questions)

    assert questions == [{"question": "A"}, {"id": "custom", "question": "B"}]
    assert {r["id"] for r in results} == {"0", "custom"}


def test_generate_single_string_returns_single_result(monkeypatch):
    gen = _generator_without_clients(monkeypatch)

    result = gen.generate("What is AI?")

    assert result["id"] == "0"
    assert result["question"] == "What is AI?"
