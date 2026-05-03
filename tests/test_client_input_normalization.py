import unittest
from unittest.mock import patch

from qworld.client import CriteriaGenerator


def _generator_without_init():
    gen = object.__new__(CriteriaGenerator)
    gen.n_scenario_expands = 0
    gen.n_perspective_expands = 0
    gen.n_criteria_expands = 0
    gen.dedup_threshold = 0.6
    gen.max_workers = 1
    gen._verbose = False
    gen._log_fn = None
    gen._call_llm = lambda *args, **kwargs: None
    gen._get_embeddings = lambda texts: []
    return gen


def _pipeline_result(item, **kwargs):
    return {
        "prompt_id": item["prompt_id"],
        "question": item["question"],
        "final_criteria": [],
    }


class GenerateInputNormalizationTest(unittest.TestCase):
    def test_single_dict_input_is_not_mutated_when_id_is_added(self):
        question = {"question": "What is AI?"}
        gen = _generator_without_init()

        with patch("qworld.client.run_pipeline", side_effect=_pipeline_result):
            result = gen.generate(question)

        self.assertEqual(question, {"question": "What is AI?"})
        self.assertEqual(result["id"], "0")

    def test_list_dict_inputs_are_not_mutated_when_ids_are_added(self):
        questions = [{"question": "What is AI?"}, {"id": "custom", "question": "What is ML?"}]
        original = [item.copy() for item in questions]
        gen = _generator_without_init()

        with patch("qworld.client.run_pipeline", side_effect=_pipeline_result):
            results = gen.generate(questions)

        self.assertEqual(questions, original)
        self.assertEqual([result["id"] for result in results], ["0", "custom"])


if __name__ == "__main__":
    unittest.main()
