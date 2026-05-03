import unittest

from qworld.__main__ import normalize_input_data


class CliInputNormalizationTest(unittest.TestCase):
    def test_string_input_becomes_single_question_item(self):
        self.assertEqual(
            normalize_input_data("What is AI?"),
            [{"id": "0", "question": "What is AI?"}],
        )

    def test_dict_input_gets_default_id_without_mutation(self):
        data = {"question": "What is AI?"}

        result = normalize_input_data(data)

        self.assertEqual(result, [{"id": "0", "question": "What is AI?"}])
        self.assertEqual(data, {"question": "What is AI?"})

    def test_list_input_accepts_strings_and_dicts(self):
        self.assertEqual(
            normalize_input_data(["What is AI?", {"id": "custom", "question": "What is ML?"}]),
            [
                {"id": "0", "question": "What is AI?"},
                {"id": "custom", "question": "What is ML?"},
            ],
        )

    def test_invalid_list_item_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "strings or objects"):
            normalize_input_data([42])


if __name__ == "__main__":
    unittest.main()
