"""Malformed-verdict fixtures: python -m unittest discover -s tests"""
import unittest

from judge.parsing import ParseError, parse_json, regex_fallback, validate_pairwise, validate_pointwise

CRIT = ["correctness", "completeness"]


class TestParsing(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(parse_json('{"a": 1}'), ({"a": 1}, "ok"))

    def test_code_fence_and_prose(self):
        obj, st = parse_json('Sure! Here is my verdict:\n```json\n{"winner": "A"}\n```\nHope that helps.')
        self.assertEqual((obj, st), ({"winner": "A"}, "extracted"))

    def test_think_block(self):
        obj, st = parse_json('<think>the {weird} reasoning</think>{"winner": "B"}')
        self.assertEqual(obj["winner"], "B")

    def test_trailing_comma_single_quotes_python_bools(self):
        obj, st = parse_json("{'winner': 'tie', 'ok': True,}")
        self.assertEqual((obj["winner"], obj["ok"], st), ("tie", True, "repaired"))

    def test_unquoted_keys(self):
        obj, st = parse_json('{winner: "A", confidence: 0.8}')
        self.assertEqual((obj["winner"], st), ("A", "repaired"))

    def test_truncated(self):
        obj, _ = parse_json('{"criteria": {"correctness": {"score": 4, "rationale": "fine"}')
        self.assertEqual(obj["criteria"]["correctness"]["score"], 4)

    def test_braces_inside_strings(self):
        obj, _ = parse_json('{"rationale": "uses {x} and }", "winner": "A"}')
        self.assertEqual(obj["winner"], "A")

    def test_no_json(self):
        with self.assertRaises(ParseError):
            parse_json("I think response A is better.")

    def test_regex_fallback(self):
        out = regex_fallback('correctness: 4 ... "completeness": {"score": 2 ... winner = B', CRIT)
        self.assertEqual(out["criteria"]["correctness"]["score"], 4)
        self.assertEqual(out["criteria"]["completeness"]["score"], 2)
        self.assertEqual(out["winner"], "B")

    def test_validate_pointwise_clamps_and_flags(self):
        v, issues = validate_pointwise({"criteria": {"correctness": {"score": 7}, "completeness": "3"}}, CRIT)
        self.assertEqual(v["criteria"]["correctness"]["score"], 5)
        self.assertEqual(v["criteria"]["completeness"]["score"], 3)
        self.assertIn("clamped:correctness=7", issues)

    def test_validate_pointwise_list_form_and_missing(self):
        v, issues = validate_pointwise({"criteria": [{"name": "correctness", "score": 2}]}, CRIT)
        self.assertEqual(v["criteria"]["correctness"]["score"], 2)
        self.assertIn("missing:completeness", issues)

    def test_validate_pairwise_variants(self):
        self.assertEqual(validate_pairwise({"verdict": "Response B"})[0]["winner"], "B")
        self.assertEqual(validate_pairwise({"winner": "TIE"})[0]["winner"], "tie")
        self.assertIn("bad_winner", validate_pairwise({"winner": "both"})[1])


if __name__ == "__main__":
    unittest.main()
