from __future__ import annotations

import unittest

from math_answer_utils import extract_last_boxed


class ExtractLastBoxedTests(unittest.TestCase):
    def test_nested_fraction(self) -> None:
        self.assertEqual(extract_last_boxed(r"Answer: \boxed{\frac{1}{2}}."), r"\frac{1}{2}")

    def test_last_of_multiple_boxes(self) -> None:
        text = r"First \boxed{3}, corrected to \boxed{\frac{7}{9}}."
        self.assertEqual(extract_last_boxed(text), r"\frac{7}{9}")

    def test_malformed_box_is_ignored(self) -> None:
        self.assertIsNone(extract_last_boxed(r"Answer: \boxed{\frac{1}{2}"))

    def test_escaped_braces_are_literal(self) -> None:
        self.assertEqual(extract_last_boxed(r"\boxed{\{x, y\}}"), r"\{x, y\}")

    def test_empty_box_is_distinct_from_no_box(self) -> None:
        self.assertEqual(extract_last_boxed(r"\boxed{}"), "")
        self.assertIsNone(extract_last_boxed(""))
        self.assertIsNone(extract_last_boxed("The answer is 4."))

    def test_integer_box(self) -> None:
        self.assertEqual(extract_last_boxed(r"Therefore \boxed{-42}"), "-42")


if __name__ == "__main__":
    unittest.main()
