"""Scoring helpers of scripts/eval_gold.py (no vLLM, no gold files needed)."""

import pytest

from scripts.eval_gold import edit_distance, match_lines, normalise, tokens


@pytest.mark.parametrize("a, b", [
    (r"\begin{pmatrix} 6 & 3 \\ -3 & -6 \end{pmatrix}", r"$\begin{pmatrix}6&3\\-3&-6\end{pmatrix}$"),
    (r"\left( \begin{matrix} 1 & 0 \\ 0 & 1 \end{matrix} \right)", r"\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}"),
    (r"x \geq 0 \Rightarrow x^{2} \in \mathbb{R}", "x ≥ 0 ⇒ x^2 ∈ \\mathbb{R}"),
    (r"$$\det(A) \quad = \, 21$$", r"\det(A)=21"),
    (r"\dfrac{1}{2}", r"\frac{1}{2}"),
    (r"x = 2 (=) y = 3", r"x = 2 \Leftrightarrow y = 3"),  # the gold uses both spellings
    (r"\text{Subiectul } 3", "Subiectul 3"),
    (r"=) e = 4", r"\Rightarrow e = 4"),
    (r"= 27 \text{,,A''}", '= 27 "A "'),
])
def test_formatting_differences_do_not_count(a, b):
    assert normalise(a) == normalise(b)


@pytest.mark.parametrize("a, b", [
    (r"A(5)", r"A(s)"),
    (r"x = 1", r"x = l"),
    (r"\begin{pmatrix} 4 & -2 \\ 2 & -4 \end{pmatrix}", r"\begin{pmatrix} 4 & -2 & 2 & -4 \end{pmatrix}"),
    (r"\Leftrightarrow", r"\Rightarrow"),
])
def test_content_differences_count(a, b):
    assert normalise(a) != normalise(b)


def test_row_separator_survives_spacing_removal():
    assert r"\\" not in normalise(r"1 \\ 2") and normalise(r"1 \\ 2") != normalise(r"1 2")


def test_edit_distance_and_tokens():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1
    assert tokens(normalise(r"\frac{1}{2}")) == ["\\frac", "{", "1", "}", "{", "2", "}"]


def test_lines_matched_to_gold_regions_by_bbox():
    regions = [{"bbox": {"x": 0, "y": 0, "w": 100, "h": 50}}, {"bbox": {"x": 0, "y": 60, "w": 100, "h": 50}}]
    lines = [{"bbox": [10, 10, 50, 20]}, {"bbox": [10, 70, 50, 20]}, {"bbox": [10, 40, 50, 30]},
             {"bbox": [500, 500, 10, 10]}]
    # third line: centre (35, 55) is in neither box, but 1/3 of it overlaps region 0 -> matched there
    assert match_lines(lines, regions) == {0: [0, 2], 1: [1]}
