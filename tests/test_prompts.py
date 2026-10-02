"""The prompts carry the math conventions and fit vLLM's context budget."""

import pytest

from app.services.vlm_service import LINE_SYSTEM_PROMPT, SYSTEM_PROMPT

PROMPTS = pytest.mark.parametrize("prompt", [LINE_SYSTEM_PROMPT, SYSTEM_PROMPT], ids=["line", "page"])


@PROMPTS
def test_prompt_has_math_conventions(prompt):
    assert "Subiectul" in prompt and "I_2" in prompt and "det(A)" in prompt
    assert r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}" in prompt
    assert "never flatten" in prompt
    assert "A(5) not A(s)" in prompt and "4x not ux" in prompt
    assert r"\text{[illegible]}" in prompt


@PROMPTS
def test_prompt_has_no_control_characters(prompt):
    # A single backslash in the source turns "\text" into a tab and "\begin" into a backspace.
    assert [c for c in prompt if ord(c) < 32 and c != "\n"] == []


@PROMPTS
def test_prompt_fits_context_budget(prompt):
    """vLLM runs with --max-model-len 1024 = prompt + image (<= 256 tokens) + 512 output.
    Measured against vLLM, the line prompt (779 chars) with a max-size image takes
    506 tokens, leaving 518 for output. Re-measure before letting it grow."""
    assert len(prompt) <= 800
