"""Targeted re-ask pass (app/services/reask.py), end to end through the API with a fake VLM."""

import pytest

from app.config import get_settings
from app.main import app
from app.services.pipeline_service import TranscriptionPipeline, get_pipeline
from app.services.reask import build_question, describe_problem, judge, match_option
from app.services.verifier import Issue, VerifiedLine
from tests.conftest import make_settings


def post(client, page, **settings):
    if settings:
        s = make_settings(**settings)
        fake = app.dependency_overrides[get_pipeline]().vlm
        app.dependency_overrides[get_settings] = lambda: s
        app.dependency_overrides[get_pipeline] = lambda: TranscriptionPipeline(fake, s)
    r = client.post("/api/transcribe-page", files={"file": ("p.png", page, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def reask_calls(fake):
    return getattr(fake, "reask_calls", 0)


# ---- no cost for clean pages ------------------------------------------------

def test_green_page_makes_no_reask_calls(client, fake_vlm, lined_page):
    body = post(client, lined_page)
    assert body["verification"]["green"] == 5
    assert body["verification"]["reask_calls"] == 0 and reask_calls(fake_vlm) == 0
    assert all(line["reasks"] == [] for line in body["lines"])


def test_only_non_green_lines_are_reasked(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$A(s) = 2$"}
    post(client, lined_page)
    assert reask_calls(fake_vlm) == 1  # one question for line 1, nothing for the 4 green lines


def test_reask_disabled(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$A(s) = 2$"}
    body = post(client, lined_page, reask_enabled=False)
    assert reask_calls(fake_vlm) == 0 and body["lines"][1]["status"] == "yellow"


# ---- closed questions --------------------------------------------------------

def test_confident_answer_resolves_a_flag(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$A(s) = 2$"}  # no A(5) on the page: only flagged
    fake_vlm.classify_answers = {"A(s)": "5"}
    body = post(client, lined_page)
    line = body["lines"][1]
    assert line["latex"] == r"$A(5) = 2$" and line["raw_latex"] == r"$A(s) = 2$"
    assert line["status"] == "green"
    assert line["issues"][0]["resolved_by"] == "reask" and line["issues"][0]["key"] == "s_vs_5:A(s)"
    assert line["reasks"] == [{"kind": "question", "key": "s_vs_5:A(s)", "question": line["reasks"][0]["question"],
                               "answer": "5", "accepted": True, "detail": ""}]
    assert body["verification"]["reask_calls"] == 1 and body["verification"]["reask_accepted"] == 1
    assert "$A(5) = 2$" in body["latex"]


def test_flag_decided_either_way(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$A(s) = 2$"}
    fake_vlm.classify_answers = {"A(s)": "s"}  # the model says it really is an s
    line = post(client, lined_page)["lines"][1]
    assert line["latex"] == r"$A(s) = 2$" and line["status"] == "green"
    assert line["reasks"][0]["accepted"] is True


def test_reask_requests_no_logprobs():
    """Under WSL a logprobs request kills vLLM (torch.compile needs nvcc): classify must not ask."""
    import asyncio
    from types import SimpleNamespace

    from app.services.vlm_service import VLMService

    seen = {}

    class Completions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" 5. "))])

    vlm = VLMService(make_settings())
    vlm.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    assert asyncio.run(vlm.classify(b"img", "image/png", "5 or s?", ["5", "s"])) == "5"
    assert "logprobs" not in seen and "top_logprobs" not in seen and "extra_body" not in seen


def test_confirmed_rule_correction_turns_green(client, fake_vlm, lined_page):
    fake_vlm.overrides = {0: r"$A(5) = 1$", 1: r"$A(s) = 2$"}  # page evidence -> rule corrects to A(5)
    fake_vlm.classify_answers = {"A(s)": "5"}
    line = post(client, lined_page)["lines"][1]
    assert line["latex"] == r"$A(5) = 2$" and line["status"] == "green"


def test_reask_never_reverses_a_rule_correction(client, fake_vlm, lined_page):
    fake_vlm.overrides = {0: r"$A(5) = 1$", 1: r"$A(s) = 2$"}  # page evidence -> rule corrects to A(5)
    fake_vlm.classify_answers = {"A(s)": "s"}                   # the model insists on its first reading
    line = post(client, lined_page)["lines"][1]
    assert line["latex"] == r"$A(5) = 2$" and line["status"] == "yellow"  # kept, still flagged for review
    assert line["reasks"][0]["accepted"] is False and "contradicts" in line["reasks"][0]["detail"]


def test_misread_arrow_confirmed(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$x = 2 (=) y = 3$"}
    fake_vlm.classify_answers = {"connector": "A"}  # A = ⇔, the verifier's own reading
    line = post(client, lined_page)["lines"][1]
    assert line["latex"] == r"$x = 2 \Leftrightarrow y = 3$" and line["status"] == "green"


def test_answer_outside_the_options_is_ignored(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$A(s) = 2$"}
    fake_vlm.classify_answers = {"A(s)": "The character is"}
    line = post(client, lined_page)["lines"][1]
    assert line["status"] == "yellow" and line["reasks"][0]["detail"] == "answer is not one of the options"


def test_lone_one_is_not_asked(client, fake_vlm, lined_page):
    # The 2B model repeats its l/t misreading when asked (0/3 on tema17), so no call is spent on it.
    fake_vlm.overrides = {1: r"$16(t-x^2) \geq 0$"}
    line = post(client, lined_page)["lines"][1]
    assert line["latex"] == r"$16(1-x^2) \geq 0$" and line["status"] == "yellow"
    assert line["reasks"] == [] and reask_calls(fake_vlm) == 0


def test_questions_per_line_are_capped(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$A(s) + B(s) + C(s) + D(s) = 2$"}
    body = post(client, lined_page, reask_max_questions_per_line=2)
    assert len(body["lines"][1]["reasks"]) == 2


def test_reasks_respect_vlm_concurrency(client, fake_vlm, lined_page):
    fake_vlm.overrides = {i: rf"$A(s) + {i} = 2$" for i in range(5)}
    post(client, lined_page, vlm_concurrency=2)
    assert reask_calls(fake_vlm) == 5
    assert fake_vlm.max_in_flight == 2  # overlapped, but never more than VLM_CONCURRENCY


# ---- re-transcription of grey lines ------------------------------------------

def test_grey_line_is_retranscribed(client, fake_vlm, lined_page):
    fake_vlm.overrides = {2: r"$\frac{1}{2$"}
    fake_vlm.retranscription = r"$\frac{1}{2}$"
    line = post(client, lined_page)["lines"][2]
    assert line["status"] == "green" and line["latex"] == r"$\frac{1}{2}$"
    assert line["raw_latex"] == r"$\frac{1}{2$"
    assert line["reasks"][0]["kind"] == "retranscribe" and line["reasks"][0]["accepted"] is True
    assert "not valid KaTeX" in fake_vlm.problems[0] or "unclosed" in fake_vlm.problems[0]


def test_worse_retranscription_is_discarded(client, fake_vlm, lined_page):
    fake_vlm.overrides = {2: r"$\frac{1}{2$"}
    fake_vlm.retranscription = r"$\frac{1}{2$ \end{pmatrix}"
    line = post(client, lined_page)["lines"][2]
    assert line["status"] == "grey" and line["latex"] == r"$\frac{1}{2$"
    assert line["reasks"][0]["accepted"] is False


def test_failed_line_is_retried(client, fake_vlm, lined_page):
    fake_vlm.fail = {3}
    fake_vlm.retranscription = r"$x = 3$"
    body = post(client, lined_page)
    line = body["lines"][3]
    assert line["error"] is None and line["status"] == "green" and line["latex"] == "$x = 3$"
    assert body["status"] == "success"  # the failure was recovered


# ---- unit: questions, problems, judging ---------------------------------------------

@pytest.mark.parametrize("key, choices, context, options", [
    ("s_vs_5:A(s)", ("5", "s"), "A(s)", {"5": "5", "s": "s"}),
    ("ell_vs_1:A(\\ell)", ("1", "\\ell"), "A(\\ell)", {"1": "1", "l": "\\ell"}),
    ("t_vs_1", ("1", "t"), "16(t-x^2)", {"1": "1", "t": "t"}),
    ("arrow:(=)", ("⇔", "⇒", "=", "(=)"), "x = 2 (=) y", {"A": "⇔", "B": "⇒", "C": "="}),
    ("o_vs_circ", ("∘", "o"), "x o y", {"A": "∘", "B": "o"}),
    ("grade_mark", ("yes", "no"), "A^r", {"yes": "yes", "no": "no"}),
    ("en_vs_ln", ("ln", "e^{n}"), "(e^{n} x)'", {"A": "ln", "B": "e^{n}"}),
    ("50_vs_inf", ("∞", "50"), "\\lim_{x \\to 50}", {"A": "∞", "B": "50"}),
])
def test_build_question(key, choices, context, options):
    q = build_question(Issue("k", "m", key=key, choices=choices, context=context))
    assert q.options == options
    assert all(opt in q.text for opt in options)


def test_no_question_without_template():
    assert build_question(Issue("matrix_shape", "m")) is None


def test_describe_problem():
    line = VerifiedLine("", "grey", [Issue("syntax", "KaTeX: Expected '}'"), Issue("matrix_shape", "x")])
    text = describe_problem(line, None)
    assert "not valid KaTeX (Expected '}')" in text and "2 or 3 rows" in text
    assert describe_problem(line, "timeout") == "the request failed"


ARROWS = {"A": "⇔", "B": "⇒", "C": "="}
DIGIT = {"5": "5", "s": "s"}
YES_NO = {"yes": "yes", "no": "no"}


@pytest.mark.parametrize("answer, options, expected", [
    ("B", ARROWS, "B"),
    ("B: ⇒ (", ARROWS, "B"),          # seen on the gold set
    ("C: = (", ARROWS, "C"),
    ("(B)", ARROWS, "B"),
    ("b) implies", ARROWS, "B"),
    ("Answer: A", ARROWS, "A"),
    ("The answer is C.", ARROWS, "C"),
    ("⇒", ARROWS, "B"),                # names the symbol only
    ("A small raised circle", {"A": "∘", "B": "o"}, None),  # an article, not option A
    ("D", ARROWS, None),
    ("5", DIGIT, "5"),
    ("5.", DIGIT, "5"),
    (" S ", DIGIT, "s"),
    ("The digit 5", DIGIT, "5"),
    ("It is 5, not s", DIGIT, None),   # names both: ambiguous
    ("Yes", YES_NO, "yes"),            # seen on the gold set
    ("Yes, there is a mark", YES_NO, "yes"),
    ("no.", YES_NO, "no"),
    ("I am not sure", YES_NO, None),
    ("", DIGIT, None),
])
def test_match_option(answer, options, expected):
    assert match_option(answer, options) == expected


@pytest.mark.parametrize("raw, expected", [
    (r"$x = 2 =) y = 3$", r"$x = 2 \Rightarrow y = 3$"),    # verifier read ⇒: B confirms
    (r"$x = 2 (=) y = 3$", r"$x = 2 \Rightarrow y = 3$"),   # verifier read ⇔: B switches the reading
])
def test_arrow_answer_b_with_explanation_resolves_to_implies(client, fake_vlm, lined_page, raw, expected):
    fake_vlm.overrides = {1: raw}
    fake_vlm.classify_answers = {"connector": "B: ⇒ ("}
    line = post(client, lined_page)["lines"][1]
    assert line["latex"] == expected and line["status"] == "green"
    assert line["reasks"][0]["answer"] == "⇒" and line["reasks"][0]["accepted"] is True


def test_arrow_answer_equals_is_still_refused(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$x = 2 (=) y = 3$"}
    fake_vlm.classify_answers = {"connector": "C: = ("}
    line = post(client, lined_page)["lines"][1]
    assert line["latex"] == r"$x = 2 \Leftrightarrow y = 3$" and line["status"] == "yellow"
    assert "contradicts" in line["reasks"][0]["detail"]


@pytest.mark.parametrize("strength, chosen, decision, accepted", [
    ("flag", "s", "5", True),     # decides a flag
    ("flag", "s", "s", True),     # ...either way
    ("strong", "5", "5", True),   # confirms
    ("strong", "5", "s", False),  # never reverses
    ("weak", "⇔", "⇒", False),
    ("flag", "s", None, False),   # not an option
])
def test_judge(strength, chosen, decision, accepted):
    issue = Issue("k", "m", key="k", chosen=chosen, strength=strength)
    assert judge(issue, decision)[0] is accepted


@pytest.mark.parametrize("chosen, decision, accepted", [
    ("⇔", "⇒", True), ("⇒", "⇔", True),  # either arrow reading
    ("⇔", "=", False), ("⇒", "=", False),  # but not the literal "="
])
def test_judge_arrows(chosen, decision, accepted):
    issue = Issue("arrow", "m", key="arrow:(=)", chosen=chosen, strength="weak")
    assert judge(issue, decision)[0] is accepted
