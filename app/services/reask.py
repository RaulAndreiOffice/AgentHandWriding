"""Targeted re-ask of problematic lines (the agentic second pass).

Runs after the verifier, only on lines that are not green, so a clean page
costs no extra VLM calls:

  1. Re-transcription. Grey lines (KaTeX error, empty, illegible, failed call)
     and lines with a malformed matrix are transcribed again with a strict
     output schema and a description of what went wrong. The new text is kept
     only if the verifier rates it better than the old one.
  2. Closed questions. For every open ambiguity on a yellow line (A(s) vs
     A(5), a misread arrow, o vs ∘, the grade mark, e^n x vs ln x, x -> 50 vs
     x -> infinity) the line crop is sent with
     one multiple-choice question. Only the text of the answer is used (no
     logprobs: they crash vLLM under WSL). An answer is used when it is one of
     the options and either decides a flag the verifier left open or confirms
     the verifier's correction: the verifier re-runs with it and the line turns
     green. It never reverses a correction (see judge); a disagreement or an
     answer outside the options leaves the line yellow.
     Lone l / t vs 1 is not asked (SKIP_QUESTION_KINDS).

All calls go through one semaphore of size VLM_CONCURRENCY.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Literal

from app.config import Settings
from app.services.verifier import Issue, VerifiedLine
from app.services.vlm_service import VLMService, VLMServiceError

if TYPE_CHECKING:
    from app.services.pipeline_service import LineTranscription

logger = logging.getLogger(__name__)

#: await verify(texts, errors, decisions) -> one VerifiedLine per text (bound to the page by the caller)
Verify = Callable[[list[str], list[str | None], list[dict[str, str]] | None], Awaitable[list[VerifiedLine]]]

STATUS_RANK = {"green": 0, "yellow": 1, "grey": 2}
#: Issue kinds that make a line worth transcribing again (in addition to any grey line).
RETRANSCRIBE_KINDS = {"matrix_shape", "illegible"}
#: Ambiguities not worth a question. Measured on tema17 (all formats tried): asked whether a
#: lone stroke is 1 or l/t, the 2B model answered the letter every time (0/3 right) - it
#: repeats its own misreading, so the call only costs time.
SKIP_QUESTION_KINDS = {"digit_one"}


@dataclass
class ReaskEvent:
    line: int
    kind: Literal["question", "retranscribe"]
    key: str | None
    question: str
    answer: str | None
    accepted: bool
    detail: str = ""


@dataclass
class Question:
    text: str
    #: what the model may answer -> the decision label recorded for the verifier
    options: dict[str, str]


# ------------------------------------------------------------------ questions

_DIGIT_ARG = re.compile(r"^(\w+)_vs_(\d):(.+)$")


def build_question(issue: Issue) -> Question | None:
    """A closed question for one open decision point, or None if it has no template."""
    key, ctx = issue.key or "", issue.context
    if m := _DIGIT_ARG.match(key):
        digit, call = m.group(2), m.group(3)
        letter = issue.choices[-1]
        shown = "l" if letter == "\\ell" else letter
        return Question(
            f"Look at {call} in this handwritten line: a capital letter followed by a character in "
            f"parentheses. Is the character inside the parentheses the digit {digit} or the letter "
            f"{shown}? Answer {digit} or {shown}.",
            {digit: digit, shown: letter})
    if key.endswith("_vs_1"):
        letter = issue.choices[-1]
        shown = "l" if letter == "\\ell" else letter
        return Question(
            f"This handwritten line was read as: {ctx}. Look at the character read as '{shown}' "
            f"standing on its own. Is it the digit 1 or the letter {shown}? Answer 1 or {shown}.",
            {"1": "1", shown: letter})
    if key.startswith("arrow:"):
        return Question(
            f"This handwritten line was read as: {ctx}. Look at the connector symbol read as "
            f"'{key[6:]}'. Which is it? A: ⇔ (double arrow, if and only if), B: ⇒ (implies), "
            f"C: = (equals). Answer A, B or C.",
            {"A": "⇔", "B": "⇒", "C": "="})
    if key == "o_vs_circ":
        return Question(
            f"This handwritten line was read as: {ctx}. Look at the symbol between the operands read as "
            f"'o'. Is it A: the composition operator ∘ (a small raised circle), or B: the letter o? "
            f"Answer A or B.",
            {"A": "∘", "B": "o"})
    if key == "en_vs_ln":
        return Question(
            f"This handwritten line was read as: {ctx}. Look at the part read as 'e^n x'. Is it "
            f"A: the natural logarithm ln x, or B: e to the power n, times x? Answer A or B.",
            {"A": "ln", "B": issue.choices[-1]})
    if key == "50_vs_inf":
        return Question(
            f"This handwritten line was read as: {ctx}. Look at what x tends to, read as '50'. Is it "
            f"A: infinity ∞ (a sideways 8), or B: the number 50? Answer A or B.",
            {"A": "∞", "B": "50"})
    if key == "0_vs_inf":
        return Question(
            f"This handwritten line was read as: {ctx}. Look at what x tends to under lim, read as '0'. "
            f"Is it A: infinity ∞ (a sideways 8), or B: the digit 0? Answer A or B.",
            {"A": "∞", "B": "0"})
    if key == "frac0_vs_inf":
        return Question(
            f"This handwritten line was read as: {ctx}. Look at the denominator read as '0' in the fraction "
            f"before '= 0'. Is it A: infinity ∞ (a sideways 8), or B: the digit 0? Answer A or B.",
            {"A": "∞", "B": "0"})
    if key.startswith("eval_bar:"):
        return Question(
            f"This handwritten line was read as: {ctx}. Look at the part read as a matrix of two numbers "
            f"stacked one above the other. Is it A: a vertical evaluation bar with the integration limits "
            f"(F(x) evaluated from the bottom number to the top one), or B: a real matrix or determinant? "
            f"Answer A or B.",
            {"A": "bar", "B": "matrix"})
    if key == "grade_mark":
        return Question(
            "Look at the end of this handwritten line. Is there the mark „A” (a capital A, usually "
            "between quotation marks, meaning the result is correct)? Answer yes or no.",
            {"yes": "yes", "no": "no"})
    return None


def describe_problem(line: VerifiedLine, error: str | None) -> str:
    """What went wrong, in words the re-transcription prompt can use."""
    if error:
        return "the request failed"
    parts = []
    for i in line.issues:
        if i.kind == "syntax":
            parts.append(f"it was not valid KaTeX ({i.message.removeprefix('KaTeX: ')[:120]})")
        elif i.kind == "matrix_shape":
            parts.append("a matrix came out as a single row or with uneven rows; handwritten matrices here "
                         "have 2 or 3 rows, write every row")
        elif i.kind == "illegible":
            parts.append("part of it was marked illegible; look again closely")
        elif i.kind == "empty":
            parts.append("it came back empty, but the crop contains handwriting")
    return "; ".join(dict.fromkeys(parts)) or "it looked malformed"


_PUNCT = " \t\r\n\"'`*.,;:!?"
#: "B", "B:", "(B)", "B) ⇒", "B. implies", "B - ..." : a letter option followed by a delimiter or nothing
_LEADING_LETTER = re.compile(r"^\(?([A-Za-z])\)?(?=\s*(?:[:.)\-,]|$))")
_ANSWER_IS = re.compile(r"^(?:the\s+)?(?:answer|option|choice)(?:\s+is)?\s*:?\s*\(?([A-Za-z])\)?(?![A-Za-z])", re.I)
_TOKEN = re.compile(r"[^\s,.;:!?()\[\]'\"`*]+")


def match_option(answer: str, options: dict[str, str]) -> str | None:
    """Which option did the model pick? Tolerant of the ways a small model phrases it.

    - exact, ignoring surrounding whitespace/punctuation and (when the options do not
      differ only by case) case: "5.", "Yes", " S " -> 5 / yes / s;
    - lettered options (A, B, C): a leading letter followed by a delimiter or nothing
      ("B: ⇒ (", "(B)", "B) implies"), "Answer: B", or the one option symbol it names
      ("⇒"). A bare leading word is not enough: "A small circle" is not option A;
    - other options: the first word ("yes, there is"), else the only option that appears
      as a standalone word ("the digit 5"). Two different options -> ambiguous -> None.
    """
    text = answer.strip(_PUNCT)
    if not text:
        return None
    folded = {k.casefold(): k for k in options}
    case_free = len(folded) == len(options)

    def lookup(token: str) -> str | None:
        token = token.strip(_PUNCT)
        if token in options:
            return token
        return folded.get(token.casefold()) if case_free else None

    if (key := lookup(text)) is not None:
        return key

    if all(re.fullmatch(r"[A-Z]", k) for k in options):
        for pattern in (_LEADING_LETTER, _ANSWER_IS):
            if (m := pattern.match(text)) and m.group(1).upper() in options:
                return m.group(1).upper()
        tokens = set(_TOKEN.findall(text))
        named = {k for k, label in options.items() if not label.isalnum() and label in tokens}
        return named.pop() if len(named) == 1 else None

    tokens = _TOKEN.findall(text)
    if tokens and (key := lookup(tokens[0])) is not None:
        return key
    hits = {k for t in tokens if (k := lookup(t)) is not None}
    return hits.pop() if len(hits) == 1 else None


#: Arrow readings the re-ask may switch between: "(=)" could be either, the verifier's
#: ⇔ is a guess (the gold has a "(=)" that is ⇒ on tema17). "=" is not among them: the
#: model's "=" answers (36 on the gold set) read the symbol literally and contradict the gold.
_ARROW_READINGS = {"⇔", "⇒"}


def judge(issue: Issue, decision: str | None) -> tuple[bool, str]:
    """Use a re-ask answer? Returns (accepted, reason when not).

    Decided on the answer text alone: it must name one of the options and either
    decide a flag (the verifier made no choice), confirm the verifier's own choice,
    or swap one arrow reading for the other (⇔ / ⇒). Otherwise it never reverses a
    rule correction: measured on the gold pages, when this 2B model disagrees with a
    rule it is mostly repeating its first misreading (arrows: 36 "=" answers).
    """
    if decision is None:
        return False, "answer is not one of the options"
    if decision == issue.chosen or issue.strength == "flag":
        return True, ""
    if issue.kind == "arrow" and {decision, issue.chosen} <= _ARROW_READINGS:
        return True, ""
    return False, f"contradicts the verifier's correction ({issue.chosen}); left for review"


def _better(new: VerifiedLine, old: VerifiedLine) -> bool:
    rank_new, rank_old = STATUS_RANK[new.status], STATUS_RANK[old.status]
    if rank_new != rank_old:
        return rank_new < rank_old
    return sum(i.blocking for i in new.issues) < sum(i.blocking for i in old.issues)


# --------------------------------------------------------------------- runner


class Reasker:
    def __init__(self, vlm: VLMService, settings: Settings):
        self.vlm = vlm
        self.settings = settings
        self.sem = asyncio.Semaphore(max(1, settings.vlm_concurrency))
        self.calls = 0

    async def _call(self, coro_fn, *args):
        async with self.sem:
            self.calls += 1
            return await coro_fn(*args)

    async def run(self, lines: list[LineTranscription], crops: dict[int, tuple[bytes, str]],
                  verify: Verify) -> list[ReaskEvent]:
        """Re-ask the non-green lines of a page in place; returns what was asked and decided.

        `lines[i].source_latex` is the VLM text the verifier works on; on success it is
        replaced by the re-transcription. latex / status / issues are updated at the end.
        """
        sources = [l.source_latex for l in lines]
        errors = [l.error for l in lines]
        current = await verify(sources, errors, None)
        events: list[ReaskEvent] = []

        # 1. Re-transcribe grey lines and malformed matrices.
        targets = [i for i, v in enumerate(current) if i in crops and (
            v.status == "grey" or any(x.kind in RETRANSCRIBE_KINDS and x.blocking for x in v.issues))]

        async def retranscribe(i: int) -> tuple[int, str | None, str]:
            problem = describe_problem(current[i], errors[i])
            try:
                text = await self._call(self.vlm.retranscribe_line, *crops[i], problem, sources[i] or None)
                return i, text, problem
            except VLMServiceError as exc:
                return i, None, f"{problem} (re-ask failed: {exc})"

        retried = await asyncio.gather(*(retranscribe(i) for i in targets))
        if retried:
            trial_sources, trial_errors = list(sources), list(errors)
            for i, text, _ in retried:
                if text is not None:
                    trial_sources[i], trial_errors[i] = text, None
            trial = await verify(trial_sources, trial_errors, None)
            for i, text, problem in retried:
                accepted = text is not None and _better(trial[i], current[i])
                if accepted:
                    sources[i], errors[i] = text, None
                events.append(ReaskEvent(i, "retranscribe", None, problem, text, accepted,
                                         f"{current[i].status} -> {trial[i].status}" if text is not None else ""))
            if any(e.accepted for e in events):
                current = await verify(sources, errors, None)

        # 2. Closed questions for the open ambiguities of yellow lines.
        asks: list[tuple[int, Issue, Question]] = []
        for i, v in enumerate(current):
            if v.status != "yellow" or i not in crops:
                continue
            open_issues = [x for x in v.issues if x.blocking and x.key and x.kind not in SKIP_QUESTION_KINDS]
            for issue in open_issues[: self.settings.reask_max_questions_per_line]:
                if (q := build_question(issue)) is not None:
                    asks.append((i, issue, q))

        async def ask(i: int, issue: Issue, q: Question) -> ReaskEvent:
            try:
                answer = await self._call(self.vlm.classify, *crops[i], q.text, list(q.options))
            except VLMServiceError as exc:
                return ReaskEvent(i, "question", issue.key, q.text, None, False, str(exc))
            key = match_option(answer, q.options)
            decision = q.options[key] if key is not None else None
            accepted, detail = judge(issue, decision)
            return ReaskEvent(i, "question", issue.key, q.text, decision if decision is not None else answer,
                              accepted, detail)

        answers = await asyncio.gather(*(ask(*a) for a in asks))
        events.extend(answers)
        decisions: list[dict[str, str]] = [{} for _ in lines]
        for e in answers:
            if e.accepted and e.key and e.answer is not None:
                decisions[e.line][e.key] = e.answer
        if any(decisions):
            current = await verify(sources, errors, decisions)

        for i, (line, v) in enumerate(zip(lines, current)):
            line.source_latex, line.error = sources[i], errors[i]
            line.latex, line.status, line.issues = v.latex, v.status, v.issues
            line.reasks = [e for e in events if e.line == i]
        logger.info("Re-ask: %d call(s), %d decision(s) accepted, %d re-transcription(s) kept",
                    self.calls, sum(map(len, decisions)),
                    sum(e.accepted for e in events if e.kind == "retranscribe"))
        return events
