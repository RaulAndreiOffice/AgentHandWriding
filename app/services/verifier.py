"""Phase 2: deterministic verification and correction of line transcriptions.

Runs after the VLM, over all lines of a page at once (some rules need page
context). For every line it:

  1. fixes formatting: wraps math that came back without $...$ (keeping exercise
     labels and Romanian words as text), closes an unclosed $ / $$, turns
     unicode math (≥ ∈ ² ⇒ ...) and ASCII arrows into LaTeX, and rewrites
     \\left( \\begin{array}...\\right) as pmatrix / vmatrix;
  2. corrects systematic misreadings of the 2B model, using page context:
     - a function argument read as a letter: A(s) -> A(5) when the page also
       has A(5) (s->5, u->4, g->9, o->0; l, \\ell -> 1 always; t -> 1 unless the
       page defines t as a variable);
     - a lone l / \\ell / t standing for the digit 1: "x = l", "[-l, l]", "+ t";
     - "(=)", "c=)" for <=>, "=)" for =>, and the student's grade mark
       (A^r, A^{\\alpha} ... at the end of a line) as \\text{,,A''};
  3. validates: KaTeX parse of every math segment (Node + tools/katex_check.mjs
     when available, otherwise balanced braces / environments / \\left-\\right),
     and matrix shape (a one-row matrix is probably a flattened 2x2).

Each line gets a status:
  green  - parses, nothing corrected or flagged (formatting fixes do not count);
  yellow - a content correction was applied, or something is flagged for review;
  grey   - does not parse, came back empty / illegible, or the VLM call failed.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.services.vlm_service import strip_code_fences

logger = logging.getLogger(__name__)

Status = Literal["green", "yellow", "grey"]

# --------------------------------------------------------------------- results


@dataclass
class Issue:
    kind: str
    message: str
    #: True when the verifier changed the text, False when it only flags it.
    fixed: bool = False
    #: False for pure formatting (delimiters, unicode -> LaTeX): does not lower the status.
    content: bool = True
    #: For a decision the VLM can be re-asked about: an ambiguity key such as "s_vs_5:A(s)",
    #: "t_vs_1", "arrow:(=)"; its possible answers, the one currently applied, and a snippet
    #: of the text around it (used to phrase the question).
    key: str | None = None
    choices: tuple[str, ...] = ()
    chosen: str | None = None
    context: str = ""
    #: "reask" when a targeted VLM re-ask settled it: it no longer lowers the status.
    resolved_by: str | None = None
    #: How sure the verifier is of its own choice at a decision point:
    #: "strong" (page evidence, or a letter that is never a variable), "weak" (a plausible
    #: default), "flag" (no choice made, the text is left as the VLM wrote it).
    strength: Literal["strong", "weak", "flag"] | None = None

    @property
    def blocking(self) -> bool:
        """Keeps the line yellow: a content correction or flag nobody has confirmed yet."""
        return self.content and self.resolved_by is None


@dataclass
class VerifiedLine:
    latex: str
    status: Status
    issues: list[Issue] = field(default_factory=list)

    @property
    def corrections(self) -> int:
        return sum(1 for i in self.issues if i.fixed and i.content)


# -------------------------------------------------------------- page context

#: Letters the 2B model substitutes for digits, and the digit they stand for.
LETTER_DIGIT = {"s": "5", "S": "5", "u": "4", "g": "9", "o": "0", "O": "0", "l": "1", "t": "1", "\\ell": "1"}

_FUNC_ARG = re.compile(r"(?<![\\A-Za-z])([A-Z])\(\s*(-?)\s*(\\ell(?![A-Za-z])|[A-Za-z0-9]+)\s*\)")
#: Evidence that the page uses t as a variable: a substitution ("notăm t = 2^x"),
#: "t > 0", t with an exponent or index, f(t), or a coefficient like 2t.
_T_VARIABLE = re.compile(
    r"not[ăa]m|notez"
    r"|(?<![\\A-Za-z])t\s*=\s*[^=]*x|(?<![\\A-Za-z])t\s*(?:>|\\geq?|\\in)\s*0"
    r"|(?<![\\A-Za-z])t\s*[\^_]|(?<![\\A-Za-z])[a-z]\(\s*t\s*\)|\d\s*t(?![A-Za-z])")


@dataclass
class PageContext:
    #: function letter -> arguments seen on the page, e.g. {"A": {"5", "-1", "x"}}
    func_args: dict[str, set[str]]
    #: the page itself uses t as a variable (substitution "notăm t = 2^x", "t > 0")
    t_is_variable: bool

    @classmethod
    def from_texts(cls, texts: list[str]) -> PageContext:
        joined = "\n".join(texts)
        args: dict[str, set[str]] = {}
        for name, sign, arg in _FUNC_ARG.findall(joined):
            args.setdefault(name, set()).add(sign + arg)
        return cls(args, bool(_T_VARIABLE.search(joined)))


# ------------------------------------------------------------ math detection

LABEL = re.compile(r"^\s*(Subiectul\s+(?:[IVX]+|\d+)\b|\d{1,2}\)|[a-hA-H]\)|[IVX]+\.)\s*")

#: Words that are math, not prose, even though they are alphabetic.
MATH_WORDS = {"sin", "cos", "tg", "ctg", "tan", "cot", "ln", "lg", "log", "lim", "det", "max", "min", "rang", "arctg",
              "arcsin", "arccos", "mod", "gcd"}
#: Short Romanian prose words that would otherwise look like variables.
PROSE_WORDS = {"pt", "pt.", "pe", "si", "și", "şi", "in", "în", "de", "la", "cu", "iar", "sau", "deci", "unde", "nu",
               "este", "sunt", "fie", "ca", "că", "din", "al", "ale"}  # not "a"/"o": variables and x o y
_WORD = re.compile(r"^[(\[]?([A-Za-zăâîșțĂÂÎȘȚşţ]+)[.,;:)\]]?$")
_MATHY = re.compile(r"[\\^_=<>≥≤≠∈∉∞⇒⇔→²³√±·×∪∩∀∃ℝ]|\d\s*[-+*/]\s*\w|\w\s*[-+*/]\s*\d|\d\)|\([^)]*\d")


def _looks_math(text: str) -> bool:
    return bool(_MATHY.search(text))


#: Abbreviations: pt. nr. caz. dr. C.E. e.g. (optionally with a ":" or in parentheses)
_ABBREV = re.compile(r"^\(?(?:[A-Za-zăâîșț]{1,4}\.){1,3}[:,)]?$")


def _is_prose(token: str) -> bool:
    if _ABBREV.match(token) or re.fullmatch(r"[:;,.!?]+", token):
        return True
    m = _WORD.match(token)
    if not m:
        return False
    word = m.group(1)
    low = word.lower()
    if low in MATH_WORDS:
        return False
    return low in PROSE_WORDS or token.lower() in PROSE_WORDS or len(word) >= 3


def _wrap_math_runs(text: str) -> tuple[str, bool]:
    """Wrap the math parts of plain text in $...$, leaving prose words as text.

    Returns (new_text, changed). Falls back to wrapping the whole text when a
    run would split a {...} group (e.g. a \\text{...} with spaces inside).
    """
    m = LABEL.match(text)
    label, rest = (text[: m.end()], text[m.end():]) if m else ("", text)
    if not rest.strip() or not _looks_math(rest):
        return text, False

    tokens = rest.split()
    runs: list[tuple[bool, list[str]]] = []
    for tok in tokens:
        prose = _is_prose(tok)
        if runs and runs[-1][0] == prose:
            runs[-1][1].append(tok)
        else:
            runs.append((prose, [tok]))

    parts = []
    for prose, toks in runs:
        chunk = " ".join(toks)
        if prose or not _looks_math(chunk) and not re.search(r"[A-Za-z0-9]", chunk):
            parts.append(chunk)
        else:
            if chunk.count("{") != chunk.count("}"):
                return f"{label}${rest.strip()}$", True
            parts.append(f"${chunk}$")
    return label + " ".join(parts), True


# ----------------------------------------------------------------- segments


def split_segments(line: str) -> tuple[list[tuple[str, str]], str | None]:
    """Split a line into ("text"|"$"|"$$", content) segments.

    Returns the segments and the delimiter left open at the end, if any.
    """
    segs: list[tuple[str, str]] = []
    mode, buf, i = "text", [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "$":
            buf.append("\\$")
            i += 2
            continue
        if line.startswith("$$", i) and mode in ("text", "$$"):
            segs.append((mode, "".join(buf)))
            mode, buf = ("$$" if mode == "text" else "text"), []
            i += 2
            continue
        if line[i] == "$" and mode in ("text", "$"):
            segs.append((mode, "".join(buf)))
            mode, buf = ("$" if mode == "text" else "text"), []
            i += 1
            continue
        buf.append(line[i])
        i += 1
    segs.append((mode, "".join(buf)))
    return [s for s in segs if s[1] or s[0] != "text"], (mode if mode != "text" else None)


def join_segments(segs: list[tuple[str, str]]) -> str:
    return "".join(c if kind == "text" else f"{kind}{c}{kind}" for kind, c in segs)


# ------------------------------------------------------------ math rewrites

UNICODE_MATH = {
    "≥": r"\geq ", "≤": r"\leq ", "≠": r"\neq ", "∈": r"\in ", "∉": r"\notin ", "∞": r"\infty ",
    "⇒": r"\Rightarrow ", "⇔": r"\Leftrightarrow ", "→": r"\to ", "·": r"\cdot ", "×": r"\times ",
    "±": r"\pm ", "∪": r"\cup ", "∩": r"\cap ", "∀": r"\forall ", "∃": r"\exists ", "ℝ": r"\mathbb{R}",
    "Δ": r"\Delta ", "π": r"\pi ", "°": r"^\circ ", "²": "^{2}", "³": "^{3}", "⁴": "^{4}", "⁵": "^{5}",
    "√": r"\sqrt ", "ℓ": r"\ell ",
}

#: (pattern, replacement, issue kind, message, content?)
ARROWS = [
    (re.compile(r"\\left\(\s*=\s*\\right\)|\(\s*=\s*\)|(?<![A-Za-z\\])c\s*=\s*\)|<=>"), r"\\Leftrightarrow ",
     "arrow", "read as <=>", True),
    (re.compile(r"(?<![<=!])=\s*>|(?<=\s)=\)(?=\s|$)|^=\)(?=\s)"), r"\\Rightarrow ", "arrow", "read as =>", True),
]

_TEXT_GROUP = re.compile(r"\\(?:text|mathrm|operatorname|textbf|mbox)\s*\{[^{}]*\}")
_GRADE = re.compile(r"(?:(?<=,)|(?<=\s)|(?<=\\quad)|(?<=\\,)|(?<=\\ll)|(?<=,,)|(?<=''))\s*(?:,{1,2}|a|„)?\s*"
                    r"(?:\\text\s*\{\s*A\s*\}|A)\s*\^\s*(?:\{(?:[^{}]|\{[^{}]*\}){1,30}\}|\\?[A-Za-z0-9]+)\s*$")
#: The grade mark as a superscript of whatever precedes it: ...\end{pmatrix}^{\text{A}^{\text{a}}}
_GRADE_SUP = re.compile(r"\^\s*\{\s*(?:\\text\s*\{\s*A\s*\}|A)\s*\^\s*(?:\{(?:[^{}]|\{[^{}]*\}){1,30}\}|\\?[A-Za-z0-9]+)"
                        r"\s*\}\s*$")
_COMPOSE = re.compile(r"(?<=[A-Za-z0-9)}])\s+o\s+(?=[A-Za-z0-9(\\])")
#: "." between factors, not after a word ("dr. y") nor in an abbreviation ("C.E.", "e.g.").
_DOT_PRODUCT = re.compile(r"(?<![A-Za-z][A-Za-z])(?<![A-Za-z]\.[A-Za-z])(?<=[A-Za-z0-9}])(?<!(?<![A-Za-z])[A-Z])"
                          r"\s*\.\s*(?=[A-Za-z\\(])(?![A-Za-z]\.)")
_SQRT_ARG = re.compile(r"√\s*(\([^()]*\)|[0-9A-Za-z]+)")
_LONE_ONE = re.compile(r"\\ell(?![A-Za-z])|(?<![A-Za-z\\0-9])l(?![A-Za-z0-9(])")
_LONE_T = re.compile(r"(?<![A-Za-z\\0-9])t(?![A-Za-z0-9(_^])")
_ARRAY_MATRIX = re.compile(
    r"\\left\s*([(|\[])\s*\\begin\{(?:array\}\{[^}]*|matrix)\}(.*?)\\end\{(?:array|matrix)\}\s*\\right\s*[)|\]]",
    re.DOTALL)
_ENV = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
_MATRIX_ENV = re.compile(r"\\begin\{(pmatrix|vmatrix|bmatrix|Bmatrix|Vmatrix|matrix|array)\}(?:\{[^}]*\})?(.*?)\\end\{\1\}",
                         re.DOTALL)


def _protect(tex: str) -> tuple[str, list[str]]:
    """Hide \\text{...}-like groups from the letter->digit rules."""
    kept: list[str] = []

    def keep(m: re.Match) -> str:
        kept.append(m.group(0))
        return f"\x00{len(kept) - 1}\x00"

    return _TEXT_GROUP.sub(keep, tex), kept


def _restore(tex: str, kept: list[str]) -> str:
    return re.sub("\x00(\\d+)\x00", lambda m: kept[int(m.group(1))], tex)


#: Answer labels of the arrow decision -> LaTeX
ARROW_CHOICES = {"⇔": r"\Leftrightarrow ", "⇒": r"\Rightarrow ", "=": "= "}


def _snippet(text: str, start: int, end: int, width: int = 18) -> str:
    return text[max(0, start - width): end + width].strip()


def _decide(issues: list[Issue], decisions: dict[str, str], *, kind: str, key: str, choices: tuple[str, ...],
            default: str | None, context: str, message: str, strong: bool = False) -> str | None:
    """Pick the answer for one decision point and record it as an Issue.

    `default` is the verifier's own choice (None = it only flags, the text stays);
    `strong` marks a default backed by evidence. A re-ask answer in
    `decisions[key]` overrides it and marks the issue resolved (app/services/reask.py
    only sends answers that decide a flag or confirm the default).
    Returns the chosen label (None when the text is left as it is).
    """
    strength = "flag" if default is None else "strong" if strong else "weak"
    if key in decisions:
        chosen = decisions[key]
        issues.append(Issue(kind, f"{message}: re-ask answered '{chosen}'", fixed=chosen != choices[-1],
                            key=key, choices=choices, chosen=chosen, context=context, resolved_by="reask",
                            strength=strength))
        return chosen
    issues.append(Issue(kind, message, fixed=default is not None and default != choices[-1], key=key,
                        choices=choices, chosen=default if default is not None else choices[-1], context=context,
                        strength=strength))
    return default


def fix_math(tex: str, ctx: PageContext, issues: list[Issue], decisions: dict[str, str] | None = None) -> str:
    """Formatting fixes and content corrections inside one math segment.

    Every content correction is a decision point (see _decide): `decisions`
    holds answers from a targeted VLM re-ask, keyed like the issues' `key`.
    The last entry of each `choices` tuple is "keep the text as the VLM wrote it".
    """
    decisions = decisions or {}
    # Formatting: unicode symbols, \left( array \right) -> pmatrix.
    before = tex
    tex = _SQRT_ARG.sub(r"\\sqrt{\1}", tex)  # √10n -> \sqrt{10n} (before the plain √ mapping)
    for ch, rep in UNICODE_MATH.items():
        tex = tex.replace(ch, rep)
    tex = re.sub(r"\^\{(\d)\}\^\{(\d)\}", r"^{\1\2}", tex)
    if tex != before:
        issues.append(Issue("unicode", "unicode math symbols converted to LaTeX", fixed=True, content=False))

    def to_env(m: re.Match) -> str:
        env = {"(": "pmatrix", "|": "vmatrix", "[": "bmatrix"}[m.group(1)]
        return f"\\begin{{{env}}}{m.group(2)}\\end{{{env}}}"

    new = _ARRAY_MATRIX.sub(to_env, tex)
    if new != tex:
        issues.append(Issue("matrix_env", "\\left( array \\right) rewritten as pmatrix/vmatrix", fixed=True, content=False))
        tex = new

    # Arrows. Plain "=>" / "<=>" are formatting; "(=)", "c=)", "=)" are misreadings (decisions).
    for pattern, rep, kind, msg, _ in ARROWS:
        def arrow(m: re.Match, rep=rep, msg=msg, kind=kind) -> str:
            raw = m.group(0).strip()
            default = "⇔" if "Leftrightarrow" in rep else "⇒"
            if raw in ("<=>", "=>", "= >"):
                issues.append(Issue(kind, f"'{raw}' {msg}", fixed=True, content=False))
                return ARROW_CHOICES[default]
            chosen = _decide(issues, decisions, kind=kind, key=f"arrow:{raw}", choices=(*ARROW_CHOICES, raw),
                             default=default, context=_snippet(tex, m.start(), m.end()), message=f"'{raw}' {msg}")
            return ARROW_CHOICES.get(chosen, m.group(0))
        tex = pattern.sub(arrow, tex)

    # Grade mark at the end of the line: A^r, \text{A}^{\alpha}, ,,A^4 -> \text{,,A''}
    m = _GRADE.search(tex) or _GRADE_SUP.search(tex)
    if m and not re.search(r"(?:=|\+|-|\\cdot)\s*$", tex[: m.start()]):
        chosen = _decide(issues, decisions, kind="grade_mark", key="grade_mark", choices=("yes", "no"),
                         default="yes", context=m.group(0).strip(),
                         message=f"'{m.group(0).strip()}' read as the grade mark ,,A''")
        if chosen == "yes":
            comma = "," if m.group(0).lstrip().startswith(",") or tex[: m.start()].rstrip().endswith(",") else ""
            tex = tex[: m.start()].rstrip().rstrip(",") + comma + r" \text{,,A''}"

    # The composition law of bac algebra (x ∘ y) read as the letter o: "x o y".
    m = _COMPOSE.search(tex)
    if m:
        chosen = _decide(issues, decisions, kind="circ", key="o_vs_circ", choices=("∘", "o"), default="∘",
                         context=_snippet(tex, m.start(), m.end()), message="'o' between operands read as \\circ")
        if chosen == "∘":
            tex = _COMPOSE.sub(r" \\circ ", tex)

    # A multiplication dot read as a full stop: x^{2}.I_{2} (not a decimal point).
    new = _DOT_PRODUCT.sub(r" \\cdot ", tex)
    if new != tex:
        issues.append(Issue("cdot", "'.' between factors written as \\cdot", fixed=True, content=False))
        tex = new

    body, kept = _protect(tex)

    # Function arguments read as letters: A(s) -> A(5).
    def fix_arg(m: re.Match) -> str:
        name, sign, arg = m.group(1), m.group(2), m.group(3)
        digit = LETTER_DIGIT.get(arg)
        if digit is None or (arg == "t" and ctx.t_is_variable):
            return m.group(0)
        seen = ctx.func_args.get(name, set())
        evidence = (sign + digit) in seen or digit in seen
        always = arg in ("l", "\\ell")
        why = (f": page also has {name}({sign}{digit})" if evidence
               else f": '{arg}' is never a variable here" if always else "")
        letter = arg.lstrip("\\")
        chosen = _decide(issues, decisions, kind="digit_arg", key=f"{letter}_vs_{digit}:{name}({sign}{arg})",
                         choices=(digit, arg), default=digit if evidence or always else None,
                         context=f"{name}({sign}{arg})", strong=True,
                         message=f"{name}({sign}{arg}) -> {name}({sign}{digit}){why}" if evidence or always
                         else f"{name}({sign}{arg}): '{arg}' may be the digit {digit}")
        return f"{name}({sign}{chosen})" if chosen else m.group(0)

    body = _FUNC_ARG.sub(fix_arg, body)

    # A lone l / \ell / t standing for the digit 1: one decision per letter and line.
    def lone_ones(pattern: re.Pattern, text: str) -> str:
        first: dict[str, str] = {}
        for m in pattern.finditer(text):
            first.setdefault(m.group(0), _snippet(_restore(text, kept), m.start(), m.end()))
        for tok, context in first.items():
            name = tok.lstrip("\\")
            chosen = _decide(issues, decisions, kind="digit_one", key=f"{name}_vs_1", choices=("1", tok),
                             default="1", context=context, message=f"lone '{tok}' read as 1",
                             strong=tok != "t")  # l / \ell are never variables; t can be
            if chosen == "1":
                text = pattern.sub(lambda mm, tok=tok: "1" if mm.group(0) == tok else mm.group(0), text)
        return text

    body = lone_ones(_LONE_ONE, body)
    if not ctx.t_is_variable:
        body = lone_ones(_LONE_T, body)

    return _restore(body, kept)


# ---------------------------------------------------------------- validation


def structural_errors(tex: str) -> list[str]:
    """Balanced braces, environments and \\left/\\right (used without KaTeX)."""
    errs = []
    depth = 0
    i = 0
    while i < len(tex):
        if tex[i] == "\\":
            i += 2
            continue
        if tex[i] == "{":
            depth += 1
        elif tex[i] == "}":
            depth -= 1
            if depth < 0:
                errs.append("unmatched '}'")
                depth = 0
        i += 1
    if depth:
        errs.append(f"{depth} unclosed '{{'")
    stack = []
    for kind, name in _ENV.findall(tex):
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            errs.append(f"\\end{{{name}}} without matching \\begin")
    if stack:
        errs.append(f"\\begin{{{stack[-1]}}} without \\end")
    lefts, rights = len(re.findall(r"\\left(?![A-Za-z])", tex)), len(re.findall(r"\\right(?![A-Za-z])", tex))
    if lefts != rights:
        errs.append(f"{lefts} \\left vs {rights} \\right")
    return errs


def matrix_issues(tex: str) -> list[Issue]:
    out = []
    for env, body in _MATRIX_ENV.findall(tex):
        rows = [r for r in re.split(r"\\\\", body.replace("\\hline", "")) if r.strip()]
        if env == "array":
            continue  # sign tables are arrays with a free shape
        if len(rows) == 1:
            out.append(Issue("matrix_shape", f"{env} with a single row: possibly a flattened matrix"))
        elif len({r.count("&") for r in rows}) > 1:
            out.append(Issue("matrix_shape", f"{env} rows have different numbers of columns"))
    return out


class KatexChecker:
    """Parses math with real KaTeX through Node (tools/katex_check.mjs)."""

    def __init__(self, tools_dir: Path | None = None):
        self.tools_dir = tools_dir or Path(__file__).resolve().parents[2] / "tools"
        self.node = shutil.which("node")
        self.available = bool(self.node and (self.tools_dir / "node_modules" / "katex").is_dir())

    def check(self, items: list[tuple[str, bool]]) -> list[str | None] | None:
        """One error message (or None) per (tex, display) item; None if KaTeX is unavailable."""
        if not self.available or not items:
            return [None] * len(items) if self.available else None
        try:
            proc = subprocess.run(
                [self.node, str(self.tools_dir / "katex_check.mjs")],
                input=json.dumps([{"tex": t, "display": d} for t, d in items]),
                capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
            return json.loads(proc.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            logger.warning("KaTeX check failed, using structural checks: %s", exc)
            return None


@lru_cache
def get_katex_checker() -> KatexChecker:
    return KatexChecker()


# ---------------------------------------------------------------------- main


#: "(=)" whose "=" alone was put in math: "($=$)" -> "$\Leftrightarrow$"
_SPLIT_IFF = re.compile(r"\(\s*\$\s*=\s*\$\s*\)")
_TEXT_ONLY = re.compile(r"^\s*\\text\s*\{([^{}]*)\}\s*([0-9IVX]*)\s*$")


def _fix_line(line: str, ctx: PageContext, issues: list[Issue], decisions: dict[str, str]) -> str:
    # "($=$)": put the whole "(=)" in math, where the arrow decision handles it.
    line = _SPLIT_IFF.sub("$(=)$", line)
    m = _TEXT_ONLY.match(line)
    if m and "$" not in line:  # "\text{Subiectul } 3": prose the model wrote as LaTeX
        issues.append(Issue("delimiters", "\\text{...} line written as plain text", fixed=True, content=False))
        line = " ".join(m.group(1).split() + m.group(2).split())
    if "$" not in line:
        line, changed = _wrap_math_runs(line)
        if changed:
            issues.append(Issue("delimiters", "math wrapped in $...$", fixed=True, content=False))
    segs, open_mode = split_segments(line)
    if open_mode:
        issues.append(Issue("delimiters", f"unclosed {open_mode} closed", fixed=True, content=False))
    out: list[tuple[str, str]] = []
    for kind, content in segs:
        if kind == "text" and re.search(r"\\[A-Za-z]+", content):
            # LaTeX commands in a text segment: the model forgot the delimiters there.
            wrapped, changed = _wrap_math_runs(content)
            if changed:
                issues.append(Issue("delimiters", "LaTeX outside $...$ wrapped", fixed=True, content=False))
                out.extend((k, c if k == "text" else fix_math(c, ctx, issues, decisions))
                           for k, c in split_segments(wrapped)[0])
                continue
        out.append((kind, content if kind == "text" else fix_math(content, ctx, issues, decisions)))
    # Replacements add a trailing space; keep single spaces inside math, none before the closing $.
    def tidy(c: str) -> str:
        c = re.sub(r"(?<=\S) {2,}", " ", c)
        return c.rstrip() if c.strip() else c

    return join_segments([(k, c if k == "text" else tidy(c)) for k, c in out])


def _dedupe(issues: list[Issue]) -> list[Issue]:
    """One entry per (kind, message), with a count: "matrix with a single row (x2)"."""
    counts: dict[tuple, int] = {}
    first: dict[tuple, Issue] = {}
    for i in issues:
        key = (i.kind, i.message, i.fixed, i.content, i.key, i.resolved_by)
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, i)
    out = []
    for k, i in first.items():
        if counts[k] > 1:
            i = replace(i, message=f"{i.message} (x{counts[k]})")
        out.append(i)
    return out


def _echoes_prompt(text: str) -> bool:
    """Shown a blank crop, the 2B model recites its system prompt
    ("1) a) b); I_2, A(x), det(A), \\mathbb{R}, \\ln."): two or more of the prompt's
    own phrases in one line."""
    from app.services.vlm_service import CONVENTIONS, LINE_SYSTEM_PROMPT

    compact = re.sub(r"\s+", "", text)
    phrases = {re.sub(r"\s+", "", p) for p in re.split(r"[\n.;:]", CONVENTIONS + LINE_SYSTEM_PROMPT)}
    hits = sum(1 for p in phrases if len(p) >= 12 and p in compact)
    hits += sum(1 for p in ("I_2,A(x),det(A)", "labels1)a)b)", "Subiectul1/2/3") if p in compact)
    return hits >= 2 or "I_2,A(x),det(A)" in compact


def _collapse_display_blocks(text: str) -> str:
    """$$ blocks spanning several lines become one line, so lines can be checked independently."""
    return re.sub(r"\$\$(.+?)\$\$", lambda m: "$$" + " ".join(m.group(1).split()) + "$$", text, flags=re.DOTALL)


def verify_page(texts: list[str], errors: list[str | None] | None = None,
                checker: KatexChecker | None = None,
                decisions: list[dict[str, str] | None] | None = None) -> list[VerifiedLine]:
    """Verify and correct every region of one page (same order as texts).

    `decisions[i]` (optional) holds re-ask answers for region i, keyed like Issue.key;
    they override the verifier's own choice at those decision points.
    """
    errors = errors or [None] * len(texts)
    decisions = decisions or [None] * len(texts)
    ctx = PageContext.from_texts([t for t, e in zip(texts, errors) if not e])
    checker = checker if checker is not None else get_katex_checker()

    results: list[VerifiedLine] = []
    pending: list[tuple[int, str, bool]] = []  # (result index, tex, display) for KaTeX
    for text, error, decided in zip(texts, errors, decisions):
        if error:
            results.append(VerifiedLine("", "grey", [Issue("vlm_error", error)]))
            continue
        text = _collapse_display_blocks(strip_code_fences(text.strip()))
        if _echoes_prompt(text):
            results.append(VerifiedLine("", "grey", [Issue(
                "prompt_echo", "the VLM repeated its instructions (an empty or unreadable crop); dropped")]))
            continue
        if not text:
            results.append(VerifiedLine("", "grey", [Issue("empty", "the VLM returned nothing for this region")]))
            continue
        issues: list[Issue] = []
        lines = [_fix_line(ln.strip(), ctx, issues, decided or {}) for ln in text.splitlines() if ln.strip()]
        latex = "\n".join(lines)
        for ln in lines:
            for kind, content in split_segments(ln)[0]:
                if kind != "text":
                    pending.append((len(results), content, kind == "$$"))
                    issues.extend(matrix_issues(content))
        if "[illegible]" in latex:
            issues.append(Issue("illegible", "the VLM marked part of this line as illegible"))
        results.append(VerifiedLine(latex, "green", issues))

    verdicts = checker.check([(t, d) for _, t, d in pending]) if checker else None
    for k, (idx, tex, _) in enumerate(pending):
        errs = [verdicts[k]] if verdicts is not None and verdicts[k] else [] if verdicts is not None \
            else structural_errors(tex)
        for err in errs:
            results[idx].issues.append(Issue("syntax", f"KaTeX: {err}" if verdicts is not None else err))

    for r in results:
        r.issues = _dedupe(r.issues)
        if r.status == "grey":
            continue
        kinds = {i.kind for i in r.issues}
        only_illegible = re.fullmatch(r"\$*\s*\\text\{\[illegible\]\}\s*\$*", r.latex.strip()) is not None
        if "syntax" in kinds or only_illegible:
            r.status = "grey"
        elif any(i.blocking for i in r.issues):
            r.status = "yellow"
    return results
