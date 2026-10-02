"""Phase 2 verifier: corrections, formatting fixes and green/yellow/grey status."""

import pytest

from app.services.verifier import KatexChecker, get_katex_checker, verify_page


class NoKatex(KatexChecker):
    """Forces the structural fallback (no Node / KaTeX)."""

    def __init__(self):
        self.available = False


def one(text, *, page=(), checker=None):
    """Verify `text` as the last line of a page whose other lines are `page`."""
    res = verify_page([*page, text], checker=checker if checker is not None else NoKatex())
    return res[-1]


def kinds(res):
    return [i.kind for i in res.issues]


# ---- digit-for-letter corrections ------------------------------------------

def test_function_argument_corrected_with_page_evidence():
    res = one(r"$\det(A(s)) = 21$", page=[r"$A(5) = \begin{pmatrix} 2 & 5 \\ -5 & -2 \end{pmatrix}$"])
    assert res.latex == r"$\det(A(5)) = 21$"
    assert res.status == "yellow"
    assert "digit_arg" in kinds(res) and res.issues[0].fixed


def test_function_argument_without_evidence_is_only_flagged():
    res = one(r"$A(s) = 3$")
    assert res.latex == r"$A(s) = 3$"
    assert res.status == "yellow"
    assert [(i.kind, i.fixed) for i in res.issues] == [("digit_arg", False)]


def test_negative_argument_uses_signed_evidence():
    assert one(r"$A(-l) + A(5)$").latex == r"$A(-1) + A(5)$"  # l is never a variable


def test_variable_arguments_are_left_alone():
    res = one(r"$A(x) \cdot A(-x) = I_2$")
    assert res.latex == r"$A(x) \cdot A(-x) = I_2$" and res.status == "green"


@pytest.mark.parametrize("raw, fixed", [
    (r"$x = l$", r"$x = 1$"),
    (r"$x \in [-l, l]$", r"$x \in [-1, 1]$"),
    (r"$16(t-x^2) \geq 0$", r"$16(1-x^2) \geq 0$"),
    (r"$\ell \geq x^2$", r"$1 \geq x^2$"),
    (r"$2 + l = 3$", r"$2 + 1 = 3$"),
])
def test_lone_l_and_t_become_1(raw, fixed):
    res = one(raw)
    assert res.latex == fixed
    assert res.status == "yellow" and "digit_one" in kinds(res)


@pytest.mark.parametrize("tex", [
    r"$\ln x + \lim_{x \to 0} \log_2 x$",
    r"$\left( x \right)$",
    r"$2t + 1 = 5$",                  # a coefficient: t is a variable here
    r"$280 \text{ lei} + t_1$",       # inside \text, and a subscripted t
    r"$f(t) = t^2$",
])
def test_letters_that_are_not_a_misread_1_are_kept(tex):
    assert one(tex).latex == tex


def test_t_kept_when_page_defines_it_as_variable():
    res = one(r"$t - x^2 \geq 0$", page=[r"Notăm $t = 2^x$, $t > 0$"])
    assert res.latex == r"$t - x^2 \geq 0$"


# ---- notation ---------------------------------------------------------------

@pytest.mark.parametrize("raw, fixed, status", [
    (r"$(=) \quad 1 - x^2 \geq 0$", r"$\Leftrightarrow \quad 1 - x^2 \geq 0$", "yellow"),  # misread
    (r"$c=) x = 2$", r"$\Leftrightarrow x = 2$", "yellow"),
    (r"$x = 2 => y = 3$", r"$x = 2 \Rightarrow y = 3$", "green"),                     # formatting only
    (r"$2A(-1)+A(5)=3A(1)$ ($=$)", r"$2A(-1)+A(5)=3A(1)$ $\Leftrightarrow$", "yellow"),
])
def test_arrows(raw, fixed, status):
    res = one(raw)
    assert res.latex == fixed and res.status == status


@pytest.mark.parametrize("raw", [
    r"$-4 + 25 = 21 \quad A^r$",
    r"$= 21, A^{\alpha}$",
    r"$\begin{pmatrix} 6 \\ 3 \end{pmatrix}^{\text{A}^{\text{a}}}$",
])
def test_grade_mark(raw):
    res = one(raw)
    assert res.latex.endswith(r"\text{,,A''}$")
    assert "grade_mark" in kinds(res)


def test_grade_mark_read_with_a_leading_a():
    assert one(r"$30 - 4 + 1 = 25, aA^4$").latex == r"$30 - 4 + 1 = 25, \text{,,A''}$"


def test_composition_law_o_becomes_circ():
    res = one(r"2) Pe $\mathbb{R}$, $x o y = 2xy - x - y + 2$")
    assert res.latex == r"2) Pe $\mathbb{R}$, $x \circ y = 2xy - x - y + 2$"
    assert res.status == "yellow"
    assert one(r"$\cos x + o(1)$").latex == r"$\cos x + o(1)$"  # an "o" that is not between operands


def test_text_only_line_becomes_plain_text():
    assert one(r"\text{Subiectul } 3").latex == "Subiectul 3"


def test_matrix_power_is_not_a_grade_mark():
    assert one(r"$B = A^2$").latex == r"$B = A^2$"


# ---- formatting --------------------------------------------------------------

def test_missing_delimiters_keep_label_and_words_as_text():
    res = one("a) f'(x) ≥ 0 pt. x ∈ [0,1] => f descrescătoare pe [0,1]")
    assert res.latex == r"a) $f'(x) \geq 0$ pt. $x \in [0,1] \Rightarrow f$ descrescătoare pe $[0,1]$"
    assert res.status == "green"  # formatting fixes only


@pytest.mark.parametrize("raw, fixed", [
    ("C.E. : 3x+4 ≥ 0", r"C.E. : $3x+4 \geq 0$"),
    ("c) Tangenta in A(a, f(a)) || dr. y = 5x + 1", r"c) Tangenta in $A(a, f(a)) ||$ dr. $y = 5x + 1$"),
    ("AB = AC = 5√2", r"$AB = AC = 5\sqrt{2}$"),
    ("√10n ∈ Q pt. 10, 40", r"$\sqrt{10n} \in Q$ pt. $10, 40$"),
])
def test_abbreviations_and_roots(raw, fixed):
    assert one(raw).latex == fixed


def test_bare_latex_line_is_wrapped():
    res = one(r"x^{2}.I_{2} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}")
    assert res.latex == r"$x^{2} \cdot I_{2} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$"
    assert res.status == "green"


def test_prose_line_stays_text():
    assert one("Subiectul 2").latex == "Subiectul 2"
    assert one("punct de minim").latex == "punct de minim"


def test_unclosed_dollar_is_closed():
    res = one(r"$x = 2")
    assert res.latex == r"$x = 2$" and "delimiters" in kinds(res)


def test_left_array_right_becomes_pmatrix():
    res = one(r"$\left( \begin{array}{cc} 2 & 5 \\ -5 & -2 \end{array} \right)$")
    assert res.latex == r"$\begin{pmatrix} 2 & 5 \\ -5 & -2 \end{pmatrix}$"


def test_multiline_display_block_is_collapsed():
    res = one("$$\n\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}\n$$")
    assert res.latex == r"$$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$"


# ---- status --------------------------------------------------------------------

def test_clean_line_is_green():
    res = one(r"b) $A(-1) = \begin{pmatrix} 2 & -1 \\ 1 & -2 \end{pmatrix}$")
    assert res.status == "green" and res.issues == []


def test_flattened_matrix_is_flagged():
    res = one(r"$\begin{pmatrix} 4 & -2 \end{pmatrix} + \begin{pmatrix} 2 & 5 \end{pmatrix}$")
    assert res.status == "yellow"
    assert kinds(res).count("matrix_shape") == 1  # identical issues are merged with a count
    assert "(x2)" in res.issues[0].message


def test_syntax_error_is_grey_with_structural_checks():
    res = one(r"$\frac{1}{2$")
    assert res.status == "grey" and "syntax" in kinds(res)


@pytest.mark.skipif(not get_katex_checker().available, reason="Node/KaTeX not installed (npm install in tools/)")
def test_syntax_error_is_grey_with_katex():
    res = one(r"$\frac{1}{2$", checker=get_katex_checker())
    assert res.status == "grey" and res.issues[-1].message.startswith("KaTeX:")
    assert one(r"$\frac{1}{2}$", checker=get_katex_checker()).status == "green"


def test_vlm_error_empty_and_illegible_are_grey():
    res = verify_page(["", r"$\text{[illegible]}$", "x"], errors=[None, None, "boom"], checker=NoKatex())
    assert [r.status for r in res] == ["grey", "grey", "grey"]
    assert [r.issues[0].kind for r in res] == ["empty", "illegible", "vlm_error"]


def test_prompt_echo_is_dropped():
    # seen on the gold set: a blank crop at the top of a photo
    res = one(r"$\text{[illegible]} 1) a) b); I_2, A(x), det(A), \mathbb{R}, \ln.$")
    assert res.latex == "" and res.status == "grey" and res.issues[0].kind == "prompt_echo"
    assert one(r"$\det(A) = 21, A(x) = I_2$").status == "green"  # real math using the same symbols


def test_partly_illegible_is_yellow():
    assert one(r"$x = \text{[illegible]} + 2$").status == "yellow"


def test_page_order_and_length_are_preserved():
    texts = [r"$A(5)$", "", r"$A(s)$"]
    res = verify_page(texts, checker=NoKatex())
    assert [r.latex for r in res] == [r"$A(5)$", "", r"$A(5)$"]
