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


# ---- calculus misreadings: prime after a power, e^n x, x -> 50, table arrows ---

SUBIECTUL3_A = r"$f'(x) = \left(\frac{4}{x} + \ln x - 5\right)' = (4 \cdot x^{-1})' + (e^{n}\cdot x)^{-5}'$"


def test_prime_after_power_on_a_bracket_moves_onto_the_bracket():
    res = one(SUBIECTUL3_A)
    assert r"(\ln x)'^{-5}" in res.latex                       # and e^{n} \cdot x -> \ln x (page has \ln)
    assert {"double_superscript", "ln"} <= set(kinds(res))
    assert res.status == "yellow"


@pytest.mark.parametrize("raw, fixed", [
    (r"$x^{2}' = 2x$", r"$(x^{2})' = 2x$"),                    # a plain base: the power is derived
    (r"$\left(x+1\right)^{3}' = 3$", r"$\left(x+1\right)'^{3} = 3$"),
    (r"$e^x' = e^x$", r"$(e^x)' = e^x$"),
])
def test_prime_after_power(raw, fixed):
    assert one(raw).latex == fixed


@pytest.mark.skipif(not get_katex_checker().available, reason="Node/KaTeX not installed (npm install in tools/)")
def test_prime_after_power_parses_with_katex():
    """The index-1 line of Subiectul 3 was grey: KaTeX "Double superscript"."""
    res = one(SUBIECTUL3_A, checker=get_katex_checker())
    assert "syntax" not in kinds(res) and res.status == "yellow"


@pytest.mark.parametrize("raw", [r"$(e^{n}\cdot x)' = \frac{1}{x}$", r"$f(x) = 2x + e^n x$", r"$e^{n} x$"])
def test_e_to_the_n_x_is_ln_x_on_a_calculus_page(raw):
    res = one(raw, page=[r"$f'(x) = 2$"])
    assert r"\ln x" in res.latex and "e^" not in res.latex
    assert "ln" in kinds(res) and res.status == "yellow"


def test_e_to_the_n_x_only_flagged_without_calculus():
    res = one(r"$a_n = e^{n} x$")
    assert res.latex == r"$a_n = e^{n} x$"
    assert [(i.kind, i.strength) for i in res.issues] == [("ln", "flag")]


@pytest.mark.parametrize("tex", [r"$e^{n+1} x$", r"$e^{2n} x$", r"$x \cdot e^{n}$"])
def test_other_powers_of_e_are_kept(tex):
    assert one(tex, page=[r"$f'(x) = 2$"]).latex == tex


@pytest.mark.parametrize("raw, fixed", [
    (r"$\lim_{x \to 50} \frac{x^2+1}{x} = \infty$", r"$\lim_{x \to \infty} \frac{x^2+1}{x} = \infty$"),
    (r"$\lim_{x \to +50} \frac{1}{x} = 0$", r"$\lim_{x \to +\infty} \frac{1}{x} = 0$"),
    (r"$\lim_{x \to -5o} \frac{1}{x} = 0$", r"$\lim_{x \to -\infty} \frac{1}{x} = 0$"),
])
def test_limit_to_50_is_infinity_for_rational_functions(raw, fixed):
    res = one(raw)
    assert res.latex == fixed and "infinity" in kinds(res)


def test_limit_to_50_with_asymptotes_on_the_page():
    res = one(r"$\lim_{x \to 50} (x - \ln x)$", page=[r"Cautam asimptota orizontala"])
    assert r"\to \infty" in res.latex


def test_limit_to_50_without_context_is_only_flagged():
    res = one(r"$\lim_{x \to 50} (x + 1) = 51$")
    assert res.latex == r"$\lim_{x \to 50} (x + 1) = 51$"
    assert [(i.kind, i.strength) for i in res.issues] == [("infinity", "flag")]


@pytest.mark.parametrize("tex", [r"$\lim_{x \to 500} \frac{1}{x}$", r"$\lim_{x \to 5.05} \frac{1}{x}$", r"$x = 50$"])
def test_other_numbers_are_not_infinity(tex):
    assert one(tex).latex == tex


TABLE = (r"$\begin{array}{c|ccc} x & 0 & 4 & +\infty \\ \hline f'(x) & \downarrow & 0 & \uparrow \\ \hline "
         r"f(x) & \downarrow & & \uparrow \end{array}$")


def test_table_arrows_in_the_monotony_row():
    res = verify_page([TABLE], checker=NoKatex(), kinds=["table"])[0]
    rows = res.latex.split(r"\\")
    assert r"\downarrow" in rows[1] and r"\uparrow" in rows[1]  # the f'(x) row is left alone
    assert r"f(x) & \searrow & & \nearrow" in rows[2]
    assert "table_arrows" in kinds(res) and res.status == "green"   # formatting only


def test_table_arrows_unicode_and_lines_without_rows():
    res = verify_page(["$f(x) ↓ ↗ ↑$"], checker=NoKatex(), kinds=["table"])[0]
    assert res.latex.count(r"\searrow") == 1 and res.latex.count(r"\nearrow") == 2


def test_arrows_outside_tables_are_kept():
    assert one(r"$x \downarrow 0$").latex == r"$x \downarrow 0$"


# ---- asymptotes: x -> 0 for x -> infinity, c/0 = 0, spelled-out conclusions ----

HORIZONTAL = r"c) Căutăm asimptota orizontală"
OBLIQUE = [r"Căutăm asimptota oblică", r"$y = mx + n, m \neq 0$"]


def test_slope_limit_to_zero_is_infinity():
    res = one(r"$m = \lim_{x \to 0} \frac{f(x)}{x} = \lim_{x \to 0} \frac{4}{x^2} = \frac{4}{0} = 0$", page=OBLIQUE)
    assert res.latex == (r"$m = \lim_{x \to \infty} \frac{f(x)}{x} = \lim_{x \to \infty} \frac{4}{x^2} "
                         r"= \frac{4}{\infty} = 0$")
    issue = next(i for i in res.issues if i.key == "0_vs_inf")
    assert issue.strength == "strong" and res.status == "yellow"


def test_horizontal_asymptote_limit_with_infinity_in_the_line():
    res = one(r"$\lim_{x \to 0} f(x) = \frac{4}{0} + \ln \infty - 5 = \infty$", page=[HORIZONTAL])
    assert res.latex == r"$\lim_{x \to \infty} f(x) = \frac{4}{\infty} + \ln \infty - 5 = \infty$"


def test_limit_to_zero_only_flagged_without_line_evidence():
    res = one(r"$\lim_{x \to 0} (x^2 + 3)$", page=[HORIZONTAL])
    assert res.latex == r"$\lim_{x \to 0} (x^2 + 3)$"
    assert [(i.key, i.strength) for i in res.issues] == [("0_vs_inf", "flag")]


@pytest.mark.parametrize("tex, page", [
    (r"$\lim_{x \to 0} \frac{\sin x}{x} = 1$", []),                          # no asymptote on the page
    (r"$\lim_{x \to 0^+} \ln x = -\infty$", [HORIZONTAL]),                    # one-sided: a vertical asymptote
    (r"$\lim_{x \to 0} f(x) = \infty$", [HORIZONTAL, "asimptota verticala x = 0"]),
    (r"$\lim_{x \to 0.5} \frac{1}{x}$", [HORIZONTAL]),
])
def test_genuine_limits_at_zero_are_kept(tex, page):
    res = one(tex, page=page)
    assert res.latex == tex
    assert not [i for i in res.issues if i.fixed and i.content]


def test_fraction_over_zero_equal_to_zero_is_over_infinity():
    res = one(r"$\lim_{x \to \infty} \frac{1}{x} = \frac{1}{0} = 0$")
    assert res.latex == r"$\lim_{x \to \infty} \frac{1}{x} = \frac{1}{\infty} = 0$"
    assert any(i.key == "frac0_vs_inf" and i.strength == "strong" for i in res.issues)


def test_fraction_over_zero_outside_limits_is_kept():
    assert one(r"$\frac{1}{0} = 0$").latex == r"$\frac{1}{0} = 0$"


SPELLED = (r"$\mathrm{d}i\mathrm{n}\left(\mathbb{R}\right)\ \mathrm{s}\mathrm{i}\ \left(\mathbb{R}\right)\Rightarrow "
           r"\mathrm{n}\mathrm{u}\ \mathrm{e}\mathrm{x}\mathrm{i}\mathrm{s}\mathrm{t}\mathrm{a}\ "
           r"\mathrm{a}\mathrm{s}\mathrm{i}\mathrm{m}\mathrm{p}\mathrm{t}\mathrm{o}\mathrm{t}\mathrm{a}\ "
           r"\mathrm{s}\mathrm{p}\mathrm{r}\mathrm{e}\ +\infty\ \mathrm{l}\mathrm{a}\ G_f$")


def test_spelled_out_romanian_sentence_becomes_text():
    res = one(SPELLED, page=[r"$\lim_{x \to \infty} f(x) = \infty$ (1)", r"$m = 0$ (2)"])
    assert res.latex == r"din (1) si (2) $\Rightarrow$ nu exista asimptota spre $+\infty$ la $G_f$"
    assert "spelled_prose" in kinds(res) and res.status == "yellow"


def test_references_numbered_in_order_without_page_labels():
    res = one(r"$\mathrm{d}i\mathrm{n}\ (\mathbb{R})\ \mathrm{s}\mathrm{i}\ (\mathbb{R})$")
    assert res.latex == "din (1) si (2)"


@pytest.mark.parametrize("tex", [
    r"$\int_0^1 x \, \mathrm{d}x = \frac{1}{2}$",
    r"$f : \mathbb{R} \to \mathbb{R}, \mathrm{e}^x$",
    r"$x \in (\mathbb{R})$",
    r"$\mathrm{rang}(A) = 2$",
])
def test_math_with_mathrm_is_not_prose(tex):
    assert one(tex).latex == tex


@pytest.mark.skipif(not get_katex_checker().available, reason="Node/KaTeX not installed (npm install in tools/)")
def test_spelled_sentence_parses_with_katex():
    assert "syntax" not in kinds(one(SPELLED, checker=get_katex_checker()))


def test_page_order_and_length_are_preserved():
    texts = [r"$A(5)$", "", r"$A(s)$"]
    res = verify_page(texts, checker=NoKatex())
    assert [r.latex for r in res] == [r"$A(5)$", "", r"$A(5)$"]


# ---- evaluation bars and scribbles --------------------------------------------

INTEGRAL_PAGE = (r"$\int_{-1}^{1} (2x - 1) \, dx$",)
VBAR = r"\begin{vmatrix} 1 & 1 \\ -1 & -1 \end{vmatrix}"


def test_evaluation_bar_read_as_determinant_becomes_a_bar():
    res = one(rf"$= 2 \cdot \frac{{x^2}}{{2}} {VBAR} - x {VBAR} = 1^2 - (-1)^2$", page=INTEGRAL_PAGE)
    assert res.latex == r"$= 2 \cdot \frac{x^2}{2} \Big|_{-1}^{1} - x \Big|_{-1}^{1} = 1^2 - (-1)^2$"
    assert "eval_bar" in kinds(res) and res.status == "yellow"


def test_evaluation_bar_read_as_a_column_of_limits():
    res = one(r"$\frac{x^3}{3} \begin{pmatrix} 3 \\ 0 \end{pmatrix} = 9$", page=INTEGRAL_PAGE)
    assert res.latex == r"$\frac{x^3}{3} \Big|_{0}^{3} = 9$"


@pytest.mark.parametrize("tex, page", [
    (rf"$\det(A) = {VBAR} = 0$", INTEGRAL_PAGE),  # a real determinant, after "="
    (r"$A \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$", INTEGRAL_PAGE),  # rows differ: a real matrix
    (r"$2 \cdot \begin{pmatrix} 2 \\ 3 \end{pmatrix}$", INTEGRAL_PAGE),  # a factor: a column vector
    (rf"$x {VBAR}$", ()),  # not a calculus page
])
def test_real_matrices_are_not_evaluation_bars(tex, page):
    res = one(tex, page=page)
    assert "Big|" not in res.latex and "eval_bar" not in kinds(res)


def test_evaluation_bar_parses_with_katex():
    checker = get_katex_checker()
    if not checker.available:
        pytest.skip("KaTeX (tools/node_modules) not installed")
    res = one(rf"$2 \cdot \frac{{x^2}}{{2}} {VBAR} - x {VBAR}$", page=INTEGRAL_PAGE, checker=checker)
    assert "syntax" not in kinds(res)


def test_illegible_scribble_between_equals_is_collapsed():
    res = one(r"$x \Big|_{-1}^{1} = \text{[illegible]} = 1 + 1$")
    assert res.latex == r"$x \Big|_{-1}^{1} = 1 + 1$"
    assert kinds(res) == ["scribble"] and res.status == "yellow"


def test_illegible_part_elsewhere_is_kept():
    res = one(r"$x = \text{[illegible]} + 2$")
    assert r"\text{[illegible]}" in res.latex and "illegible" in kinds(res)
