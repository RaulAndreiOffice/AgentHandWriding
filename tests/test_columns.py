"""Side-by-side columns, figures and sign tables (segmenter + pipeline routing)."""

from PIL import Image, ImageDraw, ImageFont

from app.services.segmenter import SegmentationParams, segment_page
from tests.conftest import png_bytes

INK = (20, 30, 120)
FONT = ImageFont.load_default(size=40)


def figure_beside_formulas() -> Image.Image:
    """Exercise 6 style (proportions of the gold pages: 1500x2000, a figure ~300 px):
    a right triangle with labels on the left, three formula lines on the right, and
    an ordinary line above and below."""
    img = Image.new("RGB", (1500, 2000), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 80), "6) ABC dr. in A, AC = 5, BC = 13", font=FONT, fill=INK)
    # triangle: C top-left, A bottom-left, B bottom-right
    c, a, b = (150, 220), (150, 480), (470, 480)
    d.line([c, a, b, c], fill=INK, width=5)
    for label, (x, y) in (("C", (110, 190)), ("A", (110, 490)), ("B", (480, 490)), ("5", (100, 330)), ("13", (320, 300))):
        d.text((x, y), label, font=FONT, fill=INK)
    for k, text in enumerate(["AB^2 = BC^2 - AC^2 = 144", "AB = 12", "cos B = AB / BC = 12/13"]):
        d.text((560, 240 + 110 * k), text, font=FONT, fill=INK)  # ~3 glyphs from the figure
    d.text((80, 700), "Raspuns: cos B = 12/13", font=FONT, fill=INK)
    return img


def matrix_with_wide_gap() -> Image.Image:
    img = Image.new("RGB", (1200, 700), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 80), "1) A(5) =", font=FONT, fill=INK)
    d.arc((300, 40, 360, 220), 100, 260, fill=INK, width=4)     # (
    d.text((340, 60), "2", font=FONT, fill=INK)
    d.text((620, 60), "5", font=FONT, fill=INK)                  # ~6 glyphs between the columns
    d.text((330, 140), "-5", font=FONT, fill=INK)
    d.text((610, 140), "-2", font=FONT, fill=INK)
    d.arc((620, 40, 700, 220), -80, 80, fill=INK, width=4)      # )
    d.text((80, 400), "det(A(5)) = 21", font=FONT, fill=INK)
    return img


def sign_table() -> Image.Image:
    """A sign table ~40% of the page wide (like the gold pages), rules tilting a little."""
    img = Image.new("RGB", (1500, 2000), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 60), "c) f'(x) = 0 <=> x = 1", font=FONT, fill=INK)
    top, left, width = 220, 80, 620
    d.line([(left, top + 70), (left + width, top + 76)], fill=INK, width=4)
    d.line([(left, top + 150), (left + width, top + 157)], fill=INK, width=4)
    d.line([(left + 140, top - 10), (left + 143, top + 230)], fill=INK, width=4)
    for x, text in ((left + 20, "x"), (left + 190, "0"), (left + 380, "1"), (left + 540, "+oo")):
        d.text((x, top + 10), text, font=FONT, fill=INK)
    for x, text in ((left, "f'(x)"), (left + 230, "-"), (left + 380, "0"), (left + 520, "+")):
        d.text((x, top + 90), text, font=FONT, fill=INK)
    d.text((left + 10, top + 170), "f(x)", font=FONT, fill=INK)
    d.line([(left + 200, top + 175), (left + 320, top + 215)], fill=INK, width=4)
    d.line([(left + 440, top + 215), (left + 560, top + 175)], fill=INK, width=4)
    d.text((80, 600), "f(1) = 3", font=FONT, fill=INK)
    return img


# ---- segmenter -----------------------------------------------------------------------

def test_figure_and_formulas_become_separate_regions_in_reading_order():
    seg = segment_page(png_bytes(figure_beside_formulas()))
    kinds = [r.kind for r in seg.regions]
    assert kinds.count("diagram") == 1
    fig = next(r for r in seg.regions if r.kind == "diagram")
    right = [r for r in seg.regions if r.bbox.x > 520 and r.kind == "line"]
    assert len(right) == 3                                    # one region per formula line
    assert fig.bbox.x1 < 560                                  # the figure crop holds no formulas
    order = [r.index for r in seg.regions]
    assert fig.index < min(r.index for r in right)            # figure first, then its column
    assert [r.bbox.y for r in right] == sorted(r.bbox.y for r in right)
    assert seg.regions[-1].bbox.y > 650                       # the line below comes after the column


def test_columns_can_be_turned_off():
    page = png_bytes(figure_beside_formulas())
    on = segment_page(page)
    off = segment_page(page, SegmentationParams(split_columns=False, classify_regions=False))
    assert len(off.regions) < len(on.regions)
    assert all(r.kind == "line" for r in off.regions)


def test_matrix_columns_are_not_split():
    seg = segment_page(png_bytes(matrix_with_wide_gap()))
    matrix = [r for r in seg.regions if r.bbox.y < 300]
    assert len(matrix) == 1                                   # "A(5) = ( 2 5 / -5 -2 )" stays one crop
    assert matrix[0].bbox.x <= 100 and matrix[0].bbox.x1 >= 640
    assert all(r.kind == "line" for r in seg.regions)


def test_sign_table_is_tagged():
    seg = segment_page(png_bytes(sign_table()))
    assert [r.kind for r in seg.regions].count("table") == 1
    table = next(r for r in seg.regions if r.kind == "table")
    assert table.bbox.h > 200                                 # all three rows in one crop


# ---- pipeline routing ------------------------------------------------------------------

def post(client, img):
    r = client.post("/api/transcribe-page", files={"file": ("p.png", png_bytes(img), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_diagram_is_described_not_transcribed(client, fake_vlm):
    body = post(client, figure_beside_formulas())
    fig = next(line for line in body["lines"] if line["kind"] == "diagram")
    assert fig["latex"] == r"$\text{[Figura geometrica: Triunghi ABC dreptunghic in A]}$"
    assert fig["status"] == "green" and fig["issues"][0]["kind"] == "diagram"
    assert fig["image"].startswith("data:image/jpeg;base64,")
    assert fake_vlm.figure_calls == 1
    assert all(line["image"] is None for line in body["lines"] if line["kind"] != "diagram")
    assert fig["latex"] in body["latex"]


def test_nonsense_figure_description_becomes_a_bare_placeholder(client, fake_vlm):
    fake_vlm.figure_description = r"\begin{pmatrix} 5 & 13 \\ 12 & 0 \end{pmatrix}"
    fig = next(line for line in post(client, figure_beside_formulas())["lines"] if line["kind"] == "diagram")
    assert fig["latex"] == r"$\text{[Figura geometrica]}$"


def use_settings(fake_vlm, **overrides):
    from app.config import get_settings
    from app.main import app
    from app.services.pipeline_service import TranscriptionPipeline, get_pipeline
    from tests.conftest import make_settings

    s = make_settings(**overrides)
    app.dependency_overrides[get_settings] = lambda: s
    app.dependency_overrides[get_pipeline] = lambda: TranscriptionPipeline(fake_vlm, s)


def test_table_uses_the_line_prompt_by_default(client, fake_vlm):
    # On the gold set the 2B model copies the table prompt's example or loops (c|c|c|...).
    body = post(client, sign_table())
    table = next(line for line in body["lines"] if line["kind"] == "table")
    assert getattr(fake_vlm, "table_calls", 0) == 0
    assert table["latex"].startswith("$line")


def test_table_prompt_when_enabled(client, fake_vlm):
    use_settings(fake_vlm, vlm_table_prompt=True)
    body = post(client, sign_table())
    table = next(line for line in body["lines"] if line["kind"] == "table")
    assert fake_vlm.table_calls == 1
    assert table["latex"] == fake_vlm.table_latex and table["status"] == "green"


def test_looping_table_output_falls_back_to_the_line_prompt(client, fake_vlm):
    use_settings(fake_vlm, vlm_table_prompt=True)
    fake_vlm.table_latex = "$$\\begin{array}{" + "c|" * 120 + "c}\\end{array}$$"  # seen on the gold set
    table = next(line for line in post(client, sign_table())["lines"] if line["kind"] == "table")
    assert fake_vlm.table_calls == 1 and table["latex"].startswith("$line")


def test_is_degenerate():
    from app.services.pipeline_service import is_degenerate

    assert is_degenerate("c|" * 60)
    assert is_degenerate(r"\text{ } & " * 30)
    assert is_degenerate("   ")
    assert not is_degenerate(r"$$\begin{array}{c|ccc} x & 0 & 1 & +\infty \\ \hline f(x) & \nearrow & 3 & \searrow \end{array}$$")
