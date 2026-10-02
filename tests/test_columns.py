"""Side-by-side columns, figures and sign tables (segmenter + pipeline routing)."""

import os

import pytest
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


def law_of_sines() -> Image.Image:
    """Triangle with apex A on the left, BC / sin A = 2R ... on the right. There is no
    clear vertical gutter: the side AC slants under the formula column, and the apex
    sits at the height of the first formula row."""
    img = Image.new("RGB", (1500, 2000), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 80), "6) In triunghiul ABC, BC = 6, A = 30", font=FONT, fill=INK)
    a, b, c = (300, 235), (120, 520), (520, 520)
    d.line([a, b, c, a], fill=INK, width=5)
    for label, xy in (("A", (288, 185)), ("B", (82, 520)), ("C", (535, 520))):
        d.text(xy, label, font=FONT, fill=INK)
    d.text((450, 195), "BC", font=FONT, fill=INK)
    d.line([(435, 245), (560, 245)], fill=INK, width=4)
    d.text((440, 252), "sin A", font=FONT, fill=INK)
    d.text((580, 225), "= 2R", font=FONT, fill=INK)
    d.text((500, 330), "6", font=FONT, fill=INK)
    d.line([(480, 378), (540, 378)], fill=INK, width=4)
    d.text((490, 385), "1/2", font=FONT, fill=INK)
    d.text((560, 358), "= 2R", font=FONT, fill=INK)
    d.text((570, 450), "12 = 2R  =>  R = 6", font=FONT, fill=INK)
    d.text((80, 650), "Raspuns: R = 6", font=FONT, fill=INK)
    return img


def flat_triangle_on_grid() -> Image.Image:
    """The real failure: grid paper, a flat obtuse triangle (base 550, height 130, so
    its base is long enough for the ruled-line filter), the apex label 'A' and an
    angle arc '30°' touching the top, fractions on the right."""
    from PIL import ImageFilter

    img = Image.new("RGB", (1500, 2000), (246, 246, 240))
    d = ImageDraw.Draw(img)
    for x in range(0, 1500, 44):
        d.line([(x, 0), (x, 2000)], fill=(130, 145, 170), width=2)
    for y in range(0, 2000, 44):
        d.line([(0, y), (1500, y)], fill=(130, 145, 170), width=2)
    d.text((80, 80), "6) BC = 4, A = 30, B = 45", font=FONT, fill=INK)
    a, b, c = (300, 300), (90, 430), (640, 430)
    d.line([a, b, c, a], fill=INK, width=4)
    d.text((285, 252), "A", font=FONT, fill=INK)
    d.arc((262, 300, 338, 352), 25, 155, fill=INK, width=3)
    d.text((285, 322), "30", font=ImageFont.load_default(size=30), fill=INK)
    d.text((55, 425), "B", font=FONT, fill=INK)
    d.text((650, 425), "C", font=FONT, fill=INK)
    d.text((470, 228), "BC", font=FONT, fill=INK)
    d.line([(455, 278), (575, 278)], fill=INK, width=4)
    d.text((460, 284), "sin A", font=FONT, fill=INK)
    d.text((590, 258), "=", font=FONT, fill=INK)
    d.text((640, 228), "AC", font=FONT, fill=INK)
    d.line([(625, 278), (745, 278)], fill=INK, width=4)
    d.text((630, 284), "sin B", font=FONT, fill=INK)
    d.text((720, 370), "4", font=FONT, fill=INK)
    d.line([(700, 418), (760, 418)], fill=INK, width=4)
    d.text((705, 425), "1/2", font=FONT, fill=INK)
    d.text((775, 398), "= AC / (sqrt2/2)", font=FONT, fill=INK)
    d.text((700, 500), "AC = 4 sqrt2", font=FONT, fill=INK)
    d.text((80, 640), "Raspuns: AC = 4 sqrt2", font=FONT, fill=INK)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def test_flat_triangle_on_grid_paper_with_apex_label_and_arc():
    seg = segment_page(png_bytes(flat_triangle_on_grid()))
    figs = [r for r in seg.regions if r.kind == "diagram"]
    assert len(figs) == 1
    fig = figs[0].bbox
    assert fig.y <= 255 and fig.y1 >= 450 and fig.x <= 60 and fig.x1 >= 660   # A, the arc, B and C included
    first = next(r for r in seg.regions if r.kind == "line" and 200 < r.bbox.y < 300)
    assert first.bbox.x >= 440                                  # "BC / sin A" without the apex or 'A'
    fractions = [r for r in seg.regions if r.kind == "line" and 340 < r.bbox.y < 480]
    assert len(fractions) == 1 and fractions[0].bbox.x >= 680   # "4 / (1/2) = ..." without the triangle


REAL_LAW_OF_SINES = os.environ.get(
    "LAW_OF_SINES_IMAGE", r"C:\Users\raulb\Pictures\Screenshots\Captură de ecran 2026-10-01 204740.png")


@pytest.mark.skipif(not os.path.exists(REAL_LAW_OF_SINES), reason="real screenshot not available (LAW_OF_SINES_IMAGE)")
def test_real_law_of_sines_screenshot():
    """The 595x741 screenshot that failed: grid paper, a flat triangle whose base is a
    1-2 px pen stroke (as thin as the grid), a '2' written on side AC."""
    seg = segment_page(open(REAL_LAW_OF_SINES, "rb").read())
    figs = [r for r in seg.regions if r.kind == "diagram"]
    assert len(figs) == 1
    fig = figs[0].bbox
    assert fig.x <= 60 and fig.x1 >= 230 and fig.y <= 70 and fig.y1 >= 150   # A to the base label 4, B to C
    right = [r for r in seg.regions if r.kind == "line" and r.bbox.x >= 290 and r.bbox.y < 200]
    assert len(right) == 3                                    # BC/sinA = AC/sinB | 4/(1/2) = ... | sin B = 1/4
    assert all(r.bbox.x > fig.x1 - 10 for r in right[:1])     # the first fraction row holds no part of the triangle


# ---- segmenter -----------------------------------------------------------------------

def test_figure_without_gutter_is_one_region_and_formulas_stay_separate():
    seg = segment_page(png_bytes(law_of_sines()))
    figs = [r for r in seg.regions if r.kind == "diagram"]
    assert len(figs) == 1
    fig = figs[0].bbox
    assert fig.y <= 190 and fig.y1 >= 540 and fig.x <= 90 and fig.x1 >= 560  # apex A to base, B to C, labels
    formulas = [r for r in seg.regions if r.kind == "line" and 150 < r.bbox.y < 600]
    assert len(formulas) == 3                                   # BC/sin A, 6/(1/2), 12 = 2R
    assert [r.index for r in formulas] == sorted(r.index for r in formulas)
    assert all(r.index > figs[0].index for r in formulas)      # figure first, then its calculations
    first = formulas[0].bbox
    assert first.x >= 400 and first.y1 < 320                    # BC/sin A without the apex of the triangle


def test_figure_crop_shows_only_the_figure():
    import io

    import numpy as np

    seg = segment_page(png_bytes(law_of_sines()))
    fig = next(r for r in seg.regions if r.kind == "diagram")
    crop = np.asarray(Image.open(io.BytesIO(fig.image_bytes)).convert("L")).astype(int)
    sx, sy = crop.shape[1] / fig.bbox.w, crop.shape[0] / fig.bbox.h
    # "BC" of the first fraction lies inside the figure's bounding box but is blanked
    x0, x1 = int((450 - fig.bbox.x) * sx), int((500 - fig.bbox.x) * sx)
    y0, y1 = int((200 - fig.bbox.y) * sy), int((235 - fig.bbox.y) * sy)
    assert crop[y0:y1, x0:x1].min() > 200
    # while the side AB is kept
    ax, ay = int((210 - fig.bbox.x) * sx), int((377 - fig.bbox.y) * sy)
    assert crop[ay - 6:ay + 6, ax - 6:ax + 6].min() < 120

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


def sign_table_with_conclusions() -> Image.Image:
    """Subiectul 3 layout: a variation table whose rules span ~42% of the page (as long
    as a grid line piece), the f(x) row closed by a bottom rule, and the monotony
    read off the table written in four lines on its right."""
    img = Image.new("RGB", (1500, 2000), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 60), "b) f'(x) = 0 <=> x = 4", font=FONT, fill=INK)
    top, left, width = 220, 60, 630
    for y in (top + 70, top + 150, top + 235):                 # under x, under f'(x), closing f(x)
        d.line([(left, y), (left + width, y + 4)], fill=INK, width=3)
    d.line([(left + 150, top - 5), (left + 152, top + 235)], fill=INK, width=3)
    for x, text in ((left + 30, "x"), (left + 180, "0"), (left + 360, "4"), (left + 540, "+oo")):
        d.text((x, top + 10), text, font=FONT, fill=INK)
    for x, text in ((left, "f'(x)"), (left + 230, "-"), (left + 360, "0"), (left + 480, "+"), (left + 560, "+")):
        d.text((x, top + 90), text, font=FONT, fill=INK)
    d.text((left + 10, top + 170), "f(x)", font=FONT, fill=INK)
    d.line([(left + 200, top + 180), (left + 320, top + 215)], fill=INK, width=3)
    d.line([(left + 400, top + 215), (left + 600, top + 180)], fill=INK, width=3)
    for k, text in enumerate(["f'(x) <= 0 pt. x in (0,4] =>", "=> f(x) descrescatoare pe (0,4]",
                              "f'(x) >= 0 pt. x in [4,oo) =>", "=> f(x) crescatoare pe [4,oo)"]):
        d.text((780, top - 10 + 62 * k), text, font=FONT, fill=INK)
    d.text((80, 620), "c) Cautam asimptota orizontala", font=FONT, fill=INK)
    return img


def test_whole_sign_table_is_one_region_and_text_beside_it_separate_lines():
    seg = segment_page(png_bytes(sign_table_with_conclusions()))
    tables = [r for r in seg.regions if r.kind == "table"]
    assert len(tables) == 1
    t = tables[0].bbox
    assert t.y <= 225 and t.y1 >= 450 and t.x <= 65 and t.x1 >= 680   # x, f'(x) and f(x) rows, +oo column
    assert not [r for r in seg.regions if r.kind == "line" and r.bbox.x < t.x1 and t.y < r.bbox.y < t.y1]
    right = [r for r in seg.regions if r.kind == "line" and r.bbox.x >= 700 and r.bbox.y < 500]
    assert len(right) == 4                                     # one region per conclusion line
    assert all(a.bbox.y1 <= b.bbox.y for a, b in zip(right, right[1:]))   # no overlap
    assert tables[0].index < min(r.index for r in right)      # table first, then what is read off it


REAL_SIGN_TABLE = os.environ.get(
    "SIGN_TABLE_IMAGE", r"C:\Users\raulb\Pictures\Screenshots\Captură de ecran 2026-10-02 112016.png")


@pytest.mark.skipif(not os.path.exists(REAL_SIGN_TABLE), reason="real screenshot not available (SIGN_TABLE_IMAGE)")
def test_real_sign_table_screenshot():
    """The 713x823 screenshot (Subiectul 3): the f(x) row was cut off the table at
    the rule under f'(x), and the four lines on the right overlapped."""
    seg = segment_page(open(REAL_SIGN_TABLE, "rb").read())
    tables = [r for r in seg.regions if r.kind == "table"]
    assert len(tables) == 1
    t = tables[0].bbox
    assert t.y <= 405 and t.y1 >= 510 and t.x <= 20 and t.x1 >= 320   # header to the bottom rule, +oo included
    right = [r for r in seg.regions if r.kind == "line" and r.bbox.x >= 370 and 380 < r.bbox.y < 530]
    assert len(right) == 4
    assert all(a.bbox.y1 <= b.bbox.y for a, b in zip(right, right[1:]))


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
