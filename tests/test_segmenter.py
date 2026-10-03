"""Line segmentation on synthetic pages, plus an optional check on real gold pages."""

import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.services.pipeline_service import LineTranscription, _merge
from app.services.segmenter import (CROP_MAX_UPSCALE, CROP_TARGET_PIXELS, SegmentationParams, _classify,
                                    _Comp, _is_table, load_page, segment_page)
from app.services.vlm_service import strip_code_fences
from tests.conftest import LINE_PITCH, make_page, png_bytes

LINES = ["1) 2x + 3 = 7", "2x = 4", "x = 2", "2) f(x) = 3x - 1", "f(2) = 5", "S = {2}"]


def tops(seg) -> list[int]:
    return [r.bbox.y for r in seg.regions]


@pytest.mark.parametrize("grid", [False, True], ids=["plain", "grid"])
def test_one_region_per_line_in_reading_order(grid):
    seg = segment_page(png_bytes(make_page(LINES, grid=grid)))
    assert seg.mode == "lines"
    assert len(seg.regions) == len(LINES)
    assert tops(seg) == sorted(tops(seg))
    assert [r.index for r in seg.regions] == list(range(len(LINES)))


def test_grid_lines_do_not_widen_boxes():
    plain = segment_page(png_bytes(make_page(LINES)))
    grid = segment_page(png_bytes(make_page(LINES, grid=True)))
    for a, b in zip(plain.regions, grid.regions):
        assert abs(a.bbox.w - b.bbox.w) < 40 and abs(a.bbox.h - b.bbox.h) < 40


def test_fraction_stays_in_one_region():
    seg = segment_page(png_bytes(make_page(LINES[:3], fraction_after=1)))
    assert seg.mode == "lines"
    assert len(seg.regions) == 4  # 3 lines + the fraction line, not 5
    bar_y = 100 + 2 * LINE_PITCH  # where make_page draws the fraction after line 1
    fraction = next(r.bbox for r in seg.regions if r.bbox.y <= bar_y <= r.bbox.y1)
    # numerator ink starts ~25 px above the bar, denominator ink ends ~60 px below
    assert fraction.y <= bar_y - 25
    assert fraction.y1 >= bar_y + 60


def test_crops_are_upscaled_jpegs_of_the_boxes():
    seg = segment_page(png_bytes(make_page(LINES)))
    for r in seg.regions:
        assert r.mime_type == "image/jpeg"
        img = Image.open(io.BytesIO(r.image_bytes))
        assert img.format == "JPEG"
        scale = img.width / r.bbox.w
        assert 1.0 <= scale <= CROP_MAX_UPSCALE + 0.01  # small line crops are enlarged...
        assert abs(img.height / r.bbox.h - scale) < 0.05  # ...without distortion
        assert img.width * img.height <= CROP_TARGET_PIXELS * 1.02


def test_blank_page_falls_back_to_full_page():
    seg = segment_page(png_bytes(Image.new("RGB", (800, 600), "white")))
    assert seg.mode == "full_page"
    assert len(seg.regions) == 1
    assert seg.regions[0].bbox.as_list() == [0, 0, 800, 600]


def test_single_line_falls_back_to_full_page():
    seg = segment_page(png_bytes(make_page(["x = 2"])))
    assert seg.mode == "full_page"


def test_too_many_lines_falls_back_to_full_page():
    seg = segment_page(png_bytes(make_page(LINES)), SegmentationParams(max_lines=3))
    assert seg.mode == "full_page"


def test_box_merge_gap_joins_split_matrix_rows():
    """Matrix rows written without a bracket spanning both come out as two
    boxes; box_merge_gap_px merges them into one crop, leaving the ordinary
    lines (about 70 px of whitespace apart) alone."""
    img = make_page(["1) A(5) =", "", "det(A) = 21"])
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    draw.text((420, 190), "2    5", font=font, fill=(20, 30, 120))  # row 1, ~15 px above row 2
    draw.text((420, 240), "-5  -2", font=font, fill=(20, 30, 120))
    page = png_bytes(img)

    default = segment_page(page)
    merged = segment_page(page, SegmentationParams(box_merge_gap_px=35))
    assert len(default.regions) == 4  # label, row 1, row 2, det line
    assert len(merged.regions) == 3
    matrix = merged.regions[1].bbox
    assert matrix.y <= 195 and matrix.y1 >= 270  # both rows in one crop


def test_large_box_merge_gap_joins_ordinary_lines():
    page = png_bytes(make_page(LINES))
    assert len(segment_page(page, SegmentationParams(box_merge_gap_px=200)).regions) < len(LINES)


def test_exif_rotation_is_applied():
    img = make_page(LINES, size=(1200, 1600)).rotate(90, expand=True)  # stored sideways
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    page = load_page(buf.getvalue())
    assert page.size == (1200, 1600)
    seg = segment_page(buf.getvalue())
    assert seg.mode == "lines" and len(seg.regions) == len(LINES)


def test_merge_strips_fences_and_skips_empty_and_failed():
    lines = [
        LineTranscription(0, [0, 0, 1, 1], "```latex\n$a$\n```"),
        LineTranscription(1, [0, 0, 1, 1], ""),
        LineTranscription(2, [0, 0, 1, 1], "", error="boom"),
        LineTranscription(3, [0, 0, 1, 1], "$b$\n\n$c$  "),
    ]
    assert _merge(lines) == "$a$\n\n$b$\n$c$"


def test_strip_code_fences():
    assert strip_code_fences("```latex\n$x$\n```") == "$x$"
    assert strip_code_fences("$x$") == "$x$"


# ---- real pages (hw2tex gold set); skipped when not available -------------

GOLD = Path(os.environ.get("HW2TEX_GOLD_DIR", r"D:\Worck\HandWriding_ai\hw2tex\eval\gold\pages"))
GOLD_PAGES = sorted(GOLD.glob("*.png")) if GOLD.is_dir() else []


@pytest.mark.skipif(not GOLD_PAGES, reason=f"gold pages not found in {GOLD} (set HW2TEX_GOLD_DIR)")
@pytest.mark.parametrize("path", GOLD_PAGES, ids=lambda p: p.stem)
def test_gold_page_is_split_into_lines(path):
    """Guards against the 'two huge blocks' failure: every gold page is split
    into many regions and no region covers a large part of the page."""
    seg = segment_page(path.read_bytes())
    assert seg.mode == "lines"
    assert len(seg.regions) >= 8  # the sparsest gold page (tema6_p15) has 8 lines
    tallest = max(r.bbox.h for r in seg.regions)
    assert tallest < 0.25 * seg.page_height, f"region of {tallest}px on a {seg.page_height}px page"


# ---- sign/variation tables -------------------------------------------------

def _variation_table_page() -> Image.Image:
    """Statement lines, then a sign table whose vertical rule bows (Hough sees only a
    chord of it), with the header row "x | -inf -2 2 +inf" above the first rule."""
    img = make_page(["c) f'(x) = 0", "2 - x = 0", "2 + x = 0"])
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    ink = (20, 30, 120)
    top = 470
    for i, (label, row) in enumerate([("x", "-oo    -2     2    +oo"), ("f'(x)", "   -    0  +  0   -"),
                                      ("f(x)", r"   \       /       \ ")]):
        draw.text((110, top + 15 + 75 * i), label, font=font, fill=ink)
        draw.text((260, top + 15 + 75 * i), row, font=font, fill=ink)
    for y in (top + 70, top + 145):  # row rules
        draw.line((90, y, 820, y + 4), fill=ink, width=3)
    # The vertical rule bends left in its lower half.
    draw.line([(235, top), (232, top + 90), (222, top + 150), (212, top + 225)], fill=ink, width=3)
    draw.text((100, top + 300), "f'(x) > 0 pe (-2, 2)", font=font, fill=ink)
    return img


def test_variation_table_is_one_region_with_its_header_row():
    seg = segment_page(png_bytes(_variation_table_page()))
    tables = [r for r in seg.regions if r.kind == "table"]
    assert len(tables) == 1
    t = tables[0].bbox
    assert t.y <= 480 and t.y1 >= 690  # header row (x ...) to the f(x) row
    # No other region cuts through the table's rows.
    for r in seg.regions:
        if r is not tables[0]:
            assert r.bbox.y1 <= t.y + 15 or r.bbox.y >= t.y1 - 15


def _rows_with_crossing_rules(n_rows: int) -> np.ndarray:
    """n_rows rows of glyph-sized marks, a long horizontal stroke under the first row
    and a tall vertical stroke crossing it (a "+")."""
    h = 70 * n_rows
    m = np.zeros((h, 420), np.uint8)
    for r in range(n_rows):
        for x in range(100, 400, 45):
            cv2.rectangle(m, (x, 8 + 75 * r), (x + 22, 40 + 75 * r), 1, 3)
    cv2.line(m, (10, 48), (410, 50), 1, 3)
    cv2.line(m, (60, 2), (61, h - 2), 1, 3)
    return m.astype(bool)


@pytest.mark.parametrize("n_rows, kind", [(1, "line"), (2, "table")])
def test_one_row_of_writing_is_never_a_table(n_rows, kind):
    """'C I  2 - x = 0' with an underline crossed by a tall stroke passes the "+" test
    of a table; it is still one equation. Two rows with the same rules are a table."""
    sub = _rows_with_crossing_rules(n_rows)
    assert _is_table(sub, 24.0)
    assert _classify(_Comp(0, 0, sub.shape[1], sub.shape[0], 0), sub, 24.0) == kind


def test_table_with_faint_rules_on_a_low_resolution_page():
    """A small screenshot: glyphs ~12 px, the table's rules 1 px and light grey (lighter
    than the ink threshold of the writing). Still one table region."""
    img = Image.new("RGB", (700, 640), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=18)
    ink, rule = (40, 45, 110), (150, 155, 185)
    for i, text in enumerate(["c) f'(x) = 0", "2 - x = 0", "2 + x = 0"]):
        draw.text((40, 60 + 50 * i), text, font=font, fill=ink)
    top = 260
    for i, (label, row) in enumerate([("x", "-oo     -2      2     +oo"), ("f'(x)", "  -      0  +  0     -"),
                                      ("f(x)", r"  \        /        \ ")]):
        draw.text((20, top + 8 + 40 * i), label, font=font, fill=ink)
        draw.text((100, top + 8 + 40 * i), row, font=font, fill=ink)
    for y in (top + 36, top + 76):
        draw.line((10, y, 380, y + 1), fill=rule, width=1)
    draw.line((85, top, 86, top + 120), fill=rule, width=1)
    draw.text((40, top + 160), "f'(x) > 0 pe (-2, 2)", font=font, fill=ink)
    seg = segment_page(png_bytes(img))
    tables = [r for r in seg.regions if r.kind == "table"]
    assert len(tables) == 1
    assert tables[0].bbox.y <= top + 8 and tables[0].bbox.y1 >= top + 100


# ---- crossed-out lines ------------------------------------------------------

def _page_with(mark: str) -> tuple[bytes, int]:
    """Three lines; the middle one ("f'(x) < 0 pe (-oo, -2)") is struck through,
    underlined, or a wide fraction. Returns the page and the middle line's top."""
    img = make_page(["c) f'(x) = 0", "", "x = 2"])
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    ink = (20, 30, 120)
    y = 100 + LINE_PITCH
    if mark == "fraction":  # numerator and denominator around a bar across the line
        draw.text((100, y - 30), "x^2 - 2x^2 + 4", font=font, fill=ink)
        draw.line((95, y + 22, 420, y + 22), fill=ink, width=4)
        draw.text((160, y + 30), "(x^2 + 4)^2", font=font, fill=ink)
        return png_bytes(img), y - 30
    # Bold, like pen writing: the strike crosses ink in most of its columns.
    draw.text((100, y), "f'(x)<0 pe(-oo,-2)", font=font, fill=ink, stroke_width=2, stroke_fill=ink)
    right = int(draw.textlength("f'(x)<0 pe(-oo,-2)", font=font)) + 100
    stroke_y = y + 26 if mark == "strike" else y + 52  # through the middle / under the line
    draw.line((100, stroke_y, right, stroke_y + 3), fill=ink, width=3)
    return png_bytes(img), y


def test_crossed_out_line_is_dropped():
    page, y = _page_with("strike")
    seg = segment_page(page)
    assert seg.mode == "lines"
    assert not any(r.bbox.y <= y + 20 <= r.bbox.y1 for r in seg.regions)
    assert len(seg.regions) == 2
    assert len(segment_page(page, SegmentationParams(drop_struck_out=False)).regions) == 3


@pytest.mark.parametrize("mark", ["underline", "fraction"])
def test_long_bar_that_is_not_a_strike_is_kept(mark):
    page, y = _page_with(mark)
    seg = segment_page(page)
    assert len(seg.regions) == 3
    assert any(r.bbox.y <= y + 20 <= r.bbox.y1 for r in seg.regions)


def test_double_underlined_heading_is_kept():
    """'Subiectul 2' underlined twice, the upper line through the bottoms of the letters:
    a heading, not a crossed-out line."""
    img = make_page(["c) f'(x) = 0", "", "x = 2"])
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    ink = (20, 30, 120)
    y = 100 + LINE_PITCH
    draw.text((100, y), "Subiectul 2", font=font, fill=ink, stroke_width=2, stroke_fill=ink)
    right = int(draw.textlength("Subiectul 2", font=font)) + 100
    draw.line((95, y + 28, right, y + 29), fill=ink, width=3)  # through the letters' bottoms
    draw.line((95, y + 46, right, y + 47), fill=ink, width=3)  # the underline below
    seg = segment_page(png_bytes(img))
    assert len(seg.regions) == 3
    assert any(r.bbox.y <= y + 20 <= r.bbox.y1 for r in seg.regions)
