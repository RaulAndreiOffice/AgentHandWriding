"""Line segmentation on synthetic pages, plus an optional check on real gold pages."""

import io
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.services.pipeline_service import LineTranscription, _merge
from app.services.segmenter import (CROP_MAX_UPSCALE, CROP_TARGET_PIXELS, SegmentationParams,
                                    load_page, segment_page)
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
