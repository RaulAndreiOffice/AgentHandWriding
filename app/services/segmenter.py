"""Split a photo of a handwritten math page into one crop per line.

Deterministic, no model. Works on clustered connected components rather than a
whole-width projection profile, because on notebook photos (grid paper, margin
spine, shadows, sign tables, check marks between lines) no row of the page is
ever completely empty.

  1. Ink: pixels much darker than the paper *around them* (ratio to a local
     median background). Grid squares and bleed-through are lighter than that.
  2. Clean-up, for grouping only (crops are always cut from the original):
     - straight runs spanning a large part of the page (grid/ruled lines, margin
       rule, page edges) are removed;
     - components touching the image border (desk, page edge, spine shadow),
       long thin vertical strokes in the left/right margin strips (page edge,
       spine), and ink on a dark background near the border are dropped;
     - blobs both very wide and very tall (a curved page edge with the text it
       touches) are dropped;
     - specks and green grader marks (check marks) are dropped.
  3. Blobs: ink is closed horizontally (about half a glyph) so a word or a run
     of symbols is one blob.
  4. Lines: blobs whose vertical spans overlap substantially are one line
     (union-find). A tall blob (brace, table rule, matrix bracket, integral)
     overlaps several lines and binds them into one block: systems, matrices and
     sign tables stay together. A fraction bar binds what is directly above and
     below it, so numerator and denominator stay with their line.
  5. Fragments (dots, accents, short strokes) attach to the nearest line.

Pages with fewer than two lines, or any failure, fall back to the whole page.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

logger = logging.getLogger(__name__)

#: Pages are segmented at this long side; boxes are mapped back to full size.
#: Pixel parameters below are in this working scale.
WORK_LONG_SIDE = 2000

#: Crops are encoded as JPEG at this quality before being sent to the VLM.
JPEG_QUALITY = 92

#: Small line crops are upscaled toward this many pixels (at most CROP_MAX_UPSCALE
#: times) before encoding. vLLM is started with max_pixels 262144 (512x512), so
#: a typical 90 px tall line otherwise reaches the model at a fraction of the
#: resolution it could use; the 2B model reads digits (4 vs u) much better larger.
CROP_TARGET_PIXELS = 250_000
CROP_MAX_UPSCALE = 2.5

#: Near the photo border, ink on a background darker than this share of the
#: paper brightness is off the page (desk, fold shadow, the next sheet).
PAPER_SHADE = 0.8
#: For table detection, a horizontal (vertical) line is notebook grid only when its row
#: (column) carries line runs over this fraction of the page.
TABLE_GRID_SPAN = 0.7


@dataclass(frozen=True)
class SegmentationParams:
    """Tuning knobs. Pixel values are at the working scale (long side 2000 px);
    None means derive from the measured glyph height of the page."""

    #: Final box-merging pass: two boxes that overlap horizontally are merged
    #: into one crop when the vertical gap between their ink is below this
    #: (negative = they overlap vertically). Use it to keep matrix rows or the
    #: parts of a fraction together when they come out as separate boxes, e.g.
    #: 15. Auto: merge only boxes that lie mostly (>50%) inside each other's
    #: band; a positive value also joins ordinary consecutive lines that close.
    box_merge_gap_px: int | None = None
    #: Two blobs are on the same line when their vertical spans overlap by at
    #: least this many pixels. Auto: 40% of the shorter blob's height.
    row_merge_threshold_px: int | None = None
    #: 0 = keep faint strokes, 1 = aggressive. Raises the ink contrast needed,
    #: shortens the kernels that remove grid/ruled lines, drops larger specks.
    grid_filter_strength: float = 0.5
    #: Width of the left/right strips (fraction of page width) searched for a
    #: page edge / notebook spine: long thin vertical strokes there are ignored.
    margin_exclude_frac: float = 0.08
    #: More lines than this: the page is sent whole (likely mis-segmented).
    max_lines: int = 80
    #: Split a block of two or more rows into a left and a right column when an
    #: ink-free vertical gutter at least this wide runs through it (a figure beside
    #: its calculations). None = 2.5 x glyph height.
    column_gap_px: int | None = None
    split_columns: bool = True
    #: Tag regions as "table" (sign/variation tables) or "diagram" (geometric figures).
    classify_regions: bool = True


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def y1(self) -> int:
        return self.y + self.h

    @property
    def x1(self) -> int:
        return self.x + self.w

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


#: "line": one or a few lines of writing; "table": a sign/variation table (horizontal
#: and vertical rules); "diagram": a drawing such as a triangle with its labels.
RegionKind = Literal["line", "table", "diagram"]


@dataclass(frozen=True)
class DetectedLine:
    bbox: BBox
    kind: RegionKind = "line"
    #: For a figure: which pixels of the box belong to it (its strokes and labels), at
    #: the working scale; the rest of the crop - formulas written next to it inside its
    #: bounding box - is blanked. None: keep the whole box.
    keep: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Region:
    index: int
    bbox: BBox
    image_bytes: bytes
    mime_type: str
    kind: RegionKind = "line"


@dataclass(frozen=True)
class Segmentation:
    mode: Literal["lines", "full_page"]
    regions: list[Region]
    page_width: int
    page_height: int


# --------------------------------------------------------------------- ink mask


def _ink(gray: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Pen strokes, against the paper's own local brightness; and that local
    background.

    The background is a median over a window wider than any stroke, taken on a
    1/4 subsampled image so it costs milliseconds.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(3, min(15, min(small.shape) // 2 * 2 - 1))
    bg = cv2.medianBlur(small, k)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return gray.astype(np.float32) < ratio * np.maximum(bg, 1.0), bg


def _remove_long_lines(ink: np.ndarray, strength: float) -> np.ndarray:
    """Remove straight runs spanning a large share of the page: grid and ruled
    lines, the margin rule, page edges. Handwritten table rules and fraction
    bars are shorter and survive."""
    h, w = ink.shape
    share = 0.45 - 0.25 * strength  # 0.45 .. 0.20 of the page
    u8 = ink.astype(np.uint8)
    horiz = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((1, max(40, int(w * share))), np.uint8))
    vert = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((max(40, int(h * share)), 1), np.uint8))
    # Grow the removed lines by a pixel so their anti-aliased fringe goes too.
    lines = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8)).astype(bool)
    return ink & ~lines


def _green_mask(rgb: np.ndarray) -> np.ndarray:
    """Saturated green: grader check marks and scores, not the student's pen."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (hue >= 35) & (hue <= 90) & (sat >= 70) & (val >= 40)  # OpenCV hue is 0..179


def _margin_edges(ink: np.ndarray, frac: float) -> np.ndarray:
    """Ink of page edges, the notebook spine and its shadow line, found in the
    left/right strips (frac of the page width): long, thin, roughly vertical
    strokes (taller than 10% of the page once small breaks are closed, under
    12 ink pixels per row). Only those strokes are returned; labels and text
    next to them in the same strip are kept."""
    h, w = ink.shape
    out = np.zeros_like(ink)
    width = int(w * frac)
    if width < 4:
        return out
    for x0, x1 in ((0, width), (w - width, w)):
        strip = ink[:, x0:x1]
        closed = cv2.morphologyEx(strip.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((max(3, h // 100), 1), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for i in range(1, n):
            tall = stats[i, cv2.CC_STAT_HEIGHT]
            if tall <= 0.1 * h:
                continue
            member = (labels == i) & strip
            if member.sum() / tall < 12:
                out[:, x0:x1] |= member
    return out


@dataclass
class _Comp:
    x0: int
    y0: int
    x1: int
    y1: int
    area: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[_Comp]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = [
        _Comp(int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]),
              int(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]), int(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT]),
              int(s[cv2.CC_STAT_AREA]))
        for s in stats[1:]
    ]
    return labels, comps


def _clean_ink(rgb: np.ndarray, params: SegmentationParams) -> tuple[np.ndarray, float, np.ndarray]:
    """Ink that belongs to the writing, the median glyph height, and the ink before
    long straight lines were removed (long table rules are among them; so is the
    grid, which classification tells apart by where its lines end)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    strength = float(np.clip(params.grid_filter_strength, 0.0, 1.0))
    ink, bg = _ink(gray, ratio=0.7 - 0.2 * strength)  # 0.70 .. 0.50
    # Paper brightness, from the middle of the photo. Near the photo border,
    # ink on a clearly darker background is the desk, a fold shadow or the edge
    # of the next sheet. (Only near the border: a shadow across the page itself
    # must not hide the writing under it.)
    hh, ww = bg.shape
    paper = float(np.median(bg[hh // 4: 3 * hh // 4, ww // 4: 3 * ww // 4]))
    ink &= ~_green_mask(rgb)
    full = ink
    ink = _remove_long_lines(ink, strength)

    h, w = ink.shape
    ink &= ~_margin_edges(ink, params.margin_exclude_frac)
    labels, comps = _components(ink)
    keep = np.zeros(len(comps) + 1, bool)
    for i, c in enumerate(comps, start=1):
        if c.x0 == 0 or c.y0 == 0 or c.x1 >= w or c.y1 >= h:
            continue  # desk, page edge, spine
        if c.h > 0.35 * h or (c.w > 0.8 * w and c.h < 0.05 * h):
            continue  # what is left of a page edge or rule
        cy, cx = (c.y0 + c.y1) // 2, (c.x0 + c.x1) // 2
        near_border = min(cy, h - cy) < 0.05 * h or min(cx, w - cx) < 0.05 * w
        if near_border and bg[cy, cx] < PAPER_SHADE * paper:
            continue  # off the page: desk, fold shadow, another sheet
        keep[i] = True
    ink = keep[labels]

    labels, comps = _components(ink)
    heights = np.array([c.h for c in comps if c.h >= 8 and c.area >= 20])
    glyph = float(np.median(heights)) if heights.size else 0.0
    if glyph == 0.0:
        return ink, 0.0, full

    min_area = max(6, int((0.15 + 0.2 * strength) * glyph) ** 2)
    keep = np.zeros(len(comps) + 1, bool)
    keep[1:] = [c.area >= min_area or c.w >= 0.5 * glyph for c in comps]
    return keep[labels], glyph, full


# ------------------------------------------------------------------- clustering


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _blobs(ink: np.ndarray, glyph: float) -> list[_Comp]:
    """Words / symbol runs: ink closed horizontally by about half a glyph.

    Blobs that are both very wide and very tall are not writing: a curved page
    edge or notebook binding (never straight, so the line filter misses it),
    usually with the text it touches. Dropping one only affects grouping; the
    text it touched still forms lines through its other blobs.
    """
    h, w = ink.shape
    k = max(3, int(glyph * 0.5))
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, k), np.uint8))
    labels, comps = _components(closed)
    out = []
    for i, c in enumerate(comps, start=1):
        if (c.w > 0.5 * w and c.h > 4 * glyph) or (c.h > 0.25 * h and c.w > 4 * glyph):
            continue
        out += _split_touching(labels[c.y0:c.y1, c.x0:c.x1] == i, c, glyph)
    return out


def _split_touching(mask: np.ndarray, c: _Comp, glyph: float) -> list[_Comp]:
    """Two lines whose words touch (a descender on an ascender) come out as one wide,
    tall blob, which would merge the lines. Cut it at its thinnest row in the middle
    when that row carries only a stroke or two - unless a long horizontal run (a
    fraction bar) crosses the middle: a fraction is one blob by design."""
    if c.h < 2.2 * glyph or c.w < 4 * glyph:
        return [c]
    rows = mask.sum(axis=1)
    lo, hi = int(0.3 * c.h), int(0.7 * c.h)
    bar = cv2.morphologyEx(mask[lo:hi].astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((1, max(3, int(0.5 * c.w))), np.uint8))
    if bar.any():
        return [c]
    cut = lo + int(np.argmin(rows[lo:hi]))
    if rows[cut] > max(3, 0.25 * glyph):
        return [c]
    parts = []
    for y0, part in ((0, mask[:cut]), (cut + 1, mask[cut + 1:])):
        ys, xs = np.nonzero(part)
        if not len(ys):
            continue
        sub = _Comp(c.x0 + int(xs.min()), c.y0 + y0 + int(ys.min()), c.x0 + int(xs.max()) + 1,
                    c.y0 + y0 + int(ys.max()) + 1, int(len(ys)))
        parts.append((sub, part[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    if len(parts) < 2 or min(p.h for p, _ in parts) < 0.8 * glyph:
        return [c]
    return [q for p, m in parts for q in _split_touching(m, p, glyph)]


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return min(a1, b1) - max(a0, b0)


def _is_bar(c: _Comp, glyph: float) -> bool:
    """A fraction bar or minus/underline: long, flat."""
    return c.w >= 0.8 * glyph and c.h <= 0.45 * glyph and c.w >= 3 * c.h


def _join_bar_pieces(blobs: list[_Comp], glyph: float) -> list[_Comp]:
    """A fraction bar drawn with a pen lift (or thinned to a dotted line by a downscaled
    screenshot) comes out in pieces, each too short to span the numerator. Join flat
    pieces at the same height less than half a glyph apart."""
    bars = sorted((b for b in blobs if _is_bar(b, glyph) or (b.h <= 0.3 * glyph and b.w >= 0.5 * glyph)),
                  key=lambda b: b.x0)
    rest = [b for b in blobs if b not in bars]
    joined: list[_Comp] = []
    for b in bars:
        for k, a in enumerate(joined):
            # Level with each other (a minus sign just before a fraction sits a little higher).
            if b.x0 - a.x1 <= 0.5 * glyph and abs((a.y0 + a.y1) - (b.y0 + b.y1)) / 2 <= max(2.0, 0.1 * glyph) \
                    and _is_bar(_union_box([a, b]), glyph):
                joined[k] = _union_box([a, b])
                break
        else:
            joined.append(b)
    return rest + joined


def _coverage(box: _Comp, blobs: list[_Comp]) -> float:
    """Share of the box's width covered by the blobs inside it: a row of scattered
    limits or exponents covers little, a written line most of its width."""
    covered = np.zeros(max(1, box.w), bool)
    for b in blobs:
        if box.x0 <= (b.x0 + b.x1) / 2 <= box.x1 and box.y0 <= (b.y0 + b.y1) / 2 <= box.y1:
            covered[max(0, b.x0 - box.x0):max(0, b.x1 - box.x0)] = True
    return float(covered.mean())


def _absorb_fragments(lines: list[_Comp], glyph: float, blobs: list[_Comp]) -> list[_Comp]:
    """Rows of loose marks belong to the line they touch: the limits of an evaluation bar
    |_0^1 or of an integral, exponents and an annotation above a formula, whose bar or
    sign was too faint to tie them to it. A fragment is low (<= 1.6 glyphs), carries
    little ink (<= 1.5 glyph areas), is much
    narrower than its line (<= 60%) and touches or overlaps it - or, for a row of
    scattered marks (covering <= 40% of its width: the 0s under two integrals), sits
    within half a glyph of it. Limits and exponents sit over the inside of their line;
    a short written line ("2x >= 1" above "x >= 1/2 => ...") starts where the next line
    starts, has white below it and covers its width. The line on the fragment's other side
    must keep its distance (>= 0.5 glyph): a short line packed between two others
    ("AC = 16") touches both and stays a line."""
    changed = True
    while changed:
        changed = False
        for f in sorted(lines, key=lambda c: c.w):
            # A few marks (<= 1.5 glyphs of ink: limits, an exponent, a lone numerator),
            # not a short written line ("2x >= 1" carries several glyphs).
            if f.h > 1.6 * glyph or f.area > 1.5 * glyph * glyph:
                continue
            reach = 0.5 * glyph if _coverage(f, blobs) <= 0.4 else 0.0
            best, best_gap = None, reach
            for ln in lines:
                if ln is f or f.w > 0.6 * ln.w or f.x0 < ln.x0 - glyph or f.x1 > ln.x1 + glyph:
                    continue
                if f.x0 <= ln.x0 + glyph:  # starts where the line starts: a short line of its own
                    continue
                gap = max(ln.y0 - f.y1, f.y0 - ln.y1)
                if gap <= best_gap:
                    best, best_gap = ln, gap
            if best is not None:
                below = best.y0 >= (f.y0 + f.y1) / 2  # the fragment hangs above its line
                other = [max(ln.y0 - f.y1, f.y0 - ln.y1) for ln in lines
                         if ln is not f and ln is not best and _overlap(ln.x0, ln.x1, f.x0, f.x1) > 0
                         and ((ln.y1 <= best.y0) if below else (ln.y0 >= best.y1))]
                if other and min(other) < 0.5 * glyph:
                    continue
                lines = [ln for ln in lines if ln is not f and ln is not best] + [_union_box([best, f])]
                changed = True
                break
    return lines


def _cluster_lines(blobs: list[_Comp], glyph: float, params: SegmentationParams) -> list[_Comp]:
    blobs = _join_bar_pieces(blobs, glyph)
    n = len(blobs)
    uf = _UnionFind(n)
    bars = [_is_bar(b, glyph) for b in blobs]
    small = [b.h < 0.5 * glyph and not bars[i] for i, b in enumerate(blobs)]
    tall = [b.h > 1.8 * glyph and not small[i] and not bars[i] for i, b in enumerate(blobs)]
    core = [not (small[i] or bars[i] or tall[i]) for i in range(n)]

    def same_row(a: _Comp, b: _Comp) -> bool:
        ov = _overlap(a.y0, a.y1, b.y0, b.y1)
        if params.row_merge_threshold_px is not None:
            if ov < params.row_merge_threshold_px:
                return False
        elif ov < 0.4 * min(a.h, b.h):
            return False
        # The shorter blob's centre must lie inside the taller one, or two
        # tightly packed lines chain through their ascenders and descenders.
        short, high = (a, b) if a.h <= b.h else (b, a)
        cy = (short.y0 + short.y1) / 2
        return high.y0 <= cy <= high.y1

    # 1. Core lines: blobs of ordinary height that share a row.
    order = sorted(range(n), key=lambda i: blobs[i].y0)
    for oi, i in enumerate(order):
        if not core[i]:
            continue
        a = blobs[i]
        for j in order[oi + 1:]:
            b = blobs[j]
            if b.y0 >= a.y1:
                break
            if core[j] and same_row(a, b):
                uf.union(i, j)

    def core_lines() -> dict[int, _Comp]:
        groups: dict[int, list[_Comp]] = {}
        for i in range(n):
            if core[i]:
                groups.setdefault(uf.find(i), []).append(blobs[i])
        return {root: _union_box(g) for root, g in groups.items()}

    # 2. Fraction bars bind the closest blob(s) right above and right below
    #    within the bar's width (numerator, denominator), and the row they sit in.
    reach = 0.9 * glyph
    for i in (i for i in range(n) if bars[i]):
        bar = blobs[i]
        cx0, cx1 = bar.x0 - 0.25 * glyph, bar.x1 + 0.25 * glyph
        inside = [j for j in range(n) if j != i and not small[j] and not bars[j]
                  and blobs[j].x0 >= cx0 and blobs[j].x1 <= cx1]
        above = [(bar.y0 - blobs[j].y1, j) for j in inside
                 if blobs[j].y1 <= bar.y0 + 3 and bar.y0 - blobs[j].y1 <= reach]
        below = [(blobs[j].y0 - bar.y1, j) for j in inside
                 if blobs[j].y0 >= bar.y1 - 3 and blobs[j].y0 - bar.y1 <= reach]
        if not (above and below):
            continue
        nearest = [j for side in (above, below) for d, j in side if d <= min(side)[0] + 0.3 * glyph]
        for j in nearest:
            uf.union(i, j)
        cy = (bar.y0 + bar.y1) / 2
        for j in range(n):  # the row the bar sits in: "x = <bar> + 1"
            b = blobs[j]
            if core[j] and b.y0 <= cy <= b.y1 and _overlap(b.x0, b.x1, bar.x0 - 2 * glyph, bar.x1 + 2 * glyph) > 0:
                uf.union(i, j)

    # 3. Tall blobs (braces, brackets, table rules, integrals, a handwritten f):
    #    bind the lines they fully contain; a blob that contains no whole line
    #    belongs to the one line it overlaps most.
    tol = 0.3 * glyph
    for i in (i for i in range(n) if tall[i]):
        t = blobs[i]
        lines = core_lines()
        near = {r: ln for r, ln in lines.items()
                if max(ln.x0 - t.x1, t.x0 - ln.x1) <= 4 * glyph}
        contained = [r for r, ln in near.items() if ln.y0 >= t.y0 - tol and ln.y1 <= t.y1 + tol]
        if len(contained) >= 2:
            for r in contained:
                uf.union(i, r)
        else:
            best = max(near.items(), key=lambda kv: _overlap(kv[1].y0, kv[1].y1, t.y0, t.y1), default=None)
            if best is not None and _overlap(best[1].y0, best[1].y1, t.y0, t.y1) > 0:
                uf.union(i, best[0])
        core[i] = True  # from now on part of the line it joined (or its own)

    for i in (i for i in range(n) if bars[i]):
        core[i] = True

    lines = list(core_lines().values())

    # 3. Lines that lie mostly inside each other's band, or share a row, are one.
    lines = _merge_close(lines, None)
    lines = _merge_same_row(lines, glyph)

    # 4. Fragments go to the nearest line, if one is close; lone ones are noise.
    #    Clusters shorter than ~a glyph (an exponent, a lone minus) count too.
    stubs = [ln for ln in lines if ln.h < 0.8 * glyph]
    lines = [ln for ln in lines if ln.h >= 0.8 * glyph] or stubs
    if lines is stubs:
        stubs = []
    for f in [blobs[i] for i in range(n) if small[i]] + stubs:
        best, best_d = None, 1.0 * glyph
        for k, ln in enumerate(lines):
            # A leading "=" or "-" can sit a few glyphs left of its line.
            if _overlap(ln.x0, ln.x1, f.x0 - 3 * glyph, f.x1 + 3 * glyph) <= 0:
                continue
            d = max(0, ln.y0 - f.y1, f.y0 - ln.y1)
            if d < best_d:
                best, best_d = k, d
        if best is not None:
            lines[best] = _union_box([lines[best], f])

    # Limits, exponents and other loose marks join the line they belong to; then drop
    # lines with too little ink to be writing (a stray mark, a smudge).
    lines = _absorb_fragments(lines, glyph, blobs)
    lines = [ln for ln in lines if ln.area >= glyph * glyph * 0.5]

    # 5. Box merging: fragments have grown the boxes, so boxes may now overlap
    #    or sit within box_merge_gap_px of each other (split matrix rows, a
    #    fraction cut from its row). Merge them so each goes to the VLM whole.
    return _merge_boxes(lines, glyph, params.box_merge_gap_px)


def _merge_boxes(lines: list[_Comp], glyph: float, gap_px: int | None) -> list[_Comp]:
    """Merge until stable: stacked boxes closer than gap_px, and pieces of one row."""
    while True:
        n = len(lines)
        lines = _merge_same_row(_merge_close(lines, gap_px), glyph)
        if len(lines) == n:
            return lines


def _union_box(cs: list[_Comp]) -> _Comp:
    return _Comp(min(c.x0 for c in cs), min(c.y0 for c in cs),
                 max(c.x1 for c in cs), max(c.y1 for c in cs), sum(c.area for c in cs))


def _merge_close(lines: list[_Comp], min_gap: int | None) -> list[_Comp]:
    """Merge lines that overlap horizontally and sit too close vertically.

    gap = whitespace between the two lines (negative when they overlap). With
    min_gap set, lines with gap < min_gap merge. Auto (None): merge only when
    one line lies mostly inside the other's band (overlap > half the shorter
    line); in dense math neighbouring lines legitimately overlap by a glyph.
    """
    def close(a: _Comp, b: _Comp) -> bool:
        if _overlap(a.x0, a.x1, b.x0, b.x1) <= 0:
            return False
        ov = _overlap(a.y0, a.y1, b.y0, b.y1)
        if min_gap is not None:
            return -ov < min_gap
        return ov > 0.5 * min(a.h, b.h)

    changed = True
    while changed:
        changed = False
        lines.sort(key=lambda c: c.y0)
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                if b.y0 >= a.y1 + max(0, min_gap or 0):
                    break  # sorted by y0: every later line starts lower still
                if close(a, b):
                    lines[i] = _union_box([a, b])
                    del lines[j]
                    changed = True
                    break
            if changed:
                break
    return lines


def _merge_same_row(lines: list[_Comp], glyph: float) -> list[_Comp]:
    """Join pieces of one row: they share most of their height and are less
    than 2 glyphs apart ("<fraction>   = 0 => ..."), or overlap horizontally
    (a fraction whose box was cut out of the middle of its row)."""
    lines = sorted(lines, key=lambda c: c.x0)
    merged = True
    while merged:
        merged = False
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue
                a, b = lines[i], lines[j]
                hgap = max(a.x0, b.x0) - min(a.x1, b.x1)
                if hgap < 2 * glyph and _overlap(a.y0, a.y1, b.y0, b.y1) > 0.5 * min(a.h, b.h):
                    lines[i] = _union_box([a, b])
                    del lines[j]
                    merged = True
                    break
            if merged:
                break
    return lines


def _find_gutter(ink: np.ndarray, box: _Comp, min_gap: int, min_side: int) -> tuple[int, int] | None:
    """The widest ink-free vertical band inside `box` (page x range), at least
    `min_gap` wide and leaving at least `min_side` px of the box on each side."""
    cols = ink[box.y0:box.y1, box.x0:box.x1].any(axis=0)
    best = None
    for start, stop in _runs(~cols):
        if stop - start < min_gap or start < min_side or box.w - stop < min_side:
            continue
        if best is None or stop - start > best[1] - best[0]:
            best = (start, stop)
    return (box.x0 + best[0], box.x0 + best[1]) if best else None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _rows_aligned(left: list[_Comp], right: list[_Comp], glyph: float, blobs: list[_Comp]) -> bool:
    """Both sides have the same rows: a matrix or table read across, or one formula with
    a wide space in it - not two columns. Rows of a few loose marks (the limits of an
    evaluation bar |_0^1, exponents: low, covering little of their width) do not count:
    they belong to a row beside them."""
    def real(rows: list[_Comp]) -> list[_Comp]:
        kept = [r for r in rows if r.h > 1.6 * glyph or _coverage(r, blobs) > 0.4]
        return kept or rows

    left, right = real(left), real(right)
    if len(left) != len(right):
        return False
    left, right = sorted(left, key=lambda c: c.y0), sorted(right, key=lambda c: c.y0)
    return all(_overlap(a.y0, a.y1, b.y0, b.y1) > 0.5 * min(a.h, b.h) for a, b in zip(left, right))


def _split_columns(lines: list[_Comp], blobs: list[_Comp], ink: np.ndarray, glyph: float,
                   params: SegmentationParams) -> list[list[_Comp]]:
    """Reading units: each line on its own, or - for a block with a figure beside
    its calculations - the left column's lines followed by the right column's.

    A block is split when it is at least two rows tall, an ink-free vertical gutter
    (column_gap_px, 2.5 glyphs) runs through it with at least 3.5 glyphs on each
    side, and the two sides do not have the same rows (that would be a matrix:
    the space between its columns is a gutter too). Each side is re-clustered, so
    the formulas beside a triangle become separate lines again.
    """
    min_gap = params.column_gap_px if params.column_gap_px is not None else int(2.5 * glyph)
    min_side = int(3.5 * glyph)
    units: list[list[_Comp]] = []
    for ln in lines:
        gutter = None
        if ln.h >= 1.8 * glyph and ln.w >= 2 * min_side + min_gap:
            gutter = _find_gutter(ink, ln, min_gap, min_side)
        if gutter is None:
            units.append([ln])
            continue
        mid = (gutter[0] + gutter[1]) / 2
        inside = [b for b in blobs if ln.x0 <= (b.x0 + b.x1) / 2 < ln.x1 and ln.y0 <= (b.y0 + b.y1) / 2 < ln.y1]
        left = _cluster_lines([b for b in inside if (b.x0 + b.x1) / 2 < mid], glyph, params)
        right = _cluster_lines([b for b in inside if (b.x0 + b.x1) / 2 >= mid], glyph, params)
        if not left or not right or _rows_aligned(left, right, glyph, inside):
            units.append([ln])
            continue
        units.append(sorted(left, key=lambda c: c.y0) + sorted(right, key=lambda c: c.y0))
    return units


def _classify(box: _Comp, full: np.ndarray, glyph: float) -> RegionKind:
    """table (a sign table) or diagram (a figure such as a triangle), else line;
    judged on all ink inside the box, long table rules included."""
    sub = full[box.y0:box.y1, box.x0:box.x1]
    if _is_table(sub, glyph):
        return "table"
    if _is_diagram(sub, glyph):
        return "diagram"
    return "line"


def _segments(sub: np.ndarray, min_len: int, glyph: float) -> list[tuple[int, int, int, int]]:
    """Straight stretches of ink (probabilistic Hough): handwritten rules tilt and bend."""
    segs = cv2.HoughLinesP(sub.astype(np.uint8) * 255, 1, np.pi / 180, threshold=int(glyph),
                           minLineLength=min_len, maxLineGap=max(3, int(glyph / 2)))
    return [] if segs is None else [tuple(int(v) for v in s) for s in segs.reshape(-1, 4)]


def _is_table(sub: np.ndarray, glyph: float) -> bool:
    """A sign/variation table: a long near-horizontal rule and a long near-vertical
    rule that cross - each continues past the other on both sides (the "+" of every
    such table). A right triangle's legs only meet at their ends (an "L").

    Tuned for precision on the gold set (no false tables, incl. grid paper and
    crossed-out lines): a table it misses is simply transcribed as a line.
    """
    h, w = sub.shape
    h_len = max(int(4 * glyph), int(0.3 * w))
    v_len = max(int(1.5 * glyph), int(0.35 * h))
    if w <= h_len or h <= v_len:
        return False
    horiz, vert = [], []
    for x0, y0, x1, y1 in _segments(sub, min(h_len, v_len), glyph):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dx >= h_len and dy <= 0.18 * dx:
            horiz.append((min(x0, x1), max(x0, x1), (y0 + y1) / 2))
        elif dy >= v_len and dx <= 0.18 * dy:
            vert.append(((x0 + x1) / 2, min(y0, y1), max(y0, y1)))
    m = 0.7 * glyph
    return any(hx0 + m <= vx <= hx1 - m and vy0 + m <= hy <= vy1 - m
               for hx0, hx1, hy in horiz for vx, vy0, vy1 in vert)


def _is_diagram(sub: np.ndarray, glyph: float) -> bool:
    """A drawing such as a triangle: mostly line art (>= 55% of the ink in large thin
    sparse strokes), few pieces (a figure and its labels, <= 15 components; text
    has dozens), and a long oblique stroke (20-70 degrees, >= 40% of the region's
    size) meeting another stroke at an end - a hypotenuse meeting a leg.

    Measured on the gold set: triangles 0.60-0.67 line art with 5-11 components;
    a sign table 0.52 / 27, crossed-out lines 0.31-0.47 / 36-80.
    """
    _, comps = _components(sub)
    total = sum(c.area for c in comps)
    line_art = sum(c.area for c in comps
                   if max(c.w, c.h) >= 3.5 * glyph and min(c.w, c.h) >= 2.2 * glyph
                   and c.area < 0.12 * c.w * c.h)
    if not total or line_art < 0.55 * total or len(comps) > 15:
        return False

    def angle(sg) -> float:
        return abs(np.degrees(np.arctan2(sg[3] - sg[1], sg[2] - sg[0]))) % 180

    h, w = sub.shape
    segs = _segments(sub, int(2.5 * glyph), glyph)
    near = 0.8 * glyph
    long_oblique = [a for a in segs if 20 <= min(angle(a), 180 - angle(a)) <= 70
                    and np.hypot(a[2] - a[0], a[3] - a[1]) >= 0.4 * max(w, h)]

    def dist(pt, b) -> float:
        """Distance from a point to segment b."""
        (px, py), (x0, y0, x1, y1) = pt, b
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / max(1, dx * dx + dy * dy)))
        return float(np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))

    def meets(end, a) -> bool:
        """Another stroke, at a clearly different angle, passes this end of `a`
        (Hough segments rarely end exactly at the corner)."""
        for b in segs:
            diff = abs(angle(a) - angle(b))
            if b is not a and min(diff, 180 - diff) >= 25 and dist(end, b) <= near:
                return True
        return False

    return any(meets(a[:2], a) or meets(a[2:], a) for a in long_oblique)


@dataclass
class _Figure:
    box: _Comp            # the drawing and its labels
    mask: np.ndarray      # page-sized mask of the drawing's strokes
    labels: list[_Comp] = field(default_factory=list)

    def keep(self, box: BBox, glyph: float) -> np.ndarray:
        """Pixels of `box` that belong to the figure: its strokes (widened a little) and labels."""
        keep = cv2.dilate(self.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        pad = max(2, int(0.2 * glyph))
        for b in self.labels:
            keep[max(0, b.y0 - pad):b.y1 + pad, max(0, b.x0 - pad):b.x1 + pad] = True
        return keep[box.y:box.y1, box.x:box.x1]


def _corners(segs: list[tuple[int, int, int, int]], glyph: float) -> int:
    """Distinct points where two long straight strokes at clearly different angles meet
    (an end of one lies on the other): a triangle has 3, a square root sign 1."""
    def angle(sg) -> float:
        return np.degrees(np.arctan2(sg[3] - sg[1], sg[2] - sg[0])) % 180

    def dist(pt, b) -> float:
        (px, py), (x0, y0, x1, y1) = pt, b
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / max(1, dx * dx + dy * dy)))
        return float(np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))

    points: list[tuple[float, float]] = []
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            diff = abs(angle(a) - angle(b))
            if min(diff, 180 - diff) < 12:  # an obtuse apex is still a corner
                continue
            for end in (a[:2], a[2:]):
                if dist(end, b) <= 0.8 * glyph and all(np.hypot(end[0] - q[0], end[1] - q[1]) > glyph for q in points):
                    points.append(end)
            for end in (b[:2], b[2:]):
                if dist(end, a) <= 0.8 * glyph and all(np.hypot(end[0] - q[0], end[1] - q[1]) > glyph for q in points):
                    points.append(end)
    return len(points)


def _crossing(segs: list[tuple[int, int, int, int]], glyph: float) -> bool:
    """Two long strokes that cross well inside both (a table's rules, a strike through
    an integral). A polygon's sides meet at their ends - a hand-drawn side may
    overshoot a corner by about a glyph, a table's rules run on by a cell or more."""
    m = 1.5 * glyph
    for i, (ax0, ay0, ax1, ay1) in enumerate(segs):
        for bx0, by0, bx1, by1 in segs[i + 1:]:
            d = (ax1 - ax0) * (by1 - by0) - (ay1 - ay0) * (bx1 - bx0)
            if abs(d) < 1e-6:
                continue
            t = ((bx0 - ax0) * (by1 - by0) - (by0 - ay0) * (bx1 - bx0)) / d
            u = ((bx0 - ax0) * (ay1 - ay0) - (by0 - ay0) * (ax1 - ax0)) / d
            la, lb = np.hypot(ax1 - ax0, ay1 - ay0), np.hypot(bx1 - bx0, by1 - by0)
            if m / la < t < 1 - m / la and m / lb < u < 1 - m / lb:
                return True
    return False


def _figure_ink(full: np.ndarray, glyph: float, span: float = 0.4) -> np.ndarray:
    """Ink for finding figures: everything before the long-line filter (that filter
    also removes a flat triangle's long base), minus the notebook grid.

    A grid line continues across the page - broken by writing, but on the same row
    (or column): a horizontal piece is grid when its row carries horizontal runs over
    at least 40% of the page width (vertical likewise). A pen-drawn base exists only
    for its own length. (Stroke thickness does not separate them: on a 600 px
    screenshot pen strokes are 1-2 px, as thin as the grid.) Rows within 2 px of a
    grid row count too, for slightly tilted photos.
    """
    u8 = full.astype(np.uint8)
    h, w = full.shape
    run = max(int(2 * glyph), 20)
    horiz = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)).astype(bool)
    vert = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8)).astype(bool)
    grid_rows = horiz.sum(axis=1) >= span * w
    grid_cols = vert.sum(axis=0) >= span * h
    grid_rows = np.convolve(grid_rows, np.ones(5), mode="same") > 0
    grid_cols = np.convolve(grid_cols, np.ones(5), mode="same") > 0
    grid = (horiz & grid_rows[:, None]) | (vert & grid_cols[None, :])
    return full & ~grid


def _is_triangle_outline(strokes: np.ndarray, glyph: float) -> bool:
    """A closed triangle, however flat: the convex hull of the strokes has three
    prominent corners and most of the ink lies along the hull's edges (a hollow
    outline; labels or an angle arc touching it may sit inside or just outside)."""
    pts = cv2.findNonZero(strokes.astype(np.uint8))
    if pts is None or len(pts) < 20:
        return False
    hull = cv2.convexHull(pts)
    # The smallest triangle around the strokes: a triangle's hull nearly fills it, even
    # when a label or a short mark at a vertex adds a hull corner of its own.
    area, tri = cv2.minEnclosingTriangle(pts)
    if area < (2 * glyph) ** 2 or cv2.contourArea(hull) < 0.8 * area:
        return False
    corners = np.round(tri).astype(np.int32).reshape(3, 1, 2)
    edge = np.zeros(strokes.shape, np.uint8)
    cv2.polylines(edge, [corners], True, 1, thickness=max(5, int(0.6 * glyph)))
    if (strokes & edge.astype(bool)).sum() < 0.6 * strokes.sum():
        return False
    # Each of the three sides is drawn: a strike-through covers one hull edge at most,
    # a cursive word's ink lies inside its hull rather than along the edges.
    # The fitted triangle runs a little off a hand-drawn side (a label written on a side
    # bulges the hull), more so on longer sides: allow ~6% of the side, >= 1/4 glyph.
    pts3 = corners.reshape(3, 2)
    side = max(np.hypot(*(pts3[k] - pts3[(k + 1) % 3])) for k in range(3))
    reach = max(1, int(0.25 * glyph), int(0.06 * side))
    near = cv2.dilate(strokes.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=reach)
    for k in range(3):
        (x0, y0), (x1, y1) = pts3[k], pts3[(k + 1) % 3]
        samples = np.linspace(0.25, 0.75, 30)  # the middle of the side: corners may carry labels
        xs = np.clip((x0 + samples * (x1 - x0)).astype(int), 0, strokes.shape[1] - 1)
        ys = np.clip((y0 + samples * (y1 - y0)).astype(int), 0, strokes.shape[0] - 1)
        if near[ys, xs].mean() < 0.6:
            return False
    return True


def _find_figures(ink: np.ndarray, glyph: float) -> list[_Figure]:
    """Drawings on the page (a triangle, a polygon), found as whole objects before the
    text is clustered, so a figure is never split between lines (its apex with the
    formula beside it, its base with the next one) nor merged with the formulas
    around it - with or without a clear vertical gutter.

    A drawing is one connected stroke set (strokes closed by a small dilation, pen
    lifts) that is large in both directions (>= 4 x 2.5 glyphs), sparse (< 12% filled)
    and has at least two corners between long straight strokes (a triangle has three;
    a square root sign, a brace, a strike-through or a fraction bar fewer). A sign
    table's rules cross instead of meeting at corners, so tables are excluded.
    """
    joined = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8))
    labels, comps = _components(joined)
    h, w = ink.shape
    figures = []
    for i, c in enumerate(comps, start=1):
        if max(c.w, c.h) < 4 * glyph or min(c.w, c.h) < 1.8 * glyph:  # flat triangles are low
            continue
        if c.x0 <= 0.01 * w or c.y0 <= 0.01 * h or c.x1 >= 0.99 * w or c.y1 >= 0.99 * h or c.w > 0.6 * w:
            continue  # page edge, desk, spine
        strokes = (labels[c.y0:c.y1, c.x0:c.x1] == i) & ink[c.y0:c.y1, c.x0:c.x1]
        if strokes.sum() >= 0.12 * c.w * c.h or _is_table(strokes, glyph):
            continue
        segs = [sg for sg in _segments(strokes, int(2 * glyph), glyph)
                if np.hypot(sg[2] - sg[0], sg[3] - sg[1]) >= 2.5 * glyph]
        if len(segs) < 2 or _crossing(segs, glyph):
            continue
        # A triangle always has a long oblique side (a right triangle its hypotenuse, a
        # flat one its two short sides at 8-20 degrees); a table's rules are horizontal
        # and vertical, its arrows short.
        if not any(8 <= (np.degrees(np.arctan2(abs(y1 - y0), abs(x1 - x0)))) <= 82
                   and np.hypot(x1 - x0, y1 - y0) >= 0.3 * max(c.w, c.h) for x0, y0, x1, y1 in segs):
            continue
        # Most of a figure's ink lies on its straight sides (an apex label or an angle arc
        # may touch them); crossed-out text or a circled grade has a long stroke through
        # or around writing that is not.
        sides = np.zeros(strokes.shape, np.uint8)
        for x0, y0, x1, y1 in segs:
            cv2.line(sides, (x0, y0), (x1, y1), 1, thickness=max(5, int(glyph / 4)))
        if (strokes & sides.astype(bool)).sum() < 0.6 * strokes.sum():
            continue
        # Finally the shape itself: a closed triangle, every side drawn (however flat).
        if not _is_triangle_outline(strokes, glyph):
            continue
        mask = np.zeros_like(ink)
        mask[c.y0:c.y1, c.x0:c.x1] = strokes
        figures.append(_Figure(_Comp(c.x0, c.y0, c.x1, c.y1, int(strokes.sum())), mask))
    return figures


def _horizontal_runs(ink: np.ndarray, glyph: float) -> list[tuple[float, float, float, float]]:
    """Horizontal strokes at least 2.5 glyphs long, as end points, following a rule
    that bows (Hough sees only short chords of it): over 2 glyphs a hand-drawn rule
    drifts by a pixel or two, which a 3 px vertical dilation absorbs."""
    run = max(3, int(2 * glyph))
    tall = cv2.dilate(ink.astype(np.uint8), np.ones((3, 1), np.uint8))
    rules = cv2.morphologyEx(tall, cv2.MORPH_OPEN, np.ones((1, run), np.uint8))
    labels, comps = _components(rules.astype(bool))
    out = []
    end = max(2, int(0.5 * glyph))
    for i, c in enumerate(comps, start=1):
        if c.w < 2.5 * glyph or c.h > 0.18 * c.w + 3:
            continue
        sub = labels[c.y0:c.y1, c.x0:c.x1] == i
        ys = np.arange(c.y0, c.y1)[:, None]
        left, right = sub[:, :end], sub[:, -end:]
        out.append((float(c.x0), float((ys * left).sum() / max(1, left.sum())),
                    float(c.x1 - 1), float((ys * right).sum() / max(1, right.sum()))))
    return out


def _find_tables(ink: np.ndarray, glyph: float) -> list[_Comp]:
    """Sign/variation tables on the page, found from their rules before the text is
    clustered, so every row (x, f'(x), f(x) with its arrows) stays in one region
    instead of being cut at the internal horizontal rules.

    A table is a group of long horizontal rules (>= 5 glyphs) and vertical rules
    (>= 2.5 glyphs) touching each other, at least one pair crossing (the "+" of
    every such table, as in _is_table). Its box spans all the rules: the vertical
    rule runs from the header row to the last row, the bottom rule closes it.
    """
    segs = _segments(ink, int(2 * glyph), glyph)
    # Rules as end points: horizontal ones left to right, vertical ones top to bottom.
    # A photographed rule tilts by a few degrees, so heights are read where they matter.
    horiz: list[tuple[float, float, float, float]] = []
    vert: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in segs:
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dx >= 2.5 * glyph and dy <= 0.18 * dx:  # pieces; long enough once joined
            horiz.append((x0, y0, x1, y1) if x0 <= x1 else (x1, y1, x0, y0))
        elif 2.5 * glyph <= dy <= 0.4 * ink.shape[0] and dx <= 0.18 * dy:  # not a margin line
            vert.append((x0, y0, x1, y1) if y0 <= y1 else (x1, y1, x0, y0))
    horiz += _horizontal_runs(ink, glyph)
    if not horiz or not vert:
        return []

    def y_at(r, x: float) -> float:
        x0, y0, x1, y1 = r
        t = 0.0 if x1 == x0 else min(1.0, max(0.0, (x - x0) / (x1 - x0)))
        return y0 + t * (y1 - y0)

    def x_at(r, y: float) -> float:
        x0, y0, x1, y1 = r
        t = 0.0 if y1 == y0 else min(1.0, max(0.0, (y - y0) / (y1 - y0)))
        return x0 + t * (x1 - x0)

    # A hand-drawn rule comes out of Hough in pieces (broken where it bends or where
    # the pen lifted, duplicated where it is thick): join pieces that continue each
    # other - at most 3 glyphs apart, at the same height where they meet.
    joined = True
    while joined:
        joined = False
        horiz.sort(key=lambda r: r[0])
        for i, q in enumerate(horiz):
            for k in range(i + 1, len(horiz)):
                r = horiz[k]
                gap = r[0] - q[2]
                if gap > 3 * glyph:
                    continue
                x = (r[0] + min(q[2], r[2])) / 2 if gap < 0 else None
                dy = abs(y_at(q, x) - y_at(r, x)) if x is not None else abs(q[3] - r[1])
                if dy <= 0.5 * glyph:
                    left = q if q[0] <= r[0] else r
                    right = q if q[2] >= r[2] else r
                    horiz[i] = (left[0], left[1], right[2], right[3])
                    del horiz[k]
                    joined = True
                    break
            if joined:
                break
    horiz = [r for r in horiz if r[2] - r[0] >= 5 * glyph]
    if not horiz:
        return []

    m = 0.7 * glyph
    rules = [("h", r) for r in horiz] + [("v", r) for r in vert]
    uf = _UnionFind(len(rules))
    crossing: set[int] = set()
    for i, h_ in enumerate(horiz):
        for j, v in enumerate(vert, start=len(horiz)):
            hy = y_at(h_, (v[0] + v[2]) / 2)
            vx = x_at(v, hy)
            if h_[0] - m <= vx <= h_[2] + m and v[1] - m <= hy <= v[3] + m:  # touch
                uf.union(i, j)
                if h_[0] + m <= vx <= h_[2] - m and v[1] + m <= hy <= v[3] - m:  # cross
                    crossing.add(i)
    # Pieces of one vertical rule (Hough breaks it where the horizontal rules cross).
    for a, va in enumerate(vert, start=len(horiz)):
        for b, vb in enumerate(vert[a - len(horiz) + 1:], start=a + 1):
            if vb[1] - va[3] <= glyph and va[1] - vb[3] <= glyph:
                y = (max(va[1], vb[1]) + min(va[3], vb[3])) / 2
                if abs(x_at(va, y) - x_at(vb, y)) <= m:
                    uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(rules)):
        groups.setdefault(uf.find(i), []).append(i)

    tables = []
    for members in groups.values():
        if not crossing & set(members) or not any(rules[i][0] == "v" for i in members):
            continue
        xs = [p for i in members for p in (rules[i][1][0], rules[i][1][2])]
        ys = [p for i in members for p in (rules[i][1][1], rules[i][1][3])]
        box = _Comp(int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1, 0)
        # Two row rules on one vertical rule, or the "+" test on the box (a bowed
        # rule can defeat the latter, which re-fits straight lines).
        rows = sum(rules[i][0] == "h" for i in members)
        if box.w < 0.9 * ink.shape[1] and box.h >= 1.5 * glyph and (
                rows >= 2 or _is_table(ink[box.y0:box.y1, box.x0:box.x1], glyph)):
            tables.append(box)
    return tables


def _take_table_ink(blobs: list[_Comp], tables: list[_Comp], glyph: float) -> tuple[list[_Comp], list[_Comp]]:
    """Give each table the writing inside its rules (row labels, values, arrows, what
    is left of the rules themselves) and grow its box to cover it; return the
    tables and the blobs that remain text."""
    if not tables:
        return tables, blobs
    owned: list[list[_Comp]] = [[t] for t in tables]
    text = []
    pad = 0.3 * glyph
    head = 0.8 * glyph  # the header row (x | 0 4 +inf) may sit a little above the vertical rule
    for b in blobs:
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        for k, t in enumerate(tables):
            # Mostly inside: the centre within the rules, or most of the blob's area.
            shared = max(0, _overlap(t.x0, t.x1, b.x0, b.x1)) * max(0, _overlap(t.y0, t.y1, b.y0, b.y1))
            inside = (t.x0 - pad <= cx <= t.x1 + pad and t.y0 - head <= cy <= t.y1 + pad) or shared >= 0.6 * b.w * b.h
            if inside:
                owned[k].append(b)
                break
        else:
            text.append(b)
    tables = [_union_box(o) for o in owned]
    # Rules fade out before the last column: signs and values (small blobs) in the
    # table's rows, at most 2.5 glyphs beyond its side, still belong to it. Text
    # written beside a table starts further away.
    grown = True
    while grown:
        grown = False
        for b in text:
            cy = (b.y0 + b.y1) / 2
            for k, t in enumerate(tables):
                gap = max(b.x0 - t.x1, t.x0 - b.x1)
                shared = max(0, _overlap(t.x0, t.x1, b.x0, b.x1)) * max(0, _overlap(t.y0, t.y1, b.y0, b.y1))
                in_rows = t.y0 <= cy <= t.y1 and b.h <= 2 * glyph
                # An arrow or rule starting inside the table runs on in its row.
                if shared >= 0.6 * b.w * b.h or (in_rows and t.x0 <= b.x0 < t.x1) or (
                        in_rows and b.w <= 3 * glyph and gap <= 2.5 * glyph):
                    tables[k] = _union_box([t, b])
                    text.remove(b)
                    grown = True
                    break
            if grown:
                break
    return tables, text


def _take_labels(blobs: list[_Comp], figures: list[_Figure], glyph: float) -> list[_Comp]:
    """Give each figure its labels (A, B, C, 13: small blobs within ~1 glyph of its
    strokes); return the blobs that remain text."""
    if not figures:
        return blobs
    near = cv2.distanceTransform((~np.any([f.mask for f in figures], axis=0)).astype(np.uint8), cv2.DIST_L2, 3)
    text = []
    for b in blobs:
        small = b.w <= 2.2 * glyph and b.h <= 1.6 * glyph
        if small and near[b.y0:b.y1, b.x0:b.x1].min() <= glyph:
            cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
            fig = min(figures, key=lambda f: abs((f.box.x0 + f.box.x1) / 2 - cx) + abs((f.box.y0 + f.box.y1) / 2 - cy))
            fig.box = _union_box([fig.box, b])
            fig.labels.append(b)
            continue
        text.append(b)
    return text


def _attach_beside(blocks: list[_Comp], units: list[list[_Comp]]) -> list[list[_Comp]]:
    """A figure (or table) and the lines written beside it (sharing most of their
    height with it) form one reading unit: the lines on its left (the statement:
    "ABC dr. in A, BC = 2"), the block, then the lines on its right (the
    calculations, the monotony read off a table), each side top to bottom."""
    out = [u for u in units]
    for f in blocks:
        cx = (f.x0 + f.x1) / 2
        left, right, rest = [], [], []
        for u in out:
            ln = u[0]
            if len(u) == 1 and _overlap(f.y0, f.y1, ln.y0, ln.y1) > 0.5 * ln.h and (ln.x0 >= cx or ln.x1 <= cx):
                (right if ln.x0 >= cx else left).append(ln)
            else:
                rest.append(u)
        out = rest + [sorted(left, key=lambda c: c.y0) + [f] + sorted(right, key=lambda c: c.y0)]
    return out


def _reading_order(units: list[list[_Comp]]) -> list[_Comp]:
    """Top to bottom; units that share a row go left to right; a split block keeps
    its own order (left column, then right column)."""
    boxes = [(_union_box(u), u) for u in units]
    boxes.sort(key=lambda bu: bu[0].y0)
    rows: list[list[tuple[_Comp, list[_Comp]]]] = []
    for box, unit in boxes:
        if rows and _overlap(rows[-1][0][0].y0, rows[-1][0][0].y1, box.y0, box.y1) > 0.5 * min(rows[-1][0][0].h, box.h):
            rows[-1].append((box, unit))
        else:
            rows.append([(box, unit)])
    return [ln for row in rows for _, unit in sorted(row, key=lambda bu: bu[0].x0) for ln in unit]


def _pad(lines: list[_Comp], glyph: float, w: int, h: int) -> list[BBox]:
    """Pad each box by ~0.4 glyph, but never past the middle of the gap to an
    overlapping neighbour above or below. Neighbours whose ink overlaps by up to a
    glyph (an ascender reaching into the line above) are cut at the middle of the
    overlap, so consecutive lines never share a strip of the page."""
    pad = max(4, int(0.4 * glyph))
    out = []
    for a in lines:
        top, bottom = max(0, a.y0 - pad), min(h, a.y1 + pad)
        for b in lines:
            if b is a or _overlap(a.x0, a.x1, b.x0, b.x1) <= 0:
                continue
            above = b.y1 <= a.y0 or (b.y0 < a.y0 and b.y1 < a.y1 and b.y1 - a.y0 <= glyph)
            below = b.y0 >= a.y1 or (b.y0 > a.y0 and b.y1 > a.y1 and a.y1 - b.y0 <= glyph)
            if above:
                top = max(top, (b.y1 + a.y0) // 2)
            elif below:
                bottom = min(bottom, (a.y1 + b.y0) // 2)
        x0, x1 = max(0, a.x0 - pad), min(w, a.x1 + pad)
        out.append(BBox(x0, top, x1 - x0, bottom - top))
    return out


# ----------------------------------------------------------------------- public


def load_page(image_bytes: bytes) -> Image.Image:
    """Decode an upload, apply the phone's EXIF rotation, normalise to RGB."""
    img = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(img).convert("RGB")


def detect_lines(page: Image.Image, params: SegmentationParams | None = None) -> list[DetectedLine]:
    """Line boxes (with their kind) in full-resolution page coordinates, in reading
    order; [] when fewer than two lines (or more than params.max_lines) are found."""
    params = params or SegmentationParams()
    scale = min(1.0, WORK_LONG_SIDE / max(page.size))
    work = page if scale == 1.0 else page.resize(
        (round(page.width * scale), round(page.height * scale)), Image.Resampling.LANCZOS)
    rgb = np.asarray(work)

    ink, glyph, full = _clean_ink(rgb, params)
    if glyph == 0.0:
        return []
    # Figures first: taken out of the ink as whole objects, so the text around them
    # clusters into lines on its own and the drawing becomes one region.
    # Tables likewise: every row inside the rules is one region, the text beside it
    # clusters on its own.
    fig_ink = _figure_ink(full, glyph) if params.classify_regions else None
    figures = _find_figures(fig_ink, glyph) if params.classify_regions else []
    text_ink = ink & ~np.any([f.mask for f in figures], axis=0) if figures else ink
    blobs = _take_labels(_blobs(text_ink, glyph), figures, glyph)
    # A table's rules can span 40% of the page too: only lines across most of it are grid here.
    tables = _find_tables(_figure_ink(full, glyph, TABLE_GRID_SPAN), glyph) if params.classify_regions else []
    tables, blobs = _take_table_ink(blobs, tables, glyph)
    if tables:
        text_ink = text_ink.copy()
        for t in tables:
            text_ink[t.y0:t.y1, t.x0:t.x1] = False
    lines = _cluster_lines(blobs, glyph, params)
    units = _split_columns(lines, blobs, text_ink, glyph, params) if params.split_columns else [[ln] for ln in lines]
    units = _attach_beside([f.box for f in figures] + tables, units)
    lines = _reading_order(units)
    if len(lines) < 2 or len(lines) > params.max_lines:
        return []

    by_box = {id(f.box): f for f in figures}
    table_ids = {id(t) for t in tables}
    kinds = ["diagram" if id(ln) in by_box else "table" if id(ln) in table_ids
             else _classify(ln, full, glyph) if params.classify_regions else "line"
             for ln in lines]
    inv = 1.0 / scale
    boxes = _pad(lines, glyph, rgb.shape[1], rgb.shape[0])
    return [DetectedLine(BBox(round(b.x * inv), round(b.y * inv), round(b.w * inv), round(b.h * inv)), kind,
                         by_box[id(ln)].keep(b, glyph) if id(ln) in by_box else None)
            for ln, b, kind in zip(lines, boxes, kinds)]


def _encode(img: Image.Image, upscale: bool = False) -> bytes:
    if upscale:
        scale = min(CROP_MAX_UPSCALE, (CROP_TARGET_PIXELS / (img.width * img.height)) ** 0.5)
        if scale > 1.05:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _crop(page: Image.Image, d: DetectedLine) -> Image.Image:
    """The region's crop; for a figure, everything that is not the figure is blanked."""
    crop = page.crop((d.bbox.x, d.bbox.y, d.bbox.x1, d.bbox.y1))
    if d.keep is None:
        return crop
    keep = Image.fromarray(d.keep.astype(np.uint8) * 255).resize(crop.size, Image.Resampling.NEAREST)
    return Image.composite(crop, Image.new("RGB", crop.size, "white"), keep)


def segment_page(image_bytes: bytes, params: SegmentationParams | None = None) -> Segmentation:
    """Split a page into line regions, or fall back to one full-page region."""
    page = load_page(image_bytes)
    try:
        detected = detect_lines(page, params)
    except Exception:  # segmentation is best-effort; the full page still works
        logger.exception("Line segmentation failed; falling back to the full page")
        detected = []

    if not detected:
        full = BBox(0, 0, page.width, page.height)
        return Segmentation("full_page", [Region(0, full, _encode(page), "image/jpeg")],
                            page.width, page.height)

    regions = [Region(i, d.bbox, _encode(_crop(page, d), upscale=True), "image/jpeg", d.kind)
               for i, d in enumerate(detected)]
    return Segmentation("lines", regions, page.width, page.height)


def render_preview(image_bytes: bytes, params: SegmentationParams | None = None) -> bytes:
    """The page with each detected region boxed and numbered, as PNG. For tuning.
    Lines alternate red/blue/green; tables are orange ("T"), diagrams purple ("D")."""
    page = load_page(image_bytes)
    seg = segment_page(image_bytes, params)
    draw = ImageDraw.Draw(page)
    stroke = max(2, page.width // 600)
    colors = [(220, 40, 40), (30, 120, 220), (20, 150, 60)]
    kind_colors = {"table": ((235, 140, 0), "T"), "diagram": ((150, 40, 200), "D")}
    for r in seg.regions:
        c, tag = kind_colors.get(r.kind, (colors[r.index % len(colors)], ""))
        b = r.bbox
        draw.rectangle((b.x, b.y, b.x1 - 1, b.y1 - 1), outline=c, width=stroke)
        draw.text((b.x + stroke + 2, b.y + stroke), f"{r.index}{tag}", fill=c)
    buf = io.BytesIO()
    page.save(buf, format="PNG")
    return buf.getvalue()
