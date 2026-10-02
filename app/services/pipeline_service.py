"""Page transcription workflow: segment -> VLM per region -> verify -> re-ask -> merge.

The verifier (app/services/verifier.py) corrects systematic misreadings and tags
each line green / yellow / grey; the re-ask pass (app/services/reask.py) then goes
back to the VLM, with the line crop, for the yellow and grey lines only.
The raw VLM text is kept alongside.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from app.config import Settings, get_settings
from app.services.segmenter import Region, RegionKind, segment_page
from app.services.reask import Reasker, ReaskEvent
from app.services.verifier import Issue, Status, verify_page
from app.services.vlm_service import VLMService, VLMServiceError, get_vlm_service, strip_code_fences

logger = logging.getLogger(__name__)


@dataclass
class LineTranscription:
    index: int
    bbox: list[int]
    latex: str = ""
    error: str | None = None
    #: What the VLM returned, before verification (latex is the verified text).
    raw_latex: str = ""
    #: The VLM text the verifier works on: raw_latex, or a kept re-transcription.
    source_latex: str = ""
    status: Status | None = None
    issues: list[Issue] = field(default_factory=list)
    reasks: list[ReaskEvent] = field(default_factory=list)
    #: "line", "table" (sign table, transcribed as an array) or "diagram" (a figure:
    #: described, not transcribed; the crop is attached as image_data_url)
    kind: RegionKind = "line"
    image_data_url: str | None = None


@dataclass
class PageTranscription:
    latex: str
    mode: Literal["lines", "full_page"]
    page_size: tuple[int, int]
    lines: list[LineTranscription] = field(default_factory=list)
    segmentation_ms: int = 0
    vlm_ms: int = 0

    @property
    def failed_lines(self) -> int:
        return sum(1 for line in self.lines if line.error)

    @property
    def verified(self) -> bool:
        return any(line.status for line in self.lines)

    def status_counts(self) -> dict[str, int]:
        counts = {"green": 0, "yellow": 0, "grey": 0}
        for line in self.lines:
            if line.status:
                counts[line.status] += 1
        return counts

    @property
    def corrections(self) -> int:
        return sum(1 for line in self.lines for i in line.issues if i.fixed and i.content)

    @property
    def reask_calls(self) -> int:
        return sum(len(line.reasks) for line in self.lines)

    @property
    def reask_accepted(self) -> int:
        return sum(e.accepted for line in self.lines for e in line.reasks)


class TranscriptionPipeline:
    def __init__(self, vlm: VLMService, settings: Settings):
        self.vlm = vlm
        self.settings = settings

    @property
    def model(self) -> str:
        return self.vlm.model

    async def transcribe(self, image_bytes: bytes, mime_type: str, *, segment: bool = True) -> PageTranscription:
        if not segment:
            started = time.perf_counter()
            latex = await self.vlm.transcribe_page(image_bytes, mime_type)
            lines = await self._verify([LineTranscription(0, [0, 0, 0, 0], latex)])
            return PageTranscription(latex=_merge(lines), mode="full_page", page_size=(0, 0),
                                     lines=lines, vlm_ms=_ms_since(started))

        started = time.perf_counter()
        seg = await asyncio.to_thread(segment_page, image_bytes, self.settings.segmentation_params())
        segmentation_ms = _ms_since(started)
        logger.info("Segmented page %dx%d into %d region(s) [%s]",
                    seg.page_width, seg.page_height, len(seg.regions), seg.mode)

        started = time.perf_counter()
        if seg.mode == "full_page":
            region = seg.regions[0]
            latex = await self.vlm.transcribe_page(region.image_bytes, region.mime_type)
            lines = [LineTranscription(0, region.bbox.as_list(), latex)]
        else:
            lines = await self._transcribe_regions(seg.regions)
            if all(line.error for line in lines):
                raise VLMServiceError(f"All {len(lines)} line transcriptions failed; first error: {lines[0].error}")
        lines = await self._verify(lines)
        if seg.mode == "lines":
            # Re-ask ordinary lines only: its prompts are line prompts.
            await self._reask(lines, {r.index: (r.image_bytes, r.mime_type) for r in seg.regions if r.kind == "line"})
        latex = _merge(lines)

        return PageTranscription(latex=latex, mode=seg.mode, page_size=(seg.page_width, seg.page_height),
                                 lines=lines, segmentation_ms=segmentation_ms, vlm_ms=_ms_since(started))

    async def _transcribe_regions(self, regions: list[Region]) -> list[LineTranscription]:
        """One VLM call per region, at most `vlm_concurrency` in flight; order preserved.

        Tables get the sign-table prompt; diagrams are described, not transcribed
        (a 2B model reading a triangle as math invents matrices), and keep their crop.
        """
        sem = asyncio.Semaphore(max(1, self.settings.vlm_concurrency))

        async def one(region: Region) -> LineTranscription:
            result = LineTranscription(region.index, region.bbox.as_list(), kind=region.kind)
            async with sem:
                try:
                    if region.kind == "table" and self.settings.vlm_table_prompt:
                        result.latex = await self.vlm.transcribe_table(region.image_bytes, region.mime_type)
                        if is_degenerate(result.latex):  # loops like "c|c|c|..." or empty cells
                            result.latex = await self.vlm.transcribe_line(region.image_bytes, region.mime_type)
                    elif region.kind == "diagram":
                        description = await self.vlm.describe_figure(region.image_bytes, region.mime_type)
                        result.latex = figure_placeholder(description)
                    else:
                        result.latex = await self.vlm.transcribe_line(region.image_bytes, region.mime_type)
                except VLMServiceError as exc:
                    logger.warning("Line %d failed: %s", region.index, exc)
                    result.error = str(exc)
            if region.kind == "diagram":
                result.image_data_url = (f"data:{region.mime_type};base64,"
                                         f"{base64.b64encode(region.image_bytes).decode('ascii')}")
                if result.error:  # the figure is still there: keep the placeholder and the image
                    result.latex, result.error = figure_placeholder(""), None
            return result

        return list(await asyncio.gather(*(one(r) for r in regions)))

    async def _verify(self, lines: list[LineTranscription]) -> list[LineTranscription]:
        """Correct and tag every line with page context (Phase 2 verifier)."""
        for line in lines:
            line.raw_latex = line.source_latex = line.latex
        if not self.settings.verify_transcriptions:
            return lines
        results = await asyncio.to_thread(verify_page, [l.latex for l in lines], [l.error for l in lines])
        for line, res in zip(lines, results):
            if line.kind == "diagram":  # a placeholder by design: nothing to correct
                line.status = "green"
                line.issues = [Issue("diagram", "geometric figure: described, not transcribed; crop attached",
                                     content=False)]
                continue
            line.latex, line.status, line.issues = res.latex, res.status, res.issues
        return lines

    async def _reask(self, lines: list[LineTranscription], crops: dict[int, tuple[bytes, str]]) -> None:
        """Targeted second pass on yellow/grey lines; green lines cost nothing."""
        if not (self.settings.verify_transcriptions and self.settings.reask_enabled):
            return
        if all(line.status == "green" for line in lines):
            return

        async def verify(texts, errors, decisions):
            return await asyncio.to_thread(verify_page, texts, errors, None, decisions)

        await Reasker(self.vlm, self.settings).run(lines, crops, verify)


def _merge(lines: list[LineTranscription]) -> str:
    """Join line results top to bottom into one document.

    Each region becomes its own paragraph (blank line between), which renders as
    a line break in both Markdown+KaTeX and LaTeX. Lines inside a multi-line
    region keep their own single newlines.
    """
    parts = []
    for line in lines:
        text = strip_code_fences(line.latex.strip())
        text = "\n".join(row.rstrip() for row in text.splitlines() if row.strip())
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def is_degenerate(text: str) -> bool:
    """A transcription stuck in a loop: a short chunk repeated many times
    ("c|c|c|c|...", "\\text{ } & \\text{ } & ..."), or mostly empty cells."""
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    step = max(1, len(compact) // 20)  # candidate chunks from across the text, not just its start
    for size in range(1, 13):
        for start in range(0, len(compact) - size + 1, step):
            chunk = compact[start:start + size]
            if len(chunk) == size and compact.count(chunk) * size > 0.5 * len(compact) and compact.count(chunk) >= 12:
                return True
    return False


_FIGURE_JUNK = re.compile(r"[\\${}&^_]|begin|matrix")


def figure_placeholder(description: str) -> str:
    """$\\text{[Figura geometrica: <description>]}$, or a bare placeholder when the
    description is empty, too long or looks like math the model made up."""
    text = " ".join(description.replace("\n", " ").split()).strip(" .\"'")
    if not text or len(text) > 100 or _FIGURE_JUNK.search(text):
        return r"$\text{[Figura geometrica]}$"
    return r"$\text{[Figura geometrica: " + text + "]}$"


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


@lru_cache
def get_pipeline() -> TranscriptionPipeline:
    """FastAPI dependency. Override in tests via app.dependency_overrides."""
    return TranscriptionPipeline(get_vlm_service(), get_settings())
