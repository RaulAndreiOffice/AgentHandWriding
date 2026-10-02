"""Shared fixtures: synthetic pages and a scripted fake VLM (no vLLM needed)."""

import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.config import Settings, get_settings
from app.main import app
from app.services.pipeline_service import TranscriptionPipeline, get_pipeline
from app.services.vlm_service import VLMServiceError

FONT_SIZE = 40
LINE_PITCH = 110  # px between baselines; leaves a clear gap between lines


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_page(lines: list[str], *, grid: bool = False, fraction_after: int | None = None,
              size: tuple[int, int] = (1200, 1600)) -> Image.Image:
    """White page with one text line per entry, optionally on grid paper.

    fraction_after=i inserts "a+b / 2" (numerator, bar, denominator) as an
    extra line after line i.
    """
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    if grid:  # light grey squares, like a math notebook
        for x in range(0, size[0], 40):
            draw.line((x, 0, x, size[1]), fill=(200, 205, 215), width=1)
        for y in range(0, size[1], 40):
            draw.line((0, y, size[0], y), fill=(200, 205, 215), width=1)

    font = ImageFont.load_default(size=FONT_SIZE)
    y = 100
    for i, text in enumerate(lines):
        draw.text((100, y), text, font=font, fill=(20, 30, 120))
        y += LINE_PITCH
        if fraction_after == i:
            draw.text((180, y), "x = ", font=font, fill=(20, 30, 120))
            draw.text((290, y - 34), "a+b", font=font, fill=(20, 30, 120))
            draw.line((280, y + 22, 380, y + 22), fill=(20, 30, 120), width=4)
            draw.text((315, y + 30), "2", font=font, fill=(20, 30, 120))
            y += LINE_PITCH + 40
    return img


class FakeVLM:
    """Records calls; line calls fail when their call index is in `fail`."""

    model = "fake-vlm"

    def __init__(self):
        self.page_calls = 0
        self.line_calls = 0
        self.fail: set[int] = set()
        self.fail_page = False
        self.overrides: dict[int, str] = {}  # call index -> text to return instead

    async def transcribe_page(self, image_bytes: bytes, mime_type: str) -> str:
        self.page_calls += 1
        if self.fail_page:
            raise VLMServiceError("page boom")
        return "PAGE"

    async def transcribe_line(self, image_bytes: bytes, mime_type: str) -> str:
        n = self.line_calls
        self.line_calls += 1
        if n in self.fail:
            raise VLMServiceError(f"boom {n}")
        if n in self.overrides:
            return self.overrides[n]
        return f"```latex\n$line{n}$\n```"  # fences must be stripped by the pipeline

    # ---- re-ask pass ---------------------------------------------------------
    #: question substring -> answer text; unmatched questions get an answer outside the options
    classify_answers: dict[str, str]
    #: text returned by retranscribe_line ("" = nothing better)
    retranscription: str = ""

    def _reask_started(self):
        self.reask_calls = getattr(self, "reask_calls", 0) + 1
        self.in_flight = getattr(self, "in_flight", 0) + 1
        self.max_in_flight = max(getattr(self, "max_in_flight", 0), self.in_flight)

    #: what describe_figure / transcribe_table return
    figure_description: str = "Triunghi ABC dreptunghic in A"
    table_latex: str = r"$$\begin{array}{c|cc} x & 0 & 1 \\ \hline f(x) & \nearrow & 3 \end{array}$$"

    async def describe_figure(self, image_bytes: bytes, mime_type: str) -> str:
        self.figure_calls = getattr(self, "figure_calls", 0) + 1
        return self.figure_description

    async def transcribe_table(self, image_bytes: bytes, mime_type: str) -> str:
        self.table_calls = getattr(self, "table_calls", 0) + 1
        return self.table_latex

    async def classify(self, image_bytes: bytes, mime_type: str, question: str, options: list[str]):
        self._reask_started()
        self.questions = getattr(self, "questions", []) + [question]
        await asyncio.sleep(0.01)  # let concurrent calls overlap
        self.in_flight -= 1
        for needle, answer in getattr(self, "classify_answers", {}).items():
            if needle in question:
                return answer
        return "I am not sure"

    async def retranscribe_line(self, image_bytes: bytes, mime_type: str, problem: str, previous=None) -> str:
        self._reask_started()
        self.problems = getattr(self, "problems", []) + [problem]
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return self.retranscription


def make_settings(**overrides) -> Settings:
    # _env_file=None: tests never depend on the developer's .env
    base = {"vlm_concurrency": 1, "reask_enabled": True}  # deterministic call order for FakeVLM.fail
    return Settings(_env_file=None, **{**base, **overrides})


@pytest.fixture
def fake_vlm() -> FakeVLM:
    return FakeVLM()


@pytest.fixture
def client(fake_vlm):
    settings = make_settings()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pipeline] = lambda: TranscriptionPipeline(fake_vlm, settings)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def lined_page() -> bytes:
    return png_bytes(make_page(["1) 2x + 3 = 7", "2x = 4", "x = 2", "2) f(x) = 3x - 1", "f(2) = 5"]))
