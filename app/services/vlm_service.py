"""Business logic for transcribing a handwritten math page with a Vision-Language Model.

Talks to any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, ...).
"""

import base64
import re
from functools import lru_cache

from openai import APIError, AsyncOpenAI

from app.config import Settings, get_settings

# Shared by both prompts. Kept short: vLLM runs with --max-model-len 1024, which must hold
# this prompt + the image (<= 256 tokens) + VLM_MAX_TOKENS of output (tests/test_prompts.py).
CONVENTIONS = (
    "Romanian bac math: Subiectul 1/2/3, labels 1) a) b); I_2, A(x), det(A), \\mathbb{R}, \\ln.\n"
    "- Numbers stacked in rows inside ( ) form a matrix: \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}; "
    "inside | | use vmatrix. Keep every row; never flatten a matrix into one row.\n"
    "- In arguments, matrices and numbers write digits: A(5) not A(s), 4x not ux, "
    "a stroke like l or ℓ is 1 (except ln, lim, log).\n"
)

SYSTEM_PROMPT = (
    "Transcribe this page of handwritten math exercises into KaTeX/LaTeX, in order, "
    "preserving its structure.\n"
    + CONVENTIONS +
    "- $...$ inline, $$...$$ display. Keep numbering and line breaks.\n"
    "- Illegible symbol: \\text{[illegible]}.\n"
    "- Output only the transcription, no commentary, no code fences."
)

USER_PROMPT = "Transcribe this page."

LINE_SYSTEM_PROMPT = (
    "Transcribe this crop (one line, or a few tightly packed lines, of handwritten math) "
    "into KaTeX/LaTeX exactly as written; do not solve or correct.\n"
    + CONVENTIONS +
    "- Math in $...$, words as plain text; one output line per written line.\n"
    "- Ignore check marks, scores, stickers. Illegible symbol: \\text{[illegible]}.\n"
    "- Output only the transcription, no commentary, no code fences; nothing if the crop is empty."
)

LINE_USER_PROMPT = "Transcribe this line."

# ---- region kinds (segmenter: "table", "diagram") ------------------------------

TABLE_SYSTEM_PROMPT = (
    "The image is a sign/variation table (tabel de semn / tabel de variatie) from Romanian bac math. "
    "Transcribe it as one KaTeX array with exactly the rows and cells written in the image "
    "(often only some of x, f'(x), f''(x), f(x)): "
    "$$\\begin{array}{c|...} <row label> & <cell> & ... \\\\ \\hline ... \\end{array}$$\n"
    "- Copy the values from the image only; never invent rows or values.\n"
    "- Arrows: \\nearrow (up) and \\searrow (down); infinity: \\infty.\n"
    "- Output only the $$...$$ block, no commentary, no code fences."
)
TABLE_USER_PROMPT = "Transcribe this table."

FIGURE_SYSTEM_PROMPT = (
    "The image is a geometric figure drawn by a student (for example a triangle with labelled "
    "vertices and side lengths). Describe it in Romanian in at most 10 words, for example: "
    "Triunghi ABC dreptunghic in A, AC=5, BC=13. Output only the description, no LaTeX."
)
FIGURE_USER_PROMPT = "Describe this figure."

# ---- targeted re-ask (app/services/reask.py) ----------------------------------

CLASSIFY_SYSTEM_PROMPT = (
    "You inspect one character or symbol in a crop of handwritten math. "
    "Look closely at the image, not at what would make the math correct. Answer with one option only."
)

RETRANSCRIBE_SYSTEM_RULES = (
    "\nStrict output:\n"
    "- Valid KaTeX: every { closed, every \\begin{...} matched by \\end{...}.\n"
    "- A matrix: \\begin{pmatrix} with exactly the written rows, rows separated by \\\\, "
    "entries by &; a determinant: \\begin{vmatrix}.\n"
)
RETRANSCRIBE_USER_PROMPT = "Transcribe this line again, carefully. The first attempt had a problem: {problem}."
RETRANSCRIBE_PREVIOUS = "\nFirst attempt: {previous}"
RETRANSCRIBE_MAX_PREVIOUS_CHARS = 400
#: vLLM is started with max_pixels 262144 = 256 image tokens at most (32x32 px per token).
IMAGE_TOKENS_MAX = 256

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


class VLMServiceError(Exception):
    """Raised when the VLM call fails or returns unusable output."""


class VLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.vlm_base_url,
            api_key=settings.vlm_api_key,
            timeout=settings.vlm_timeout,
        )

    @property
    def model(self) -> str:
        return self.settings.vlm_model

    async def transcribe_page(self, image_bytes: bytes, mime_type: str) -> str:
        """Transcribe a whole page in one call. Raises on empty output."""
        content = await self._complete(image_bytes, mime_type, SYSTEM_PROMPT, USER_PROMPT)
        if not content:
            raise VLMServiceError("VLM returned an empty transcription.")
        return content

    async def transcribe_line(self, image_bytes: bytes, mime_type: str) -> str:
        """Transcribe one line crop. Empty output is valid (the crop held no writing)."""
        return await self._complete(image_bytes, mime_type, LINE_SYSTEM_PROMPT, LINE_USER_PROMPT)

    async def transcribe_table(self, image_bytes: bytes, mime_type: str) -> str:
        """Transcribe a sign/variation table crop as one KaTeX array."""
        return await self._complete(image_bytes, mime_type, TABLE_SYSTEM_PROMPT, TABLE_USER_PROMPT)

    async def describe_figure(self, image_bytes: bytes, mime_type: str) -> str:
        """A short Romanian description of a drawn figure (the figure is not transcribed)."""
        response = await self._create(
            [{"role": "system", "content": FIGURE_SYSTEM_PROMPT},
             self._user_message(image_bytes, mime_type, FIGURE_USER_PROMPT, image_first=True)],
            max_tokens=40)
        return (response.choices[0].message.content or "").strip()

    async def classify(self, image_bytes: bytes, mime_type: str, question: str, options: list[str]) -> str:
        """Ask a closed question about the image; return the model's text answer.

        No logprobs: under WSL vLLM computes them in a torch.compile'd sampler
        helper that needs nvcc, and the first such request kills the engine.
        Unconstrained (no structured-outputs choice): the answer counts only if
        the model gives one of `options` by itself; the caller checks that.
        Image before text and the system prompt: best of the formats measured.
        """
        response = await self._create(
            [{"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
             self._user_message(image_bytes, mime_type, question, image_first=True)],
            max_tokens=4)
        return (response.choices[0].message.content or "").strip().rstrip(".").strip()

    async def retranscribe_line(self, image_bytes: bytes, mime_type: str, problem: str,
                                previous: str | None = None) -> str:
        """Transcribe a line again with a strict output schema, telling the model what went wrong."""
        prompt = RETRANSCRIBE_USER_PROMPT.format(problem=problem)
        if previous and len(previous) <= RETRANSCRIBE_MAX_PREVIOUS_CHARS:
            prompt += RETRANSCRIBE_PREVIOUS.format(previous=previous)
        # The context window (prompt + image + output) is small: shrink the output budget to fit.
        prompt_tokens = (len(LINE_SYSTEM_PROMPT) + len(RETRANSCRIBE_SYSTEM_RULES) + len(prompt)) // 3
        budget = self.settings.vlm_max_model_len - prompt_tokens - IMAGE_TOKENS_MAX - 32
        response = await self._create(
            [{"role": "system", "content": LINE_SYSTEM_PROMPT + RETRANSCRIBE_SYSTEM_RULES},
             self._user_message(image_bytes, mime_type, prompt)],
            max_tokens=max(64, min(self.settings.vlm_max_tokens, budget)))
        return strip_code_fences((response.choices[0].message.content or "").strip())

    async def _complete(self, image_bytes: bytes, mime_type: str, system_prompt: str, user_prompt: str) -> str:
        response = await self._create(
            [{"role": "system", "content": system_prompt}, self._user_message(image_bytes, mime_type, user_prompt)],
            max_tokens=self.settings.vlm_max_tokens)
        content = (response.choices[0].message.content or "").strip()
        return strip_code_fences(content)

    @staticmethod
    def _user_message(image_bytes: bytes, mime_type: str, text: str, image_first: bool = False) -> dict:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        parts = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": data_url}}]
        return {"role": "user", "content": parts[::-1] if image_first else parts}

    async def _create(self, messages: list[dict], **kwargs):
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.vlm_model, temperature=self.settings.vlm_temperature,
                messages=messages, **kwargs)
        except APIError as exc:
            raise VLMServiceError(f"VLM request failed: {exc}") from exc
        if not response.choices:
            raise VLMServiceError("VLM returned no choices.")
        return response


def strip_code_fences(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


@lru_cache
def get_vlm_service() -> VLMService:
    """FastAPI dependency. Override in tests via app.dependency_overrides."""
    return VLMService(get_settings())
