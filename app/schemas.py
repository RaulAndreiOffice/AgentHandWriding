from typing import Literal

from pydantic import BaseModel, Field


class IssueResult(BaseModel):
    kind: str = Field(description="e.g. digit_arg, digit_one, arrow, grade_mark, delimiters, syntax, matrix_shape")
    message: str
    fixed: bool = Field(description="true: the verifier corrected it; false: flagged for review only")
    key: str | None = Field(default=None, description="Ambiguity key, when the VLM can be re-asked about it")
    resolved_by: str | None = Field(default=None, description="'reask' when a targeted re-ask settled it")


class ReaskResult(BaseModel):
    kind: Literal["question", "retranscribe"]
    key: str | None = Field(description="Ambiguity asked about, e.g. s_vs_5:A(s), t_vs_1, arrow:(=)")
    question: str = Field(description="The question, or the problem described to the re-transcription")
    answer: str | None = Field(description="The decision taken from the answer, or the re-transcribed text")
    accepted: bool
    detail: str = ""


class LineResult(BaseModel):
    index: int
    bbox: list[int] = Field(description="[x, y, width, height] in page pixels (after EXIF rotation)")
    latex: str = Field(description="Verified text (equal to raw_latex when verification is off)")
    raw_latex: str = Field(description="What the VLM returned for this region")
    status: Literal["green", "yellow", "grey"] | None = Field(
        description="green: parses, nothing corrected or flagged; yellow: corrected or flagged, worth a glance; "
                    "grey: does not parse, empty/illegible, or the VLM failed. null when verification is off")
    issues: list[IssueResult]
    reasks: list[ReaskResult] = Field(description="Targeted re-asks for this line (empty for green lines)")
    kind: Literal["line", "table", "diagram"] = Field(
        default="line", description="table: sign table (KaTeX array); diagram: a figure, described not transcribed")
    image: str | None = Field(default=None, description="data: URL of the cropped figure (diagrams only)")
    error: str | None = None


class SegmentationInfo(BaseModel):
    mode: Literal["lines", "full_page"] = Field(
        description="'lines' when the page was split, 'full_page' when it was sent whole (fallback or segment=false)")
    regions_detected: int
    failed_regions: int
    page_size: list[int] = Field(description="[width, height]; [0, 0] when segmentation was skipped")


class VerificationInfo(BaseModel):
    enabled: bool
    green: int
    yellow: int
    grey: int
    corrections: int = Field(description="Content corrections applied (formatting fixes not counted)")
    reask_calls: int = Field(description="Extra VLM calls made by the re-ask pass (0 when every line was green)")
    reask_accepted: int = Field(description="Re-asks whose answer was used")


class TranscriptionResponse(BaseModel):
    status: Literal["success", "partial"] = Field(
        description="'partial' when some lines failed; their text is missing from `latex`")
    latex: str
    model: str
    filename: str
    processing_time_ms: int
    segmentation: SegmentationInfo
    verification: VerificationInfo
    lines: list[LineResult]


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    detail: str
