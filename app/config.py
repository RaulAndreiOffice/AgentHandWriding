from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from app.services.segmenter import SegmentationParams


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vlm_base_url: str = "http://localhost:11434/v1"
    vlm_api_key: str = "not-needed"
    vlm_model: str = "qwen2.5vl:7b"
    vlm_max_tokens: int = 4096
    vlm_temperature: float = 0.0
    vlm_timeout: float = 300.0
    # vLLM --max-model-len: prompt + image + output must fit (serve.sh: MAX_LEN, default 1024).
    vlm_max_model_len: int = 1024
    # Regions tagged "table" use a dedicated sign-table prompt. Off by default: on the gold set
    # the 2B model copies the prompt's example or loops (c|c|c|...), so tables use the line prompt.
    vlm_table_prompt: bool = False

    # Parallel VLM calls per page. serve.sh runs vLLM with --max-num-seqs 1, so extra
    # requests just queue on the server; 2 keeps the next crop uploaded and waiting.
    vlm_concurrency: int = 2

    # Line segmentation (app/services/segmenter.py, SegmentationParams). Pixel values
    # are at the 2000 px working scale; leave unset for values derived from glyph height.
    seg_box_merge_gap_px: int | None = None
    seg_row_merge_threshold_px: int | None = None
    seg_grid_filter_strength: float = 0.5  # 0 = keep faint strokes .. 1 = aggressive
    seg_margin_exclude_frac: float = 0.08
    # Split side-by-side blocks (a figure beside its calculations) into columns; gutter
    # width in px at the working scale, unset = 2.5 x glyph height.
    seg_split_columns: bool = True
    seg_column_gap_px: int | None = None
    # Tag regions as "table" (sign tables -> array prompt) or "diagram" (figures -> placeholder).
    seg_classify_regions: bool = True
    # Pages with more detected lines than this are sent whole (likely mis-segmented).
    seg_max_lines: int = 80

    def segmentation_params(self) -> "SegmentationParams":
        from app.services.segmenter import SegmentationParams

        return SegmentationParams(
            box_merge_gap_px=self.seg_box_merge_gap_px,
            row_merge_threshold_px=self.seg_row_merge_threshold_px,
            grid_filter_strength=self.seg_grid_filter_strength,
            margin_exclude_frac=self.seg_margin_exclude_frac,
            split_columns=self.seg_split_columns,
            column_gap_px=self.seg_column_gap_px,
            classify_regions=self.seg_classify_regions,
            max_lines=self.seg_max_lines,
        )

    # Phase 2: correct systematic misreadings and tag lines green/yellow/grey.
    verify_transcriptions: bool = True
    # Targeted re-ask of yellow/grey lines (app/services/reask.py). Green lines cost no extra calls.
    # Off by default: extra VLM calls for a small measured gain (see README).
    reask_enabled: bool = False
    reask_max_questions_per_line: int = 3

    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
