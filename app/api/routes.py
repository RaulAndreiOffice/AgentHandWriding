"""HTTP layer only: validate the request, delegate to the service, shape the response."""

import asyncio
import time

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from PIL import UnidentifiedImageError

from app.config import Settings, get_settings
from app.schemas import (ErrorResponse, IssueResult, LineResult, ReaskResult, SegmentationInfo,
                         TranscriptionResponse, VerificationInfo)
from app.services.pipeline_service import TranscriptionPipeline, get_pipeline
from app.services.segmenter import render_preview
from app.services.vlm_service import VLMServiceError

router = APIRouter(prefix="/api", tags=["transcription"])

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


async def _read_image(file: UploadFile, settings: Settings) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if len(image_bytes) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            413,  # Content Too Large (constant renamed across Starlette versions)
            f"File exceeds the {settings.max_upload_mb} MB limit.",
        )
    return image_bytes


@router.post(
    "/transcribe-page",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def transcribe_page(
    file: UploadFile = File(..., description="Photo/scan of a page of handwritten math exercises"),
    segment: bool = Query(True, description="Split the page into lines first. false = send the whole page in one call."),
    pipeline: TranscriptionPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> TranscriptionResponse:
    image_bytes = await _read_image(file, settings)

    started = time.perf_counter()
    try:
        result = await pipeline.transcribe(image_bytes, file.content_type, segment=segment)
    except UnidentifiedImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File could not be decoded as an image.") from exc
    except VLMServiceError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return TranscriptionResponse(
        status="partial" if result.failed_lines else "success",
        latex=result.latex,
        model=pipeline.model,
        filename=file.filename or "",
        processing_time_ms=elapsed_ms,
        segmentation=SegmentationInfo(
            mode=result.mode,
            regions_detected=len(result.lines) if result.mode == "lines" else 1,
            failed_regions=result.failed_lines,
            page_size=list(result.page_size),
        ),
        verification=VerificationInfo(enabled=result.verified, corrections=result.corrections,
                                      reask_calls=result.reask_calls, reask_accepted=result.reask_accepted,
                                      **result.status_counts()),
        lines=[
            LineResult(index=l.index, bbox=l.bbox, latex=l.latex, raw_latex=l.raw_latex, status=l.status,
                       issues=[IssueResult(kind=i.kind, message=i.message, fixed=i.fixed, key=i.key,
                                           resolved_by=i.resolved_by) for i in l.issues],
                       reasks=[ReaskResult(kind=e.kind, key=e.key, question=e.question, answer=e.answer,
                                           accepted=e.accepted, detail=e.detail)
                                for e in l.reasks],
                       kind=l.kind, image=l.image_data_url, error=l.error)
            for l in result.lines
        ],
    )


@router.post(
    "/segment-preview",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Page with detected line boxes drawn on it"}},
)
async def segment_preview(
    file: UploadFile = File(..., description="Same image you would send to /api/transcribe-page"),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Show how the page would be split, without calling the VLM."""
    image_bytes = await _read_image(file, settings)
    try:
        png = await asyncio.to_thread(render_preview, image_bytes, settings.segmentation_params())
    except UnidentifiedImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File could not be decoded as an image.") from exc
    return Response(content=png, media_type="image/png")
