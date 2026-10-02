from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.routes import router

app = FastAPI(
    title="Handwritten Math Transcriber",
    description="Phase 1 MVP: upload a page of handwritten math, get LaTeX back from a VLM.",
    version="0.1.0",
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "detail": exc.detail})


app.include_router(router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
