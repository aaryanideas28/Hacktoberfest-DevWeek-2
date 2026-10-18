from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from app.services.compressor import compress_prompt
from app.core.sentry import record_compression_span

router = APIRouter(prefix="/api/v1", tags=["Compression"])

MAX_PROMPT_CHARS = 200_000


class CompressRequest(BaseModel):
    prompt: str
    mode: Optional[str] = "hybrid"  # "hybrid", "rule", "gemma_entropy", "generative"
    drop_rate: Optional[float] = 0.40


class CompressResponse(BaseModel):
    original_prompt: str
    compressed_prompt: str
    mode_used: str
    engine_used: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: float
    latency_ms: Optional[float] = 0.0
    note: Optional[str] = None
    removed_words: List[str] = []
    protected_pct: Optional[int] = None


@router.post("/compress", response_model=CompressResponse)
def compress_endpoint(
    request: CompressRequest,
    mode: Optional[str] = Query(None, description="Query parameter override for mode"),
):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if len(request.prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail="Prompt too large (max 200k characters).")

    selected_mode = (mode or request.mode or "hybrid").lower()
    drop_rate = request.drop_rate if request.drop_rate is not None else 0.40

    # Every mode goes through the validated pipeline (safety gate + fallbacks).
    result = compress_prompt(request.prompt, mode=selected_mode, drop_rate=drop_rate)

    record_compression_span(
        original_tokens=result["original_tokens"],
        compressed_tokens=result["compressed_tokens"],
        mode=selected_mode,
        tokens_saved=result["tokens_saved"],
    )

    return CompressResponse(
        original_prompt=request.prompt,
        compressed_prompt=result["compressed_prompt"],
        mode_used=selected_mode,
        engine_used=result.get("engine", "unknown"),
        original_tokens=result["original_tokens"],
        compressed_tokens=result["compressed_tokens"],
        tokens_saved=result["tokens_saved"],
        compression_ratio=result["compression_ratio"],
        latency_ms=result.get("latency_ms", 0.0),
        note=result.get("note"),
        removed_words=result.get("removed_words", []),
        protected_pct=result.get("protected_pct"),
    )