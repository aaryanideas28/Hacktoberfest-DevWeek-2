from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from app.services.compressor import compress_prompt
from app.services.gemma import neural_compress_with_gemma
from app.core.sentry import record_compression_span

router = APIRouter(prefix="/api/v1", tags=["Compression"])

class CompressRequest(BaseModel):
    prompt: str
    mode: Optional[str] = "hybrid"  # "gemma_entropy", "rule", "hybrid", "gemma_generative"
    drop_rate: Optional[float] = 0.40  # Fraction of lowest-information tokens to drop (default 40%)

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

@router.post("/compress", response_model=CompressResponse)
def compress_endpoint(
    request: CompressRequest,
    mode: Optional[str] = Query(None, description="Query parameter override for mode"),
):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    selected_mode = (mode or request.mode or "hybrid").lower()
    drop_rate = request.drop_rate if request.drop_rate is not None else 0.40

    # Mode: Pure generative Gemma 2 rewrite (API / Studio)
    if selected_mode in ["gemma_generative", "gemma"]:
        gemma_out = neural_compress_with_gemma(request.prompt)
        if gemma_out:
            orig_t = max(1, len(request.prompt) // 4)
            comp_t = max(1, len(gemma_out) // 4)
            saved = max(0, orig_t - comp_t)
            record_compression_span(orig_t, comp_t, "gemma_generative", saved)
            return CompressResponse(
                original_prompt=request.prompt,
                compressed_prompt=gemma_out,
                mode_used="gemma_generative",
                engine_used="gemma-2-9b-it (Google AI Studio API)",
                original_tokens=orig_t,
                compressed_tokens=comp_t,
                tokens_saved=saved,
                compression_ratio=round((saved / orig_t) * 100, 2),
                latency_ms=0.0
            )

    # Modes: "gemma_entropy", "hybrid", "rule" (Token-level perplexity / forward pass pruning)
    result = compress_prompt(request.prompt, mode=selected_mode, drop_rate=drop_rate)

    record_compression_span(
        original_tokens=result["original_tokens"],
        compressed_tokens=result["compressed_tokens"],
        mode=selected_mode,
        tokens_saved=result["tokens_saved"]
    )

    return CompressResponse(
        original_prompt=request.prompt,
        compressed_prompt=result["compressed_prompt"],
        mode_used=selected_mode,
        engine_used=result.get("engine", "gemma-2-2b-entropy-pruning"),
        original_tokens=result["original_tokens"],
        compressed_tokens=result["compressed_tokens"],
        tokens_saved=result["tokens_saved"],
        compression_ratio=result["compression_ratio"],
        latency_ms=result.get("latency_ms", 0.0)
    )