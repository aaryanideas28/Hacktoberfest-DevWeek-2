from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.services.compressor import compress_prompt
from app.services.gemma import compress_with_gemma
from app.core.sentry import record_compression_span

router = APIRouter(prefix="/api/v1", tags=["Compression"])

class CompressRequest(BaseModel):
    prompt: str
    mode: Optional[str] = "hybrid"  # Options: "rule", "gemma", "hybrid"

@router.post("/compress")
def compress_endpoint(request: CompressRequest):
    # Pass 1: Rule-based regex cleaning
    rule_result = compress_prompt(request.prompt)
    
    final_compressed = rule_result["compressed_prompt"]
    mode_used = request.mode

    # Pass 2: Neural Gemma 2 compression (if requested and API key present)
    if request.mode in ["gemma", "hybrid"]:
        gemma_compressed = compress_with_gemma(rule_result["compressed_prompt"])
        if gemma_compressed:
            final_compressed = gemma_compressed
            mode_used = request.mode
        else:
            mode_used = "rule (fallback)"

    # Calculate token savings metrics
    original_tokens = rule_result["original_tokens"]
    compressed_tokens = max(1, len(final_compressed) // 4)
    tokens_saved = max(0, original_tokens - compressed_tokens)

    # Record telemetry span in Sentry
    record_compression_span(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        tokens_saved=tokens_saved,
        mode=mode_used,
    )

    return {
        "original_prompt": request.prompt,
        "compressed_prompt": final_compressed,
        "mode_used": mode_used,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": tokens_saved,
    }