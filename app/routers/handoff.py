from fastapi import APIRouter
from pydantic import BaseModel

from app.services.extractor import extract_context_items, format_markdown_artifact
from app.services.gemma import extract_context_with_gemma
from app.db import save_snapshot

router = APIRouter(prefix="/api/v1", tags=["Handoff"])

class HandoffRequest(BaseModel):
    project_name: str
    raw_transcript: str

@router.post("/handoff")
def handoff_endpoint(request: HandoffRequest):
    # Try Gemma 2 neural extraction first
    gemma_data = extract_context_with_gemma(request.raw_transcript)
    
    if gemma_data:
        structured_data = gemma_data
    else:
        # Fallback to rule-based keyword extraction
        structured_data = extract_context_items(request.raw_transcript)

    # Format into copyable System Prompt Artifact
    formatted_md = format_markdown_artifact(structured_data, request.project_name)

    # Calculate token savings
    original_tokens = len(request.raw_transcript) // 4
    artifact_tokens = len(formatted_md) // 4
    tokens_saved = max(0, original_tokens - artifact_tokens)

    # Save snapshot record to SQLite
    snapshot_id = save_snapshot(
        project_name=request.project_name,
        structured_data=structured_data,
        markdown_artifact=formatted_md,
        tokens_saved=tokens_saved,
    )

    return {
        "snapshot_id": snapshot_id,
        "project_name": request.project_name,
        "structured_data": structured_data,
        "formatted_markdown": formatted_md,
        "tokens_saved": tokens_saved,
    }