from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db import get_snapshot, latest_snapshot, list_projects, list_snapshots, save_snapshot
from app.services.extractor import (
    extract_context_items,
    format_markdown_artifact,
    validate_and_merge,
)
from app.services.gemma import extract_context_with_gemma
from app.services.gemma_entropy import count_tokens

router = APIRouter(prefix="/api/v1", tags=["Handoff"])


class HandoffRequest(BaseModel):
    project_name: str
    raw_transcript: str


@router.post("/handoff")
def handoff_endpoint(request: HandoffRequest):
    transcript = (request.raw_transcript or "").strip()
    project_name = (request.project_name or "").strip() or "Untitled Project"
    if not transcript:
        raise HTTPException(status_code=400, detail="raw_transcript cannot be empty.")

    notes: List[str] = []

    # 1. Deterministic extraction always runs (cheap): it is the fallback AND the safety net.
    rule_data = extract_context_items(transcript)

    # 2. LLM extraction (optional). Its output is verified, never trusted.
    llm_data = None
    try:
        llm_data = extract_context_with_gemma(transcript)
    except Exception as e:  # noqa: BLE001
        notes.append(f"AI extraction failed ({type(e).__name__}); used rule-based extraction.")

    if llm_data:
        structured_data, merge_notes = validate_and_merge(llm_data, rule_data, transcript)
        notes.extend(merge_notes)
        engine = "gemma-api (verified against transcript)"
    else:
        structured_data = rule_data
        engine = "rule-based"
        if not notes:
            notes.append("AI extraction was unavailable; used rule-based extraction. Please review the result.")

    # 3. Format into the copyable artifact for the new chat
    formatted_md = format_markdown_artifact(structured_data, project_name)

    # 4. Token accounting (same counter for both sides)
    original_tokens = count_tokens(transcript)
    artifact_tokens = count_tokens(formatted_md)
    tokens_saved = max(0, original_tokens - artifact_tokens)

    # 5. Persist. A database problem must not lose the artifact the user just generated.
    snapshot_id: Optional[int] = None
    try:
        snapshot_id = save_snapshot(
            project_name=project_name,
            structured_data=structured_data,
            markdown_artifact=formatted_md,
            tokens_saved=tokens_saved,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[handoff] could not save snapshot: {e}")
        notes.append("The snapshot could not be saved to the database, but the artifact is below.")

    return {
        "snapshot_id": snapshot_id,
        "project_name": project_name,
        "structured_data": structured_data,
        "formatted_markdown": formatted_md,
        "tokens_saved": tokens_saved,
        "original_tokens": original_tokens,
        "artifact_tokens": artifact_tokens,
        "engine": engine,
        "notes": notes,
    }


# NOTE: fixed paths must be declared before "/handoff/{snapshot_id}".
@router.get("/handoff/latest")
def latest_handoff(project_name: str = Query(..., min_length=1)):
    """Most recent handoff for a project: what a new chat should start from."""
    snap = latest_snapshot(project_name.strip())
    if not snap:
        raise HTTPException(status_code=404, detail="No snapshot found for this project.")
    return snap


@router.get("/handoff/{snapshot_id}")
def read_handoff(snapshot_id: int):
    snap = get_snapshot(snapshot_id)
    if not snap:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return snap


@router.get("/handoff")
def list_handoffs(
    project_name: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    return {"snapshots": list_snapshots(project_name.strip() if project_name else None, limit)}


@router.get("/projects")
def projects():
    return {"projects": list_projects()}