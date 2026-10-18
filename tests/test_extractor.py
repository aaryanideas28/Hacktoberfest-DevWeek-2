import pytest
from app.services.extractor import extract_structured_context, format_handoff_markdown

def test_extract_structured_context():
    transcript = "We decided to use FastAPI and SQLite. Next, implement the UI tab."
    context = extract_structured_context(transcript)
    assert "decisions_made" in context
    assert len(context["decisions_made"]) > 0

def test_format_handoff_markdown():
    data = {
        "decisions_made": ["Use SQLite"],
        "constraints": ["Keep latency low"],
        "failed_approaches": ["None"],
        "next_step": "Test routers"
    }
    md = format_handoff_markdown("DemoProject", data)
    assert "# 🔄 ContextSwitch Handoff Artifact" in md
    assert "DemoProject" in md
    assert "Use SQLite" in md
