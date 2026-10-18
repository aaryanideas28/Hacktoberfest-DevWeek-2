import pytest
from app.services.extractor import (
    NO_NEXT_STEP,
    extract_structured_context,
    format_handoff_markdown,
    merge_contexts,
    normalize_context,
    parse_json_object,
    split_sentences,
    validate_and_merge,
)


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


def test_tech_stack_is_detected_not_hardcoded():
    ctx = extract_structured_context("We are building this with Django and PostgreSQL. Deploy on Docker.")
    assert "Django" in ctx["tech_stack"]
    assert "PostgreSQL" in ctx["tech_stack"]
    assert "FastAPI" not in ctx["tech_stack"]          # used to be hard-coded for every project
    assert "Gemma" not in ctx["tech_stack"]


def test_no_invented_defaults():
    ctx = extract_structured_context("Hello there.")
    assert ctx["constraints"] == []
    assert ctx["failed_approaches"] == []
    assert ctx["next_step"] == ""


def test_negation_constraints_survive():
    ctx = extract_structured_context("You must not change the public API format. We decided to use SQLite.")
    assert any("must not change the public api format" in c.lower() for c in ctx["constraints"])


def test_sentence_split_keeps_file_names_and_versions():
    sents = split_sentences("Edit app/main.py and the v2.0 config. Next, fix the bug.")
    assert sents[0] == "Edit app/main.py and the v2.0 config."
    assert len(sents) == 2


def test_key_files_and_endpoints_extracted():
    ctx = extract_structured_context("Edit app/routers/handoff.py and call /api/v1/handoff. Keep node.js out of it.")
    assert "app/routers/handoff.py" in ctx["key_files"]
    assert "/api/v1/handoff" in ctx["key_files"]
    assert "node.js" not in [f.lower() for f in ctx["key_files"]]


def test_next_step_is_the_most_recent_one():
    ctx = extract_structured_context("Next, write the schema. Later we hit a snag. Next, add the login page.")
    assert "login page" in ctx["next_step"]


def test_questions_are_not_facts():
    ctx = extract_structured_context("Should we decide to use Redis? We decided to use SQLite.")
    assert all("Redis" not in d for d in ctx["decisions_made"])


def test_completed_vs_unresolved():
    ctx = extract_structured_context("We finished the login page. The export is still not working.")
    assert any("login page" in c for c in ctx["completed_work"])
    assert any("export" in o for o in ctx["open_issues"])


def test_parse_json_object_handles_fences_and_chatter():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": {"b": "}"}} Hope that helps') == {"a": {"b": "}"}}
    assert parse_json_object("not json at all") is None
    assert parse_json_object(None) is None


def test_normalize_context_fixes_types_and_aliases():
    ctx = normalize_context({
        "decisions": "Use SQLite\n- Use FastAPI",
        "constraints": ["Do not break the API", "do not break the api", "none"],
        "next_steps": ["Write tests", "Ship"],
        "tech_stack": None,
    })
    assert ctx["decisions_made"] == ["Use SQLite", "Use FastAPI"]
    assert ctx["constraints"] == ["Do not break the API"]
    assert ctx["next_step"] == "Write tests"
    assert ctx["tech_stack"] == []
    assert ctx["open_issues"] == [] and ctx["project_summary"] == ""


def test_validate_drops_invented_items_and_restores_constraints():
    transcript = (
        "We decided to use SQLite on port 8000. You must not change the response format. "
        "Next, add the export button."
    )
    llm = {
        "decisions_made": ["Use SQLite on port 8000", "Use PostgreSQL on port 5432"],  # 2nd is invented
        "constraints": [],                                                          # LLM missed it
        "next_step": "Add the export button",
        "tech_stack": ["SQLite", "Kubernetes"],                                     # Kubernetes invented
    }
    rule = extract_structured_context(transcript)
    data, notes = validate_and_merge(llm, rule, transcript)
    assert data["decisions_made"] == ["Use SQLite on port 8000"]
    assert "Kubernetes" not in data["tech_stack"]
    assert any("must not change the response format" in c.lower() for c in data["constraints"])
    assert notes and "removed" in notes[0]


def test_validate_falls_back_to_rule_next_step():
    transcript = "Next, add the export button."
    data, _ = validate_and_merge({"next_step": "Rewrite everything in Rust"}, extract_structured_context(transcript), transcript)
    assert "export button" in data["next_step"]


def test_merge_contexts_dedupes_and_last_next_step_wins():
    a = {"decisions_made": ["Use SQLite"], "next_step": "Step A", "project_summary": "First"}
    b = {"decisions_made": ["use sqlite", "Use FastAPI"], "next_step": "Step B", "project_summary": "Second"}
    m = merge_contexts([a, b])
    assert m["decisions_made"] == ["Use SQLite", "Use FastAPI"]
    assert m["next_step"] == "Step B"
    assert m["project_summary"] == "First"


def test_format_supports_swapped_args_and_empty_data():
    md = format_handoff_markdown({"decisions_made": ["Use SQLite"]}, "Swapped")
    assert "Swapped" in md and "Use SQLite" in md
    empty = format_handoff_markdown("Empty", {})
    assert "None reported" in empty
    assert NO_NEXT_STEP in empty


def test_format_flattens_multiline_items():
    md = format_handoff_markdown("P", {"constraints": ["Do not\nbreak the API"]})
    assert "- Do not break the API" in md