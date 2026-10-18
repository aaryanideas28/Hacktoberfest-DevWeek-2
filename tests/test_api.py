import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Isolated sqlite file so tests never touch contextswitch.db."""
    import app.db as db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return db


@pytest.fixture
def no_llm(monkeypatch):
    """Force the rule-based path (no network, deterministic)."""
    monkeypatch.setattr("app.routers.handoff.extract_context_with_gemma", lambda t: None)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gemma_active" in data
    assert "sentry_active" in data

def test_compress_api_default():
    response = client.post("/api/v1/compress", json={"prompt": "Can you please help me write tests?"})
    assert response.status_code == 200
    data = response.json()
    assert "compressed_prompt" in data
    assert "tokens_saved" in data

def test_compress_api_entropy_mode():
    response = client.post("/api/v1/compress", json={
        "prompt": "Can you please write a complex algorithm? Thanks.",
        "mode": "gemma_entropy",
        "drop_rate": 0.40
    })
    assert response.status_code == 200
    data = response.json()
    assert data["mode_used"] == "gemma_entropy"
    assert "engine_used" in data
    assert "latency_ms" in data

def test_handoff_api(temp_db, no_llm):
    response = client.post("/api/v1/handoff", json={
        "project_name": "TestRun",
        "raw_transcript": "We decided to keep tests passing."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "TestRun"
    assert "formatted_markdown" in data
    assert data["engine"] == "rule-based"
    assert data["snapshot_id"]


def test_handoff_rejects_empty_transcript(temp_db, no_llm):
    r = client.post("/api/v1/handoff", json={"project_name": "X", "raw_transcript": "   "})
    assert r.status_code == 400


def test_handoff_blank_project_name_gets_default(temp_db, no_llm):
    r = client.post("/api/v1/handoff", json={"project_name": " ", "raw_transcript": "We decided to use SQLite."})
    assert r.status_code == 200
    assert r.json()["project_name"] == "Untitled Project"


def test_handoff_verifies_llm_output_and_keeps_constraints(temp_db, monkeypatch):
    transcript = (
        "We decided to use SQLite on port 8000. You must not change the response format. "
        "Next, add the export button."
    )
    fake = {
        "decisions_made": ["Use SQLite on port 8000", "Use PostgreSQL on port 5432"],
        "constraints": [],
        "next_step": "Add the export button",
        "tech_stack": ["SQLite"],
    }
    monkeypatch.setattr("app.routers.handoff.extract_context_with_gemma", lambda t: fake)
    r = client.post("/api/v1/handoff", json={"project_name": "Verify", "raw_transcript": transcript})
    assert r.status_code == 200
    data = r.json()
    assert data["engine"].startswith("gemma-api")
    sd = data["structured_data"]
    assert sd["decisions_made"] == ["Use SQLite on port 8000"]          # invented item removed
    assert any("must not change the response format" in c.lower() for c in sd["constraints"])
    assert "must not change the response format" in data["formatted_markdown"].lower()
    assert data["notes"]


def test_handoff_survives_llm_crash(temp_db, monkeypatch):
    def boom(_):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr("app.routers.handoff.extract_context_with_gemma", boom)
    r = client.post("/api/v1/handoff", json={"project_name": "Crash", "raw_transcript": "We decided to use SQLite."})
    assert r.status_code == 200
    assert r.json()["engine"] == "rule-based"


def test_handoff_still_returns_artifact_if_db_fails(monkeypatch, no_llm):
    def bad_save(**kwargs):
        raise RuntimeError("disk full")
    monkeypatch.setattr("app.routers.handoff.save_snapshot", bad_save)
    r = client.post("/api/v1/handoff", json={"project_name": "NoDB", "raw_transcript": "We decided to use SQLite."})
    assert r.status_code == 200
    data = r.json()
    assert data["snapshot_id"] is None
    assert "SQLite" in data["formatted_markdown"]


def test_handoff_roundtrip_and_switching(temp_db, no_llm):
    first = client.post("/api/v1/handoff", json={
        "project_name": "RT", "raw_transcript": "We decided to use SQLite. Next, add auth."}).json()
    second = client.post("/api/v1/handoff", json={
        "project_name": "RT", "raw_transcript": "We decided to use SQLite. Next, add the export button."}).json()
    assert second["snapshot_id"] > first["snapshot_id"]

    one = client.get(f"/api/v1/handoff/{first['snapshot_id']}")
    assert one.status_code == 200
    assert one.json()["project_name"] == "RT"
    assert "add auth" in one.json()["structured_data"]["next_step"]

    latest = client.get("/api/v1/handoff/latest", params={"project_name": "RT"})
    assert latest.status_code == 200
    assert latest.json()["snapshot_id"] == second["snapshot_id"]
    assert "export button" in latest.json()["formatted_markdown"]

    listing = client.get("/api/v1/handoff", params={"project_name": "RT"}).json()["snapshots"]
    assert [s["snapshot_id"] for s in listing] == [second["snapshot_id"], first["snapshot_id"]]

    assert any(p["project_name"] == "RT" and p["snapshots"] == 2
               for p in client.get("/api/v1/projects").json()["projects"])


def test_handoff_missing_snapshot_is_404(temp_db):
    assert client.get("/api/v1/handoff/999999").status_code == 404
    assert client.get("/api/v1/handoff/latest", params={"project_name": "Nope"}).status_code == 404


def test_db_roundtrip_is_lossless_for_commas_and_semicolons(temp_db):
    sd = {"constraints": ["Never use a; or , separator in IDs"], "tech_stack": ["Python"], "next_step": "go"}
    sid = temp_db.save_snapshot("Lossless", sd, "# md", 5)
    got = temp_db.get_snapshot(sid)
    assert got["structured_data"]["constraints"] == ["Never use a; or , separator in IDs"]
    assert got["tokens_saved"] == 5