import json
import os
import sqlite3
from contextlib import closing
from typing import Any, Dict, List, Optional

# Override with CONTEXTSWITCH_DB (useful for tests); default is <project root>/contextswitch.db
DB_PATH = os.getenv("CONTEXTSWITCH_DB") or os.path.join(os.path.dirname(__file__), "..", "contextswitch.db")

_LEGACY_LIST_SEP = {
    "tech_stack": ",",
    "decisions_made": ";",
    "constraints": ";",
    "failed_approaches": ";",
}


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with closing(get_db()) as conn:
        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS context_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                tech_stack TEXT NOT NULL,
                decisions_made TEXT NOT NULL,
                constraints TEXT NOT NULL,
                failed_approaches TEXT NOT NULL,
                next_step TEXT NOT NULL,
                tokens_saved INTEGER DEFAULT 0,
                markdown_artifact TEXT,
                structured_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """)

            # Automatic schema migration for existing sqlite db files
            for col, ddl in (
                ("tokens_saved", "INTEGER DEFAULT 0"),
                ("markdown_artifact", "TEXT"),
                ("structured_json", "TEXT"),
            ):
                try:
                    conn.execute(f"ALTER TABLE context_snapshots ADD COLUMN {col} {ddl};")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshots_project "
                "ON context_snapshots(project_id, id DESC);"
            )


def _join(value: Any, sep: str) -> str:
    """Legacy human-readable column. The real data lives losslessly in structured_json."""
    if isinstance(value, (list, tuple)):
        return sep.join(str(v) for v in value)
    return "" if value is None else str(value)


def save_snapshot(
    project_name: str,
    structured_data: Dict[str, Any],
    markdown_artifact: str = "",
    tokens_saved: int = 0,
) -> int:
    """Save an extracted context handoff snapshot into SQLite and return its id."""
    sd = structured_data or {}
    with closing(get_db()) as conn:
        with conn:
            conn.execute("INSERT OR IGNORE INTO projects (name) VALUES (?)", (project_name,))
            row = conn.execute("SELECT id FROM projects WHERE name = ?", (project_name,)).fetchone()
            project_id = row["id"]

            cur = conn.execute(
                """
                INSERT INTO context_snapshots (
                    project_id, tech_stack, decisions_made, constraints,
                    failed_approaches, next_step, tokens_saved, markdown_artifact, structured_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    _join(sd.get("tech_stack"), ","),
                    _join(sd.get("decisions_made"), ";"),
                    _join(sd.get("constraints"), ";"),
                    _join(sd.get("failed_approaches"), ";"),
                    str(sd.get("next_step") or ""),
                    int(tokens_saved or 0),
                    markdown_artifact,
                    json.dumps(sd, ensure_ascii=False),
                ),
            )
            return cur.lastrowid


def _row_to_snapshot(row: sqlite3.Row) -> Dict[str, Any]:
    structured: Optional[Dict[str, Any]] = None
    if row["structured_json"]:
        try:
            structured = json.loads(row["structured_json"])
        except ValueError:
            structured = None
    if structured is None:  # rows saved before structured_json existed
        structured = {"next_step": row["next_step"]}
        for key, sep in _LEGACY_LIST_SEP.items():
            structured[key] = [p.strip() for p in (row[key] or "").split(sep) if p.strip()]
    return {
        "snapshot_id": row["id"],
        "project_name": row["project_name"],
        "created_at": row["created_at"],
        "tokens_saved": row["tokens_saved"],
        "structured_data": structured,
        "formatted_markdown": row["markdown_artifact"] or "",
    }


def get_snapshot(snapshot_id: int) -> Optional[Dict[str, Any]]:
    with closing(get_db()) as conn:
        row = conn.execute(
            """
            SELECT s.*, p.name AS project_name
            FROM context_snapshots s JOIN projects p ON p.id = s.project_id
            WHERE s.id = ?
            """,
            (snapshot_id,),
        ).fetchone()
    return _row_to_snapshot(row) if row else None


def list_snapshots(project_name: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Newest first. Lightweight rows (no markdown / full data); use get_snapshot for details."""
    limit = max(1, min(int(limit), 100))
    sql = (
        "SELECT s.id, s.created_at, s.tokens_saved, s.next_step, p.name AS project_name "
        "FROM context_snapshots s JOIN projects p ON p.id = s.project_id "
    )
    params: list = []
    if project_name:
        sql += "WHERE p.name = ? "
        params.append(project_name)
    sql += "ORDER BY s.id DESC LIMIT ?"
    params.append(limit)
    with closing(get_db()) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "snapshot_id": r["id"],
            "project_name": r["project_name"],
            "created_at": r["created_at"],
            "tokens_saved": r["tokens_saved"],
            "next_step": r["next_step"],
        }
        for r in rows
    ]


def latest_snapshot(project_name: str) -> Optional[Dict[str, Any]]:
    rows = list_snapshots(project_name, 1)
    return get_snapshot(rows[0]["snapshot_id"]) if rows else None


def list_projects() -> List[Dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT p.name, COUNT(s.id) AS snapshots, MAX(s.created_at) AS last_snapshot
            FROM projects p LEFT JOIN context_snapshots s ON s.project_id = p.id
            GROUP BY p.id ORDER BY last_snapshot DESC, p.id DESC
            """
        ).fetchall()
    return [
        {"project_name": r["name"], "snapshots": r["snapshots"], "last_snapshot": r["last_snapshot"]}
        for r in rows
    ]