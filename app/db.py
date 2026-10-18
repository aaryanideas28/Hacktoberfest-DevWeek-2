import sqlite3
import os
from typing import Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "contextswitch.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    cursor.execute("""
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
    );
    """)

    # Automatic schema migration for existing sqlite db files
    for col_def in [
        ("tokens_saved", "INTEGER DEFAULT 0"),
        ("markdown_artifact", "TEXT")
    ]:
        try:
            cursor.execute(f"ALTER TABLE context_snapshots ADD COLUMN {col_def[0]} {col_def[1]};")
        except sqlite3.OperationalError:
            pass  # Column already exists

    
    conn.commit()
    conn.close()

def save_snapshot(
    project_name: str,
    structured_data: Dict[str, Any],
    markdown_artifact: str = "",
    tokens_saved: int = 0
) -> int:
    """Save an extracted context handoff snapshot into SQLite."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("INSERT OR IGNORE INTO projects (name) VALUES (?)", (project_name,))
    cursor.execute("SELECT id FROM projects WHERE name = ?", (project_name,))
    row = cursor.fetchone()
    project_id = row["id"] if row else 1

    tech_stack = ",".join(structured_data.get("tech_stack", [])) if isinstance(structured_data.get("tech_stack"), list) else str(structured_data.get("tech_stack", ""))
    decisions = ";".join(structured_data.get("decisions_made", [])) if isinstance(structured_data.get("decisions_made"), list) else str(structured_data.get("decisions_made", ""))
    constraints = ";".join(structured_data.get("constraints", [])) if isinstance(structured_data.get("constraints"), list) else str(structured_data.get("constraints", ""))
    failed = ";".join(structured_data.get("failed_approaches", [])) if isinstance(structured_data.get("failed_approaches"), list) else str(structured_data.get("failed_approaches", ""))
    next_step = str(structured_data.get("next_step", "None"))

    cursor.execute("""
        INSERT INTO context_snapshots (
            project_id, tech_stack, decisions_made, constraints,
            failed_approaches, next_step, tokens_saved, markdown_artifact
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (project_id, tech_stack, decisions, constraints, failed, next_step, tokens_saved, markdown_artifact))

    conn.commit()
    snapshot_id = cursor.lastrowid
    conn.close()
    return snapshot_id or 1