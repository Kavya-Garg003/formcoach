"""
Lightweight SQLite storage layer.
PRD 6.4 says "SQLite or flat JSON -- no need for heavier infra for a student
project." SQLite is used here so queries (history, deltas) stay simple.
"""
from __future__ import annotations
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "formcoach.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    exercise_type TEXT NOT NULL,
    date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    rep_index INTEGER NOT NULL,
    joint_angles TEXT NOT NULL,
    deviations TEXT NOT NULL,
    phase_durations_ms TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS chat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def create_session(user_id: str, exercise_type: str) -> dict:
    session_id = str(uuid.uuid4())
    date = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, exercise_type, date) VALUES (?,?,?,?)",
            (session_id, user_id, exercise_type, date),
        )
    return {"session_id": session_id, "user_id": user_id, "exercise_type": exercise_type, "date": date}


def log_rep(session_id: str, rep_index: int, joint_angles: dict, deviations: list[str],
            phase_durations_ms: dict | None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO reps (session_id, rep_index, joint_angles, deviations, phase_durations_ms) "
            "VALUES (?,?,?,?,?)",
            (
                session_id,
                rep_index,
                json.dumps(joint_angles),
                json.dumps(deviations),
                json.dumps(phase_durations_ms) if phase_durations_ms else None,
            ),
        )


def log_chat(session_id: str, role: str, content: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_log (session_id, role, content, ts) VALUES (?,?,?,?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat()),
        )


def get_session_meta(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None


def get_session_deviation_summary(session_id: str) -> dict[str, int]:
    with get_conn() as conn:
        rows = conn.execute("SELECT deviations FROM reps WHERE session_id=?", (session_id,)).fetchall()
    summary: dict[str, int] = {}
    for row in rows:
        for dev in json.loads(row["deviations"]):
            summary[dev] = summary.get(dev, 0) + 1
    return summary


def get_rep_count(session_id: str) -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as c FROM reps WHERE session_id=?", (session_id,)).fetchone()
        return row["c"] if row else 0


def get_user_sessions(user_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY date ASC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_chat(session_id: str, limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_log WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def build_session_summary(session_id: str) -> dict:
    """Builds the JSON shape from PRD section 6.3, including delta vs the
    user's immediately preceding session for the same exercise type."""
    meta = get_session_meta(session_id)
    if not meta:
        raise ValueError(f"Unknown session_id {session_id}")

    dev_summary = get_session_deviation_summary(session_id)
    rep_count = get_rep_count(session_id)
    top_recurring = sorted(dev_summary, key=dev_summary.get, reverse=True)[:3]

    # find the previous session for this user + exercise type
    siblings = [
        s for s in get_user_sessions(meta["user_id"])
        if s["exercise_type"] == meta["exercise_type"] and s["date"] < meta["date"]
    ]
    delta: dict[str, int] = {}
    if siblings:
        prior = siblings[-1]
        prior_summary = get_session_deviation_summary(prior["session_id"])
        for key in set(dev_summary) | set(prior_summary):
            delta[key] = dev_summary.get(key, 0) - prior_summary.get(key, 0)

    return {
        "session_id": session_id,
        "date": meta["date"],
        "exercise_type": meta["exercise_type"],
        "rep_count": rep_count,
        "deviation_summary": dev_summary,
        "top_recurring_errors": top_recurring,
        "delta_vs_prior_session": delta,
    }
