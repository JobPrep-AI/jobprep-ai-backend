import json
import uuid
from snowflake_connection import get_snowflake_connection

_conn = None

def _get_conn():
    global _conn
    if _conn is None:
        _conn = get_snowflake_connection()
    return _conn


def ensure_table():
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS JOBPREP_DB.MARTS.INTERVIEW_HISTORY (
            session_id    VARCHAR(64),
            created_at    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
            company       VARCHAR(256),
            role          VARCHAR(256),
            total_questions INT,
            optimized_count INT,
            avg_score     FLOAT,
            scores_json   VARCHAR(16384)
        )
        """)
        conn.commit()
    finally:
        cur.close()


def save_session(company, role, results):
    ensure_table()
    conn = _get_conn()

    session_id = str(uuid.uuid4())[:16]
    total = len(results)
    optimized = sum(
        1 for r in results
        if r.get("evaluation", {}).get("is_optimized", False)
    )

    score_list = []
    for r in results:
        s = r.get("evaluation", {}).get("scores", {})
        if s:
            score_list.append(sum(s.values()) / len(s))

    avg_score = round(sum(score_list) / len(score_list), 1) if score_list else 0.0

    summary = [
        {
            "q": i + 1,
            "type": r.get("evaluation", {}).get("question_type", "coding"),
            "scores": r.get("evaluation", {}).get("scores", {}),
            "optimized": r.get("evaluation", {}).get("is_optimized", False),
        }
        for i, r in enumerate(results)
    ]

    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO JOBPREP_DB.MARTS.INTERVIEW_HISTORY
            (session_id, company, role, total_questions,
             optimized_count, avg_score, scores_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (session_id, company, role, total, optimized,
             avg_score, json.dumps(summary)),
        )
        conn.commit()
    finally:
        cur.close()

    return session_id


def load_recent_sessions(limit=5):
    ensure_table()
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(f"""
        SELECT session_id, created_at, company, role,
               total_questions, optimized_count, avg_score
        FROM JOBPREP_DB.MARTS.INTERVIEW_HISTORY
        ORDER BY created_at DESC
        LIMIT {limit}
        """)
        rows = cur.fetchall()
    finally:
        cur.close()

    return [
        {
            "session_id": r[0],
            "created_at": str(r[1])[:16],
            "company": r[2],
            "role": r[3],
            "total": r[4],
            "optimized": r[5],
            "avg_score": r[6],
        }
        for r in rows
    ]