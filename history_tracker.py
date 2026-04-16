import json
import uuid
from snowflake_utils import get_connection, execute, fetch_df

_table_created = False


def ensure_table():
    global _table_created
    if _table_created:
        return
    execute("""
    CREATE TABLE IF NOT EXISTS JOBPREP_DB.MARTS.INTERVIEW_HISTORY (
        session_id      VARCHAR(64),
        created_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
        company         VARCHAR(256),
        role            VARCHAR(256),
        total_questions INT,
        optimized_count INT,
        avg_score       FLOAT,
        scores_json     VARCHAR(16384)
    )
    """)
    _table_created = True


def save_session(company, role, results):
    ensure_table()

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

    execute(
        """
        INSERT INTO JOBPREP_DB.MARTS.INTERVIEW_HISTORY
        (session_id, company, role, total_questions,
         optimized_count, avg_score, scores_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (session_id, company, role, total, optimized,
         avg_score, json.dumps(summary)),
    )

    return session_id


def load_recent_sessions(limit=5):
    ensure_table()

    df = fetch_df(f"""
    SELECT session_id, created_at, company, role,
           total_questions, optimized_count, avg_score
    FROM JOBPREP_DB.MARTS.INTERVIEW_HISTORY
    ORDER BY created_at DESC
    LIMIT {limit}
    """)

    return [
        {
            "session_id": row.session_id,
            "created_at": str(row.created_at)[:16],
            "company": row.company,
            "role": row.role,
            "total": row.total_questions,
            "optimized": row.optimized_count,
            "avg_score": row.avg_score,
        }
        for row in df.itertuples(index=False)
    ]