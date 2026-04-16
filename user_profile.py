"""
user_profile.py
---------------
Save/load user sessions, weak areas, and history.
Called after every evaluation to update the user's profile.
"""

import json
import uuid
import logging
from snowflake_utils import fetch_df, execute, executemany

logger = logging.getLogger(__name__)


# -------------------------------
# SAVE SESSION
# -------------------------------
def save_user_session(user_id: str, company: str, role: str, results: list) -> str:
    """
    Save a completed interview session for a user.
    Also updates USER_WEAK_AREAS with latest scores.
    Returns the session_id.
    """
    session_id = str(uuid.uuid4())[:16]

    # Calculate stats
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

    # Collect weak and strong areas
    weak_areas = []
    strong_areas = []
    for r in results:
        evaluation = r.get("evaluation", {})
        q_type = evaluation.get("question_type", "coding")
        if q_type == "behavioral":
            continue
        scores = evaluation.get("scores", {})
        for k, v in scores.items():
            topic = k.replace("_", " ")
            if v < 7:
                weak_areas.append(topic)
            elif v >= 8:
                strong_areas.append(topic)
        for w in evaluation.get("weaknesses", []):
            weak_areas.append(w[:100])

    weak_areas  = list(set(weak_areas))
    strong_areas = list(set(strong_areas))

    # Get attempt number for this user
    try:
        attempt_df = fetch_df("""
            SELECT COUNT(*) AS cnt
            FROM JOBPREP_DB.MARTS.USER_SESSIONS
            WHERE user_id = %s AND role = %s AND company = %s
        """, (user_id, role, company))
        attempt_number = int(attempt_df.iloc[0]["cnt"]) + 1
    except Exception:
        attempt_number = 1

    scores_summary = [
        {
            "q": i + 1,
            "type": r.get("evaluation", {}).get("question_type", "coding"),
            "scores": r.get("evaluation", {}).get("scores", {}),
            "optimized": r.get("evaluation", {}).get("is_optimized", False),
        }
        for i, r in enumerate(results)
    ]

    # Insert session
    try:
        execute("""
            INSERT INTO JOBPREP_DB.MARTS.USER_SESSIONS
            (session_id, user_id, company, role, attempt_number,
             total_questions, optimized_count, avg_score,
             weak_areas, strong_areas, scores_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id, user_id, company, role, attempt_number,
            total, optimized, avg_score,
            json.dumps(weak_areas),
            json.dumps(strong_areas),
            json.dumps(scores_summary)
        ))
        logger.info(f"Saved session {session_id} for user {user_id}")
    except Exception as e:
        logger.error(f"save_user_session insert failed: {e}")

    # Update weak areas table
    _update_weak_areas(user_id, results)

    return session_id


# -------------------------------
# UPDATE WEAK AREAS
# -------------------------------
def _update_weak_areas(user_id: str, results: list):
    """
    Upsert weak area scores for the user.
    Tracks average score and frequency per topic.
    """
    topic_scores = {}

    for r in results:
        evaluation = r.get("evaluation", {})
        if evaluation.get("question_type") == "behavioral":
            continue
        scores = evaluation.get("scores", {})
        for k, v in scores.items():
            topic = k.replace("_", " ")
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(v)

    for topic, scores in topic_scores.items():
        avg = round(sum(scores) / len(scores), 1)
        try:
            existing = fetch_df("""
                SELECT frequency, avg_score
                FROM JOBPREP_DB.MARTS.USER_WEAK_AREAS
                WHERE user_id = %s AND topic = %s
            """, (user_id, topic))

            if existing.empty:
                execute("""
                    INSERT INTO JOBPREP_DB.MARTS.USER_WEAK_AREAS
                    (user_id, topic, avg_score, frequency)
                    VALUES (%s, %s, %s, 1)
                """, (user_id, topic, avg))
            else:
                old_freq  = int(existing.iloc[0]["frequency"])
                old_avg   = float(existing.iloc[0]["avg_score"])
                new_freq  = old_freq + 1
                new_avg   = round((old_avg * old_freq + avg) / new_freq, 1)
                execute("""
                    UPDATE JOBPREP_DB.MARTS.USER_WEAK_AREAS
                    SET avg_score = %s, frequency = %s,
                        last_seen = CURRENT_TIMESTAMP
                    WHERE user_id = %s AND topic = %s
                """, (new_avg, new_freq, user_id, topic))
        except Exception as e:
            logger.warning(f"Weak area update failed for {topic}: {e}")


# -------------------------------
# LOAD USER HISTORY
# -------------------------------
def load_user_sessions(user_id: str, limit: int = 10) -> list:
    """
    Load recent sessions for a user, newest first.
    """
    try:
        df = fetch_df("""
            SELECT session_id, created_at, company, role,
                   attempt_number, total_questions, optimized_count,
                   avg_score, weak_areas, strong_areas
            FROM JOBPREP_DB.MARTS.USER_SESSIONS
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, limit))

        sessions = []
        for row in df.itertuples(index=False):
            sessions.append({
                "session_id":     row.session_id,
                "created_at":     str(row.created_at)[:16],
                "company":        row.company,
                "role":           row.role,
                "attempt_number": row.attempt_number,
                "total":          row.total_questions,
                "optimized":      row.optimized_count,
                "avg_score":      row.avg_score,
                "weak_areas":     json.loads(row.weak_areas or "[]"),
                "strong_areas":   json.loads(row.strong_areas or "[]"),
            })
        return sessions
    except Exception as e:
        logger.error(f"load_user_sessions failed: {e}")
        return []


# -------------------------------
# LOAD WEAK AREAS
# -------------------------------
def load_weak_areas(user_id: str, min_frequency: int = 1) -> list:
    """
    Load user's persistent weak areas sorted by avg_score ascending.
    These are used to personalize the next interview.
    """
    try:
        df = fetch_df("""
            SELECT topic, avg_score, frequency
            FROM JOBPREP_DB.MARTS.USER_WEAK_AREAS
            WHERE user_id = %s
              AND frequency >= %s
              AND avg_score < 7
            ORDER BY avg_score ASC
            LIMIT 6
        """, (user_id, min_frequency))

        return [
            {
                "topic":     row.topic,
                "avg_score": row.avg_score,
                "frequency": row.frequency,
            }
            for row in df.itertuples(index=False)
        ]
    except Exception as e:
        logger.error(f"load_weak_areas failed: {e}")
        return []


# -------------------------------
# SESSION COMPARISON DATA
# -------------------------------
def get_score_trend(user_id: str, role: str, company: str) -> list:
    """
    Return score trend across attempts for a specific role+company.
    Used for the session comparison chart.
    """
    try:
        df = fetch_df("""
            SELECT attempt_number, avg_score, optimized_count,
                   total_questions, created_at, weak_areas
            FROM JOBPREP_DB.MARTS.USER_SESSIONS
            WHERE user_id = %s AND role = %s AND company = %s
            ORDER BY attempt_number ASC
        """, (user_id, role, company))

        return [
            {
                "attempt":    row.attempt_number,
                "avg_score":  row.avg_score,
                "optimized":  row.optimized_count,
                "total":      row.total_questions,
                "date":       str(row.created_at)[:10],
                "weak_areas": json.loads(row.weak_areas or "[]"),
            }
            for row in df.itertuples(index=False)
        ]
    except Exception as e:
        logger.error(f"get_score_trend failed: {e}")
        return []