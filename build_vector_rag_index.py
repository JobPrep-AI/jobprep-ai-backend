import pandas as pd
import sys
from pathlib import Path

project_root = Path(r"C:\Users\yslog\OneDrive\Desktop\GenAI Project\jobprep-ai-backend")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from snowflake_connection import get_snowflake_connection

conn = get_snowflake_connection()


def fetch_pandas_df(query, params=None):
    cur = conn.cursor()
    try:
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        df = cur.fetch_pandas_all()
        df.columns = [str(c).lower() for c in df.columns]
        return df
    finally:
        cur.close()

query = """
SELECT
    COMPANY_NAME,
    ROLE_NAME,
    INTERVIEW_QUESTION,
    QUESTION_CATEGORY_ENHANCED
FROM JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.MART_QUESTION_BANK
WHERE INTERVIEW_QUESTION IS NOT NULL
"""

print("Loading interview questions...")
jobs_df = fetch_pandas_df(query)
jobs_df = jobs_df.drop_duplicates(
    subset=["company_name", "role_name", "interview_question"]
)
print(f"Loaded {len(jobs_df)} unique questions")

cur = conn.cursor()

try:
    print("Truncating VECTOR_RAG_QUESTION_EMBEDDINGS...")
    cur.execute("""
    TRUNCATE TABLE JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.VECTOR_RAG_QUESTION_EMBEDDINGS
    """)

    insert_sql = """
    INSERT INTO JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.VECTOR_RAG_QUESTION_EMBEDDINGS
    (
        company_name,
        role_name,
        interview_question,
        question_category_enhanced
    )
    VALUES (%s, %s, %s, %s)
    """

    rows = [
        (
            row.company_name,
            row.role_name,
            row.interview_question,
            row.question_category_enhanced,
        )
        for row in jobs_df.itertuples(index=False)
    ]

    batch_size = 1000
    print("Inserting question rows...")
    for i in range(0, len(rows), batch_size):
        cur.executemany(insert_sql, rows[i : i + batch_size])
        print(f"Inserted {min(i + batch_size, len(rows))}/{len(rows)} rows")

    conn.commit()

    print("Generating embeddings in Snowflake (batched)...")
    batch_size = 500
    while True:
        cur.execute("""
        SELECT COUNT(*)
        FROM JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.VECTOR_RAG_QUESTION_EMBEDDINGS
        WHERE embedding IS NULL
        """)
        remaining = cur.fetchone()[0]
        if remaining == 0:
            break

        cur.execute(f"""
        MERGE INTO JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.VECTOR_RAG_QUESTION_EMBEDDINGS AS t
        USING (
            SELECT
                company_name,
                role_name,
                interview_question,
                question_category_enhanced
            FROM JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.VECTOR_RAG_QUESTION_EMBEDDINGS
            WHERE embedding IS NULL
            LIMIT {batch_size}
        ) AS b
        ON t.company_name = b.company_name
           AND t.role_name = b.role_name
           AND t.interview_question = b.interview_question
           AND COALESCE(t.question_category_enhanced, '') = COALESCE(b.question_category_enhanced, '')
           AND t.embedding IS NULL
        WHEN MATCHED THEN UPDATE SET
            embedding = SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                'snowflake-arctic-embed-m-v1.5',
                b.interview_question
            )
        """)
        conn.commit()
        print(f"Embedded {min(batch_size, remaining)}/{remaining} in current pass")

finally:
    cur.close()

print("Vector RAG index built successfully.")
