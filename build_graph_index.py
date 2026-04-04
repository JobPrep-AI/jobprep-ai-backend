import pandas as pd
import networkx as nx
import community as community_louvain
from collections import defaultdict
from snowflake_connection import get_snowflake_connection

conn = get_snowflake_connection()


def fetch_pandas_df(query, params=None):
    cur = conn.cursor()
    try:
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        return cur.fetch_pandas_all()
    finally:
        cur.close()

def llm(prompt):

    query = f"""
    SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
        'llama3.3-70b',
        $$ {prompt} $$
    ) AS RESPONSE
    """

    cur = conn.cursor()
    try:
        cur.execute(query)
        return cur.fetchone()[0]
    finally:
        cur.close()


print("Loading interview questions...")

query_jobs = """
SELECT *
FROM JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.MART_QUESTION_BANK
"""

jobs_df = fetch_pandas_df(query_jobs)

jobs_df = jobs_df.drop_duplicates(
    subset=["COMPANY_NAME", "ROLE_NAME", "INTERVIEW_QUESTION"]
)

jobs_df = jobs_df.dropna(
    subset=["COMPANY_NAME", "ROLE_NAME", "INTERVIEW_QUESTION", "QUESTION_CATEGORY_ENHANCED"]
)

jobs_df["COMPANY_NAME"] = jobs_df["COMPANY_NAME"].astype(str).str.strip()
jobs_df["ROLE_NAME"] = jobs_df["ROLE_NAME"].astype(str).str.strip()
jobs_df["INTERVIEW_QUESTION"] = jobs_df["INTERVIEW_QUESTION"].astype(str).str.strip()
jobs_df["QUESTION_CATEGORY_ENHANCED"] = jobs_df["QUESTION_CATEGORY_ENHANCED"].astype(str).str.strip()

jobs_df = jobs_df[jobs_df["INTERVIEW_QUESTION"].str.len() > 15]


print("Building knowledge graph...")

G = nx.Graph()

for _, row in jobs_df.iterrows():

    company = row["COMPANY_NAME"]
    role = row["ROLE_NAME"]
    question = row["INTERVIEW_QUESTION"]
    category = row["QUESTION_CATEGORY_ENHANCED"]

    G.add_node(company, type="company")
    G.add_node(role, type="role")
    G.add_node(question, type="question")
    G.add_node(category, type="category")

    G.add_edge(company, role, relation="hires_for")
    G.add_edge(role, question, relation="asks")
    G.add_edge(question, category, relation="belongs_to")

    # Optional richer graph if columns exist
    if "DIFFICULTY_LEVEL" in jobs_df.columns:
        difficulty = row["DIFFICULTY_LEVEL"]
        if pd.notna(difficulty):
            difficulty_node = f"difficulty:{difficulty}"
            G.add_node(difficulty_node, type="difficulty")
            G.add_edge(question, difficulty_node, relation="has_difficulty")

    if "IS_TECHNICAL" in jobs_df.columns:
        is_technical = row["IS_TECHNICAL"]
        if pd.notna(is_technical):
            technical_node = f"technical:{is_technical}"
            G.add_node(technical_node, type="question_type")
            G.add_edge(question, technical_node, relation="is_technical")

    if "IS_BEHAVIORAL" in jobs_df.columns:
        is_behavioral = row["IS_BEHAVIORAL"]
        if pd.notna(is_behavioral):
            behavioral_node = f"behavioral:{is_behavioral}"
            G.add_node(behavioral_node, type="question_type")
            G.add_edge(question, behavioral_node, relation="is_behavioral")


print("Detecting communities...")

partition = community_louvain.best_partition(G)

clusters = defaultdict(list)

for node, cid in partition.items():
    clusters[cid].append(node)

cluster_questions = {}

for cid, nodes in clusters.items():

    questions = []

    for node in nodes:
        if G.nodes[node]["type"] == "question":
            questions.append(node)

    if len(questions) >= 3:
        cluster_questions[cid] = questions

print(f"Total communities detected: {len(clusters)}")
print(f"Usable question clusters after filtering: {len(cluster_questions)}")

print("Generating cluster summaries...")

cluster_summaries = {}

for cid, questions in cluster_questions.items():

    if len(questions) == 0:
        continue

    text = "\n".join(questions[:15])

    prompt = f"""
You are summarizing a cluster of interview questions.

Questions:
{text}

Return in this exact format:

Topic: ...
Role/Domain: ...
Skills: ...
Question Types: ...
Summary: ...

Rules:
- Be specific, not generic
- Mention whether the cluster is about algorithms, backend, system design, data engineering, behavioral, product, etc.
- Mention the main skills being tested
- Mention the likely interview type if possible
- Keep the summary compact but precise
"""

    summary = llm(prompt)

    cluster_summaries[cid] = summary


print("Saving summaries to Snowflake...")

cur = conn.cursor()

cur.execute("""
TRUNCATE TABLE JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.CLUSTER_SUMMARIES
""")

cur.execute("""
TRUNCATE TABLE JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.CLUSTER_QUESTIONS
""")


insert_summary_sql = """
INSERT INTO JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.CLUSTER_SUMMARIES
(cluster_id, summary)
VALUES (%s, %s)
"""

summary_rows = [(cid, summary) for cid, summary in cluster_summaries.items()]
if summary_rows:
    cur.executemany(insert_summary_sql, summary_rows)
print(f"Inserted {len(summary_rows)} rows into CLUSTER_SUMMARIES")


insert_question_sql = """
INSERT INTO JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.CLUSTER_QUESTIONS
(cluster_id, interview_question)
VALUES (%s, %s)
"""

question_rows = []
for cid, questions in cluster_questions.items():
    for q in questions:
        question_rows.append((cid, q))

batch_size = 1000
for i in range(0, len(question_rows), batch_size):
    cur.executemany(insert_question_sql, question_rows[i : i + batch_size])
    print(f"Inserted {min(i + batch_size, len(question_rows))}/{len(question_rows)} question rows")


conn.commit()

print("Generating embeddings in Snowflake...")

cur.execute("""
UPDATE JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.CLUSTER_SUMMARIES
SET embedding = SNOWFLAKE.CORTEX.EMBED_TEXT_768(
    'snowflake-arctic-embed-m-v1.5',
    summary
)
""")

conn.commit()

cur.close()

print("Graph index build complete.")