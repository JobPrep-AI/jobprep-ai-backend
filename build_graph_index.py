import pandas as pd
import networkx as nx
import community as community_louvain
from collections import defaultdict
from snowflake_utils import llm, fetch_df, executemany, execute

print("Loading interview questions...")

query_jobs = """
SELECT *
FROM JOBPREP_DB.MARTS.MART_QUESTION_BANK
"""

jobs_df = fetch_df(query_jobs)

jobs_df = jobs_df.drop_duplicates(
    subset=["company_name", "role_name", "interview_question"]
)

jobs_df = jobs_df.dropna(
    subset=["company_name", "role_name", "interview_question", "question_category_enhanced"]
)

jobs_df["company_name"] = jobs_df["company_name"].astype(str).str.strip()
jobs_df["role_name"] = jobs_df["role_name"].astype(str).str.strip()
jobs_df["interview_question"] = jobs_df["interview_question"].astype(str).str.strip()
jobs_df["question_category_enhanced"] = jobs_df["question_category_enhanced"].astype(str).str.strip()

jobs_df = jobs_df[jobs_df["interview_question"].str.len() > 15]


print("Building knowledge graph...")

G = nx.Graph()

# --- Add nodes ---
for _, row in jobs_df.iterrows():
    company  = row["company_name"]
    role     = row["role_name"]
    question = row["interview_question"]
    category = row["question_category_enhanced"]

    G.add_node(company,  type="company")
    G.add_node(role,     type="role")
    G.add_node(question, type="question")
    G.add_node(category, type="category")

    # Weak role/company edges — don't let 4 roles dominate clustering
    G.add_edge(company, role,     relation="hires_for",  weight=0.3)
    G.add_edge(role,    question, relation="asks",        weight=0.3)

    # Strong category edges — this is what drives clustering
    G.add_edge(question, category, relation="belongs_to", weight=2.0)

    if "difficulty_level" in jobs_df.columns:
        difficulty = row["difficulty_level"]
        if pd.notna(difficulty):
            difficulty_node = f"difficulty:{difficulty}"
            G.add_node(difficulty_node, type="difficulty")
            G.add_edge(question, difficulty_node,
                       relation="has_difficulty", weight=0.5)

    if "is_technical" in jobs_df.columns:
        is_technical = row["is_technical"]
        if pd.notna(is_technical):
            technical_node = f"technical:{is_technical}"
            G.add_node(technical_node, type="question_type")
            G.add_edge(question, technical_node,
                       relation="is_technical", weight=0.5)

    if "is_behavioral" in jobs_df.columns:
        is_behavioral = row["is_behavioral"]
        if pd.notna(is_behavioral):
            behavioral_node = f"behavioral:{is_behavioral}"
            G.add_node(behavioral_node, type="question_type")
            G.add_edge(question, behavioral_node,
                       relation="is_behavioral", weight=0.5)

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


print("Detecting communities...")

partition = community_louvain.best_partition(G, resolution=2.0, weight="weight")

clusters = defaultdict(list)

for node, cid in partition.items():
    clusters[cid].append(node)

cluster_questions = {}

for cid, nodes in clusters.items():
    questions = [node for node in nodes if G.nodes[node]["type"] == "question"]
    if len(questions) >= 30:
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


print("Saving summaries to Snowflake (atomic swap)...")

try:
    # Step 1 — write to temp tables
    execute("CREATE OR REPLACE TABLE JOBPREP_DB.MARTS.CLUSTER_SUMMARIES_TMP LIKE JOBPREP_DB.MARTS.CLUSTER_SUMMARIES")
    execute("CREATE OR REPLACE TABLE JOBPREP_DB.MARTS.CLUSTER_QUESTIONS_TMP LIKE JOBPREP_DB.MARTS.CLUSTER_QUESTIONS")

    insert_summary_sql = """
    INSERT INTO JOBPREP_DB.MARTS.CLUSTER_SUMMARIES_TMP
    (cluster_id, summary)
    VALUES (%s, %s)
    """
    summary_rows = [(cid, summary) for cid, summary in cluster_summaries.items()]
    if summary_rows:
        executemany(insert_summary_sql, summary_rows)
    print(f"Inserted {len(summary_rows)} rows into CLUSTER_SUMMARIES_TMP")

    insert_question_sql = """
    INSERT INTO JOBPREP_DB.MARTS.CLUSTER_QUESTIONS_TMP
    (cluster_id, interview_question)
    VALUES (%s, %s)
    """
    question_rows = []
    for cid, questions in cluster_questions.items():
        for q in questions:
            question_rows.append((cid, q))

    batch_size = 1000
    for i in range(0, len(question_rows), batch_size):
        executemany(insert_question_sql, question_rows[i : i + batch_size])
        print(f"Inserted {min(i + batch_size, len(question_rows))}/{len(question_rows)} question rows")

    # Step 2 — generate embeddings on temp table
    print("Generating embeddings in Snowflake...")
    execute("""
    UPDATE JOBPREP_DB.MARTS.CLUSTER_SUMMARIES_TMP
    SET embedding = SNOWFLAKE.CORTEX.EMBED_TEXT_768(
        'snowflake-arctic-embed-m-v1.5',
        summary
    )
    """)

    # Step 3 — everything succeeded, swap temp into production
    print("Swapping temp tables into production...")
    execute("ALTER TABLE JOBPREP_DB.MARTS.CLUSTER_SUMMARIES SWAP WITH JOBPREP_DB.MARTS.CLUSTER_SUMMARIES_TMP")
    execute("ALTER TABLE JOBPREP_DB.MARTS.CLUSTER_QUESTIONS SWAP WITH JOBPREP_DB.MARTS.CLUSTER_QUESTIONS_TMP")

    # Step 4 — drop old data (now in _TMP after swap)
    execute("DROP TABLE IF EXISTS JOBPREP_DB.MARTS.CLUSTER_SUMMARIES_TMP")
    execute("DROP TABLE IF EXISTS JOBPREP_DB.MARTS.CLUSTER_QUESTIONS_TMP")

    print("Graph index build complete.")

except Exception as e:
    print(f"ERROR during index build: {e}")
    print("Production tables were NOT modified. Cleaning up temp tables...")
    execute("DROP TABLE IF EXISTS JOBPREP_DB.MARTS.CLUSTER_SUMMARIES_TMP")
    execute("DROP TABLE IF EXISTS JOBPREP_DB.MARTS.CLUSTER_QUESTIONS_TMP")
    raise