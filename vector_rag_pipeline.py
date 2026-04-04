import pandas as pd
import numpy as np
import ast
import re
import json
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

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
        # Snowflake may return uppercase columns; normalize once.
        df.columns = [str(c).lower() for c in df.columns]
        return df
    finally:
        cur.close()


def clean_llm_output(text):
    if isinstance(text, list):
        text = "\n".join(map(str, text))
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    elif text is None:
        return ""

    text = str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def llm(prompt, model="deepseek-r1"):
    query = f"""
    SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
        '{model}',
        $$ {prompt} $$
    ) AS RESPONSE
    """
    result = fetch_pandas_df(query)
    return clean_llm_output(result.iloc[0]["response"])


def get_query_embedding(text):
    query = """
    SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
        'snowflake-arctic-embed-m-v1.5',
        %s
    ) AS EMB
    """
    cur = conn.cursor()
    try:
        cur.execute(query, (text,))
        result = cur.fetchone()[0]
    finally:
        cur.close()

    return np.array(result, dtype=float)


def load_vector_rag_df():
    query = """
    SELECT
        company_name,
        role_name,
        interview_question,
        question_category_enhanced,
        embedding
    FROM JOBPREP_DB.DBT_JOBPREP_DBT_JOBPREP_MARTS.VECTOR_RAG_QUESTION_EMBEDDINGS
    """
    df = fetch_pandas_df(query)
    if "embedding" not in df.columns:
        raise ValueError(
            "Missing 'embedding' column in VECTOR_RAG_QUESTION_EMBEDDINGS. "
            "Run build_vector_rag_index.py to populate embeddings."
        )

    def normalize_embedding(x):
        if x is None:
            return np.array([], dtype=float)
        if isinstance(x, str):
            return np.array(ast.literal_eval(x), dtype=float)
        return np.array(x, dtype=float)

    df["embedding"] = df["embedding"].apply(normalize_embedding)
    df = df[df["embedding"].apply(lambda e: isinstance(e, np.ndarray) and e.size > 0)]
    return df


def build_user_query(company, role, job_description):
    return f"""
Target company: {company}
Target role: {role}

Retrieve technical interview questions relevant for:
- data structures and algorithms
- backend engineering
- distributed systems
- debugging
- system design
- behavioral collaboration

Job description:
{job_description}
"""


def retrieve_vector_rag_questions(company, role, job_description, vector_rag_df, top_k=10):
    user_query = build_user_query(company, role, job_description)
    query_embedding = get_query_embedding(user_query)

    df = vector_rag_df.copy()

    role_lower = role.lower()

    # Strong role filtering
    role_mask = df["role_name"].fillna("").str.lower().apply(
        lambda x: role_lower in x or ("software engineer" in role_lower and "engineer" in x)
    )

    # Exclude obviously wrong roles / domains for SWE comparison
    bad_role_terms = [
        "product manager", "pm", "program manager", "technical program manager",
        "recruiter", "recruiting", "designer", "sales", "marketing"
    ]
    bad_role_mask = df["role_name"].fillna("").str.lower().apply(
        lambda x: any(term in x for term in bad_role_terms)
    )

    bad_question_terms = [
        "why do you want to work", "why did you become", "why google",
        "tell me about your day", "tell me about yourself", "introduce yourself",
        "walk me through your resume", "monetize", "product would you prioritize",
        "onboarding flow", "gmail app", "launch a coding language",
        "google cloud should offer", "recruiting system", "job description for",
        "role of a product manager", "develop yourself professionally",
        "design philosophies of apple", "reduce false negatives",
        "what do you like most about", "how will you develop",
        "how would you define the role", "how would you choose a programming language",
        "explain how you interact with a designer",
        "software engineering discussion board",
        "tell me about a system design problem you solved with machine learning",
    ]
    bad_question_mask = df["interview_question"].fillna("").str.lower().apply(
        lambda x: any(term in x for term in bad_question_terms)
    )

    # Prefer technical categories
    good_category_terms = [
        "array", "string", "tree", "graph", "dynamic", "dp", "stack", "queue",
        "linked", "binary", "system design", "backend", "distributed", "general",
        "behavioral", "algorithms"
    ]
    category_mask = df["question_category_enhanced"].fillna("").str.lower().apply(
        lambda x: any(term in x for term in good_category_terms)
    )

    filtered = df[role_mask & ~bad_role_mask & ~bad_question_mask & category_mask].copy()

    # fallback 1
    if len(filtered) < max(top_k * 3, 30):
        filtered = df[role_mask & ~bad_role_mask & ~bad_question_mask].copy()

    # fallback 2
    if len(filtered) < max(top_k * 2, 20):
        filtered = df[~bad_role_mask & ~bad_question_mask].copy()

    question_embeddings = np.vstack(filtered["embedding"].values)
    scores = cosine_similarity([query_embedding], question_embeddings)[0]

    filtered["score"] = scores

    def boost(row):
        val = 0.0
        q = str(row.get("interview_question", "")).lower()
        cat = str(row.get("question_category_enhanced", "")).lower()
        role_name = str(row.get("role_name", "")).lower()

        if "software engineer" in role_name or "engineer" in role_name:
            val += 0.05
        if any(term in cat for term in ["algorithm", "tree", "graph", "array", "string", "system", "backend", "distributed"]):
            val += 0.03
        if any(term in q for term in ["design", "distributed", "cache", "database", "thread", "debug", "graph", "tree"]):
            val += 0.02

        return val

    filtered["boost"] = filtered.apply(boost, axis=1)
    filtered["final_score"] = filtered["score"] + filtered["boost"]

    return filtered.sort_values("final_score", ascending=False).head(top_k)



def _extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def parse_interview_json(interview_text):
    text = clean_llm_output(interview_text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    candidates = [text]

    extracted = _extract_json_object(text)
    if extracted:
        candidates.append(extracted)

    if text.startswith('"') and text.endswith('"'):
        try:
            unquoted = json.loads(text)
            if isinstance(unquoted, str):
                candidates.append(unquoted)
        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def generate_vector_rag_interview(company, role, job_description, retrieved_questions_df):
    questions = retrieved_questions_df["interview_question"].dropna().tolist()
    text = "\n".join(questions)

    prompt = f"""
You are acting as a real interviewer from {company} for the role of {role}.

Job Description:
{job_description}

Use these retrieved interview questions only as grounding/context:
{text}

Generate a realistic mock interview.

Return ONLY valid JSON in this exact structure:

{{
  "coding_questions": [
    {{
      "title": "",
      "problem_statement": "",
      "example_input_output": "",
      "constraints": "",
      "test_cases": ["", ""]
    }},
    {{
      "title": "",
      "problem_statement": "",
      "example_input_output": "",
      "constraints": "",
      "test_cases": ["", ""]
    }},
    {{
      "title": "",
      "problem_statement": "",
      "example_input_output": "",
      "constraints": "",
      "test_cases": ["", ""]
    }}
  ],
  "system_design": {{
    "title": "",
    "use_case": "",
    "functional_requirements": ["", "", ""],
    "non_functional_requirements": ["", "", ""],
    "key_discussion_points": ["", "", ""]
  }},
  "behavioral": {{
    "question": ""
  }}
}}

Important:
- Return JSON only
- No markdown
- No explanation
- No extra text
"""
    return llm(prompt, model="deepseek-r1")


vector_rag_df = load_vector_rag_df()


def run_vector_rag_interview(company, role, job_description, top_k=10):
    retrieved_df = retrieve_vector_rag_questions(
        company=company,
        role=role,
        job_description=job_description,
        vector_rag_df=vector_rag_df,
        top_k=top_k
    )

    interview = generate_vector_rag_interview(
        company=company,
        role=role,
        job_description=job_description,
        retrieved_questions_df=retrieved_df
    )

    return retrieved_df, interview
