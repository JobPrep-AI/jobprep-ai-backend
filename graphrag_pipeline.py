import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import sys
from pathlib import Path

project_root = Path(r"C:\Users\yslog\OneDrive\Desktop\GenAI Project\jobprep-ai-backend")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from snowflake_connection import get_snowflake_connection

conn = get_snowflake_connection()

DEFAULT_JD_REQUIREMENTS = {
    "technical_skills": [],
    "system_topics": [],
    "behavioral_traits": [],
    "priority_requirements": [],
}


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


def clean_llm_output(text):
    if isinstance(text, list):
        text = "\n".join(map(str, text))
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    elif text is None:
        return ""

    text = str(text).strip()

    if text.startswith('"') and text.endswith('"'):
        try:
            unquoted = json.loads(text)
            if isinstance(unquoted, str):
                text = unquoted.strip()
        except Exception:
            pass

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return text.strip()


def llm(prompt, model="llama3.3-70b"):
    query = f"""
    SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
        '{model}',
        $$ {prompt} $$
    ) AS RESPONSE
    """

    result = fetch_pandas_df(query)
    raw_text = result.iloc[0]["response"]
    return clean_llm_output(raw_text)


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


def load_summary_df():
    query = """
    SELECT cluster_id, summary, embedding
    FROM JOBPREP_DB.MARTS.CLUSTER_SUMMARIES
    ORDER BY cluster_id
    """
    df = fetch_pandas_df(query)
    if "embedding" not in df.columns:
        raise ValueError(
            "Missing 'embedding' column in CLUSTER_SUMMARIES. "
            "Run build_graph_index.py to populate embeddings."
        )

    def normalize_embedding(x):
        if isinstance(x, str):
            return np.array(ast.literal_eval(x), dtype=float)
        return np.array(x, dtype=float)

    df["embedding"] = df["embedding"].apply(normalize_embedding)
    return df


def _is_real_question(q: str) -> bool:
    signals = ["?", "how ", "what ", "why ", "design ", "explain ",
               "estimate ", "describe ", "tell me", "implement ",
               "you are", "given ", "write ", "find ", "suppose "]
    if any(s in q.lower() for s in signals):
        return True
    if len(q.split()) <= 6:
        return False
    return True

_expansion_cache = {}

def expand_questions(questions, company, role):
    enriched = []
    for q in questions:
        if _is_real_question(q):
            enriched.append(q)
            continue
        if q in _expansion_cache:
            enriched.append(_expansion_cache[q])
            continue
        prompt = f"""You are a technical interviewer at {company} for a {role} role.
Convert this LeetCode problem title into a full interview-style question.
Title: {q}
Return ONLY JSON.
No explanation.
No markdown: {{"title": "", "problem_statement": "", "example": "", "constraints": ""}}
No markdown, no explanation."""
        raw = llm(prompt, model="llama3.3-70b")
        parsed = parse_json_response(raw)
        if isinstance(parsed, dict):
            result = f"{parsed.get('title', q)}\n{parsed.get('problem_statement', '')}\nExample: {parsed.get('example', '')}\nConstraints: {parsed.get('constraints', '')}"
        else:
            result = q
        _expansion_cache[q] = result
        enriched.append(result)
    return enriched


def load_cluster_questions():
    query = """
    SELECT cluster_id, interview_question
    FROM JOBPREP_DB.MARTS.CLUSTER_QUESTIONS
    ORDER BY cluster_id
    """
    df = fetch_pandas_df(query)

    cluster_questions = defaultdict(list)
    for row in df.itertuples(index=False):
        cluster_questions[int(row.cluster_id)].append(row.interview_question)

    return dict(cluster_questions)


def parse_json_response(text):
    if not isinstance(text, str):
        return None

    text = text.strip()

    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        parsed = json.loads(text)

        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        if isinstance(parsed, dict):
            return parsed

    except:
        pass

    repair_prompt = f"""
Fix this into VALID JSON.

Return ONLY JSON.
No explanation.

Input:
{text}
"""
    repaired = llm(repair_prompt, model="llama3.3-70b")

    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except:
        pass

    return None


def normalize_jd_requirements(value):
    if not isinstance(value, dict):
        return DEFAULT_JD_REQUIREMENTS.copy()

    normalized = {}
    for key in DEFAULT_JD_REQUIREMENTS.keys():
        items = value.get(key, [])
        if isinstance(items, str):
            items = [items] if items.strip() else []
        elif not isinstance(items, list):
            items = []
        normalized[key] = [str(x).strip() for x in items if str(x).strip()]

    return normalized


def _has_any_requirements(req):
    return any(req.get(k) for k in DEFAULT_JD_REQUIREMENTS.keys())


def _unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def fallback_requirements_from_jd(job_description):
    text = (job_description or "").strip()
    lower = text.lower()

    tech_map = [
        ("python", "Python"),
        ("java", "Java"),
        ("c++", "C++"),
        ("sql", "SQL"),
        ("snowflake", "Snowflake"),
        ("spark", "Apache Spark"),
        ("airflow", "Airflow"),
        ("dbt", "dbt"),
        ("pandas", "Pandas"),
        ("aws", "AWS"),
        ("azure", "Azure"),
        ("gcp", "GCP"),
        ("docker", "Docker"),
        ("kubernetes", "Kubernetes"),
    ]
    system_map = [
        ("distributed", "Distributed systems"),
        ("microservice", "Microservices"),
        ("scalable", "Scalable backend design"),
        ("system design", "System design"),
        ("api", "API design"),
        ("event", "Event-driven architecture"),
        ("queue", "Message queues"),
        ("cache", "Caching"),
        ("database", "Database design"),
        ("pipeline", "Data pipelines"),
    ]
    behavioral_map = [
        ("collaborat", "Cross-team collaboration"),
        ("communicat", "Communication"),
        ("ownership", "Ownership"),
        ("lead", "Leadership"),
        ("mentor", "Mentoring"),
        ("stakeholder", "Stakeholder management"),
        ("problem solving", "Problem solving"),
    ]

    tech = [label for kw, label in tech_map if kw in lower][:8]
    systems = [label for kw, label in system_map if kw in lower][:6]
    behavioral = [label for kw, label in behavioral_map if kw in lower][:6]

    lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip()]
    priority = []
    trigger_words = ("must", "required", "requirement", "experience with", "proficiency", "strong")
    for ln in lines:
        ll = ln.lower()
        if any(t in ll for t in trigger_words):
            priority.append(ln[:140])
    if not priority:
        priority = tech[:3] + systems[:2]

    return {
        "technical_skills": _unique_keep_order(tech)[:6],
        "system_topics": _unique_keep_order(systems)[:5],
        "behavioral_traits": _unique_keep_order(behavioral)[:5],
        "priority_requirements": _unique_keep_order(priority)[:6],
    }


def extract_jd_requirements(company, role, job_description):
    prompt = f"""
You MUST return ONLY raw JSON.
No explanation.
Extract the most important interview-relevant requirements from this job description.

Company: {company}
Role: {role}

Job Description:
{job_description}

Return valid JSON only in this format:
{{
  "technical_skills": [],
  "system_topics": [],
  "behavioral_traits": [],
  "priority_requirements": []
}}

Rules:
- Include data structures and algorithms if present
- Include object-oriented programming if present
- Include backend / scalability / distributed systems if present
- Include debugging / observability if present
- Include collaboration / teamwork if present
- Use short theme phrases, not full sentences
- priority_requirements should contain the most important 3 to 5 themes only

Example:
{{
  "technical_skills": ["data structures", "algorithms", "object-oriented programming"],
  "system_topics": ["distributed systems", "scalable backend", "debugging", "system design"],
  "behavioral_traits": ["teamwork", "cross-team collaboration"],
  "priority_requirements": ["algorithms", "scalable backend", "distributed systems", "debugging"]
}}
"""
    raw = llm(prompt, model="llama3.3-70b")
    parsed = normalize_jd_requirements(parse_json_response(raw))

    if _has_any_requirements(parsed):
        return parsed

    heuristic = normalize_jd_requirements(fallback_requirements_from_jd(job_description))
    if _has_any_requirements(heuristic):
        return heuristic

    return DEFAULT_JD_REQUIREMENTS.copy()


def simplify_requirements(jd_requirements):
    jd_requirements = normalize_jd_requirements(jd_requirements)

    simplified = {
        "technical_skills": [],
        "system_topics": [],
        "behavioral_traits": [],
        "priority_requirements": []
    }

    keyword_map = {
        "data structures and algorithms": ["data structures", "algorithms"],
        "object-oriented programming": ["oop", "object-oriented programming"],
        "java": ["java"],
        "python": ["python"],
        "c++": ["c++"],
        "go": ["go"],
        "distributed systems": ["distributed systems"],
        "microservices": ["microservices"],
        "scalable backend design": ["backend", "scalable backend"],
        "system design": ["system design"],
        "database design": ["database", "databases"],
        "databases, caching, and data pipelines": ["databases", "caching", "data pipelines"],
        "concurrency, multithreading, and networking": ["concurrency", "multithreading", "networking"],
        "observability tools, monitoring, and debugging production systems": ["observability", "monitoring", "debugging"],
        "cross-team collaboration": ["collaboration", "teamwork"]
    }

    for key, values in jd_requirements.items():
        expanded = []
        for item in values:
            lower_item = item.lower()

            matched = False
            for phrase, mapped_terms in keyword_map.items():
                if phrase in lower_item:
                    expanded.extend(mapped_terms)
                    matched = True

            if not matched:
                expanded.append(item)

        simplified[key] = _unique_keep_order(expanded)

    return simplified


def build_requirement_query(company, role, jd_requirements):
    jd_requirements = normalize_jd_requirements(jd_requirements)
    tech = ", ".join(jd_requirements.get("technical_skills", []))
    systems = ", ".join(jd_requirements.get("system_topics", []))
    behavior = ", ".join(jd_requirements.get("behavioral_traits", []))
    priority = ", ".join(jd_requirements.get("priority_requirements", []))

    return f"""
Company: {company}
Role: {role}

Technical skills: {tech}
System topics: {systems}
Behavioral traits: {behavior}
Priority requirements: {priority}
"""


def build_user_query(company, role, job_description):
    return f"""
Company: {company}
Role: {role}

Job Description:
{job_description}

Generate a mock interview aligned with this company, role, and job description.
"""


def retrieve_top_clusters(company, role, jd_requirements, summary_df, top_k=5):
    jd_requirements = normalize_jd_requirements(jd_requirements)
    user_query = build_requirement_query(company, role, jd_requirements)

    query_embedding = get_query_embedding(user_query)

    cluster_embeddings = np.vstack(summary_df["embedding"].values)
    scores = cosine_similarity([query_embedding], cluster_embeddings)[0]

    ranked = summary_df.copy()
    ranked["score"] = scores

    role_lower = role.lower()
    company_lower = company.lower()

    def boost(summary):
        s = str(summary).lower()
        val = 0.0

        if role_lower in s:
            val += 0.05
        if company_lower in s:
            val += 0.03

        for term in jd_requirements.get("technical_skills", []):
            if term.lower() in s:
                val += 0.02

        for term in jd_requirements.get("system_topics", []):
            if term.lower() in s:
                val += 0.02

        for term in jd_requirements.get("behavioral_traits", []):
            if term.lower() in s:
                val += 0.02

        return val

    ranked["boost"] = ranked["summary"].apply(boost)
    ranked["final_score"] = ranked["score"] + ranked["boost"]

    return ranked.sort_values("final_score", ascending=False).head(top_k)


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
                extracted_unquoted = _extract_json_object(unquoted)
                if extracted_unquoted:
                    candidates.append(extracted_unquoted)
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


def detect_missing_requirements(jd_requirements, selected_questions, top_clusters=None):
    simplified = simplify_requirements(jd_requirements)

    selected_text = " ".join(selected_questions).lower()

    cluster_text = ""
    if top_clusters is not None and "summary" in top_clusters.columns:
        cluster_text = " ".join(top_clusters["summary"].astype(str).tolist()).lower()

    combined_text = f"{selected_text} {cluster_text}"

    missing = []

    ignore_terms = {"python", "java", "c++", "go"}

    for group in ["technical_skills", "system_topics", "behavioral_traits", "priority_requirements"]:
        for req in simplified.get(group, []):
            req_lower = req.lower().strip()

            if req_lower in ignore_terms:
                continue

            if req_lower not in combined_text:
                missing.append(req)

    return _unique_keep_order(missing)


def coerce_interview_json(interview_text):
    parsed = parse_interview_json(interview_text)
    if isinstance(parsed, dict):
        return parsed

    repair_prompt = f"""
Convert the following interview content into valid JSON with EXACT keys:
- coding_questions (list of 3 items)
- system_design (object)
- behavioral (object)

Rules:
- Return JSON only
- No markdown
- No commentary
- No <think>
- Ensure all expected fields exist, use empty strings/lists if missing

Input:
{interview_text}
"""
    repaired = llm(repair_prompt, model="llama3.3-70b")
    return parse_interview_json(repaired)


def generate_mock_interview(company, role, job_description, selected_questions, jd_requirements, missing_requirements):
    jd_requirements = normalize_jd_requirements(jd_requirements)
    text = "\n".join(selected_questions)

    tech = ", ".join(jd_requirements.get("technical_skills", []))
    systems = ", ".join(jd_requirements.get("system_topics", []))
    behavior = ", ".join(jd_requirements.get("behavioral_traits", []))
    priority = ", ".join(jd_requirements.get("priority_requirements", []))
    missing = ", ".join(missing_requirements)

    prompt = f"""
You MUST return ONLY raw JSON.
Do NOT include:
- explanations
- extra text
- markdown
- code blocks

You are acting as a real interviewer from {company} for the role of {role}.

Job Description:
{job_description}

--------------------------------------------------

STRICT RULES:

1. Coding questions MUST ONLY be Data Structures & Algorithms:
   - arrays, strings, graphs, trees, dynamic programming, recursion

2. DO NOT include:
   - SQL queries
   - database questions
   - backend/API questions

3. If retrieved questions are not DSA, ignore them

4. Ensure:
   - 3 coding questions
   - 1 system design question
   - 1 behavioral question

--------------------------------------------------

CODING QUESTIONS (3):

Each MUST include:
- title
- problem_statement (FULL detailed problem)
- example_input_output
- constraints
- test_cases

DO NOT leave any field empty.

--------------------------------------------------

SYSTEM DESIGN (1):

You MUST return FULL structure:

{{
  "title": "",
  "use_case": "",
  "functional_requirements": [],
  "non_functional_requirements": [],
  "key_discussion_points": []
}}

⚠️ IMPORTANT:
- NOT just "Design X"
- Must include explanation
- Should be realistic (e.g., URL shortener, chat system, notification system)

--------------------------------------------------

BEHAVIORAL (1):

You MUST return:

{{
  "question": ""
}}

⚠️ IMPORTANT:
- Must be a real HR-style question
- Example:
  "Tell me about a time you handled conflict in a team"
  "Describe a challenging project and how you solved it"

--------------------------------------------------

Retrieved questions:
{text}

--------------------------------------------------

Return ONLY valid JSON.
No explanation. No extra text.
Follow the EXACT structure below:

{{
  "coding_questions": [
    {{
      "title": "",
      "problem_statement": "",
      "example_input_output": "",
      "constraints": "",
      "test_cases": []
    }}
  ],
  "system_design": {{
    "title": "",
    "use_case": "",
    "functional_requirements": [],
    "non_functional_requirements": [],
    "key_discussion_points": []
  }},
  "behavioral": {{
    "question": ""
  }}
}}
"""
    return llm(prompt, model="llama3.3-70b")


def collect_relevant_questions(top_clusters, jobs_df, cluster_questions, company, role, max_questions=12):
    selected_questions = []

    exact_df = jobs_df[
        (jobs_df["company_name"].fillna("").str.lower() == company.lower()) &
        (jobs_df["role_name"].fillna("").str.lower() == role.lower())
    ]

    role_df = jobs_df[
        jobs_df["role_name"].fillna("").str.lower().str.contains(role.lower(), na=False)
    ]

    exact_questions = set(exact_df["interview_question"].dropna().tolist())
    role_questions = set(role_df["interview_question"].dropna().tolist())

    q_to_cat = dict(
        zip(
            jobs_df["interview_question"].astype(str),
            jobs_df["question_category_enhanced"].astype(str)
        )
    )

    def category_bucket(cat):
        c = (cat or "").lower()

        if any(x in c for x in ["system", "backend", "distributed", "design", "architecture"]):
            return "system"

        if any(x in c for x in ["behavioral", "general", "leadership", "hr"]):
            return "behavioral"

        if any(x in c for x in ["sql", "database", "db", "query"]):
            return "sql"

        if any(x in c for x in [
            "array", "string", "graph", "tree", "dp",
            "algorithm", "recursion", "linked", "stack",
            "queue", "search", "sort"
        ]):
            return "coding"

        return "other"

    for cid in top_clusters["cluster_id"]:
        for q in cluster_questions.get(cid, []):
            cat = q_to_cat.get(q, "")
            bucket = category_bucket(cat)
            if q in exact_questions and q not in selected_questions:
                if bucket == "coding":
                    selected_questions.append(q)

    for cid in top_clusters["cluster_id"]:
        for q in cluster_questions.get(cid, []):
            cat = q_to_cat.get(q, "")
            bucket = category_bucket(cat)
            if q in role_questions and q not in selected_questions:
                if bucket == "coding":
                    selected_questions.append(q)

    for cid in top_clusters["cluster_id"]:
        for q in cluster_questions.get(cid, []):
            cat = q_to_cat.get(q, "")
            bucket = category_bucket(cat)
            if q not in selected_questions and bucket == "coding":
                selected_questions.append(q)
            if len(selected_questions) >= max_questions:
                return selected_questions[:max_questions]

    return selected_questions[:max_questions]


# -------------------------------
# MODULE-LEVEL DATA LOAD
# -------------------------------
query_jobs = """
SELECT *
FROM JOBPREP_DB.MARTS.MART_QUESTION_BANK
"""

jobs_df = fetch_pandas_df(query_jobs)
jobs_df = jobs_df.drop_duplicates(
    subset=["company_name", "role_name", "interview_question"]
)
jobs_df = jobs_df.dropna(
    subset=["company_name", "role_name", "interview_question", "question_category_enhanced"]
)

summary_df = load_summary_df()
cluster_questions = load_cluster_questions()


def run_graphrag_interview(company, role, job_description, top_k=5):
    jd_requirements = extract_jd_requirements(company, role, job_description)

    top_clusters = retrieve_top_clusters(
        company=company,
        role=role,
        jd_requirements=jd_requirements,
        summary_df=summary_df,
        top_k=top_k
    )

    selected_questions = collect_relevant_questions(
        top_clusters=top_clusters,
        jobs_df=jobs_df,
        cluster_questions=cluster_questions,
        company=company,
        role=role,
        max_questions=12
    )

    selected_questions = expand_questions(selected_questions, company, role)

    missing_requirements = detect_missing_requirements(
        jd_requirements=jd_requirements,
        selected_questions=selected_questions,
        top_clusters=top_clusters
    )

    missing_requirements = missing_requirements[:4]

    interview = generate_mock_interview(
        company=company,
        role=role,
        job_description=job_description,
        selected_questions=selected_questions,
        jd_requirements=jd_requirements,
        missing_requirements=missing_requirements
    )

    return top_clusters, selected_questions, interview, jd_requirements, missing_requirements