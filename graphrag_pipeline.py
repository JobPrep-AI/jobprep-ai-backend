import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import sys
import streamlit as st
from pathlib import Path
import time
import random
import logging

logger = logging.getLogger(__name__)

project_root = Path(r"C:\Users\yslog\OneDrive\Desktop\GenAI Project\jobprep-ai-backend")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from snowflake_utils import llm, get_embedding, fetch_df, clean_llm_output

DEFAULT_JD_REQUIREMENTS = {
    "technical_skills": [],
    "system_topics": [],
    "behavioral_traits": [],
    "priority_requirements": [],
}

@st.cache_data
def load_summary_df():
    query = """
    SELECT cluster_id, summary, embedding
    FROM JOBPREP_DB.MARTS.CLUSTER_SUMMARIES
    ORDER BY cluster_id
    """
    df = fetch_df(query)
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

@st.cache_data
def load_cluster_questions():
    query = """
    SELECT cluster_id, interview_question
    FROM JOBPREP_DB.MARTS.CLUSTER_QUESTIONS
    ORDER BY cluster_id
    """
    df = fetch_df(query)

    cluster_questions = defaultdict(list)
    for row in df.itertuples(index=False):
        cluster_questions[int(row.cluster_id)].append(row.interview_question)

    return dict(cluster_questions)


def parse_json_response(text, _repair_attempt=False):
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

    # Only attempt LLM repair once — never repair a repair
    if _repair_attempt:
        logger.warning("parse_json_response: repair attempt also failed, giving up.")
        return None

    logger.warning("parse_json_response: standard parse failed, attempting LLM repair...")

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

    # Parse the repaired text but flag it as already repaired
    return parse_json_response(repaired, _repair_attempt=True)


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
        ("python", "Python"), ("java", "Java"), ("c++", "C++"),
        ("sql", "SQL"), ("snowflake", "Snowflake"), ("spark", "Apache Spark"),
        ("airflow", "Airflow"), ("dbt", "dbt"), ("pandas", "Pandas"),
        ("aws", "AWS"), ("azure", "Azure"), ("gcp", "GCP"),
        ("docker", "Docker"), ("kubernetes", "Kubernetes"),
    ]
    system_map = [
        ("distributed", "Distributed systems"), ("microservice", "Microservices"),
        ("scalable", "Scalable backend design"), ("system design", "System design"),
        ("api", "API design"), ("event", "Event-driven architecture"),
        ("queue", "Message queues"), ("cache", "Caching"),
        ("database", "Database design"), ("pipeline", "Data pipelines"),
    ]
    behavioral_map = [
        ("collaborat", "Cross-team collaboration"), ("communicat", "Communication"),
        ("ownership", "Ownership"), ("lead", "Leadership"),
        ("mentor", "Mentoring"), ("stakeholder", "Stakeholder management"),
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
        "java": ["java"], "python": ["python"], "c++": ["c++"], "go": ["go"],
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

    query_embedding = get_embedding(user_query)

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
- coding_questions (list of 4 items)
- system_design_questions (list of 2 objects)
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


def generate_mock_interview(company, role, job_description, selected_questions,
                            jd_requirements, missing_requirements, weak_areas=None):
    jd_requirements = normalize_jd_requirements(jd_requirements)
    variety_seed = int(time.time()) % 1000
    text = "\n".join(selected_questions)

    # Build weak area instruction for personalized question targeting
    if weak_areas and len(weak_areas) > 0:
        wa_lines = []
        slots = ["Q2 Medium", "Q3 Hard", "Q4 Hard"]
        for i, wa in enumerate(weak_areas[:3]):
            topic = wa.get("topic", "")
            score = wa.get("avg_score", 0)
            slot = slots[i] if i < len(slots) else "Q4 Hard"
            wa_lines.append(
                f"- {slot} MUST target '{topic}' — user scored {score}/10 here"
            )
        weak_area_instruction = "\n".join(wa_lines)
    else:
        weak_area_instruction = "No specific weak areas — generate a balanced interview."

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

Variety seed (use this to ensure unique questions): {variety_seed}
Generate questions that are DIFFERENT from typical interview questions.
Pick uncommon but valid DSA problems.

STRICT RULES:

1. Coding questions MUST ONLY be Data Structures & Algorithms:
   - arrays, strings, graphs, trees, dynamic programming, recursion

2. DO NOT include:
   - SQL queries
   - database questions
   - backend/API questions

3. If retrieved questions are not DSA, ignore them

4. Ensure:
   - 4 coding questions
   - 2 system design question
   - 1 behavioral question

5. Test cases must have VERIFIED correct expected outputs.
   Trace through your own algorithm manually before writing expected_output.
   A wrong expected_output is worse than no test case.

--------------------------------------------------

CODING QUESTIONS (4) — ORDERED BY DIFFICULTY:

Question 1 MUST be EASY   — arrays, strings, basic hashing
Question 2 MUST be MEDIUM — trees, graphs, sliding window, binary search
Question 3 MUST be HARD   — dynamic programming, advanced graphs, complex optimization
Question 4 MUST be HARD   — different hard topic from Q3, complex optimization

WEAK AREA TARGETING:
{weak_area_instruction}

Each MUST include:
- title
- difficulty  ("Easy", "Medium", or "Hard")
- problem_statement (FULL detailed problem)
- example_input_output
- constraints
- test_cases (list of {{"input": "...", "expected_output": "..."}} objects)
- starter_code with python, java, and cpp keys

STARTER CODE RULES:
- Function name must match the problem
- Parameter types must be correct for each language
- Input reading must match the test case input format exactly
- Do NOT implement the solution — body is pass or return default only
- Java class must always be named Solution
- CPP must always include bits/stdc++.h
- Python must handle input parsing and print the result

VARIETY RULE:
- Do NOT reuse questions from previous interviews
- Each interview must have completely different problem titles and scenarios
- For Easy: choose a different array/string pattern each time
- For Medium: rotate between trees, graphs, sliding window, binary search
- For Hard: rotate between DP, advanced graphs, complex optimization
- The behavioral question must be different each time
- The system design must use a different real-world system each time

PYTHON starter_code template:
import json

def solve(param):
    # Write your solution here
    pass

# Input handling
try:
    data = input().strip()
    print(solve(data))
except:
    pass

JAVA starter_code template:
import java.util.*;

public class Solution {{
    public static ReturnType solve(ParamType param) {{
        // Write your solution here
        return defaultValue;
    }}

    public static void main(String[] args) {{
        Scanner sc = new Scanner(System.in);
        String input = sc.nextLine().trim();
        System.out.println(solve(input));
    }}
}}

CPP starter_code template:
#include <bits/stdc++.h>
using namespace std;

ReturnType solve(ParamType param) {{
    // Write your solution here
    return defaultValue;
}}

int main() {{
    string input;
    getline(cin, input);
    cout << solve(input) << endl;
    return 0;
}}

DO NOT leave any field empty.

--------------------------------------------------

SYSTEM DESIGN (2):

You MUST return 2 system design questions as a list.
SD1: A standard system design (URL shortener, notification system, etc.)
SD2: A more complex distributed system design (ride sharing, news feed, etc.)

Each MUST have:
- title
- use_case
- functional_requirements
- non_functional_requirements
- key_discussion_points

--------------------------------------------------

BEHAVIORAL (1):

You MUST return:

{{
  "question": ""
}}

--------------------------------------------------

Retrieved questions:
{text}

--------------------------------------------------

Return ONLY valid JSON. No explanation. No extra text.

{{
  "coding_questions": [
    {{
      "title": "",
      "difficulty": "Easy",
      "problem_statement": "",
      "example_input_output": "",
      "constraints": "",
      "test_cases": [{{"input": "", "expected_output": ""}}],
      "starter_code": {{
        "python": "",
        "java": "",
        "cpp": ""
      }}
    }}
  ],
  "system_design_questions": [
    {{
      "title": "",
      "use_case": "",
      "functional_requirements": [],
      "non_functional_requirements": [],
      "key_discussion_points": []
    }}
  ],
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
        if any(x in c for x in ["array", "string", "graph", "tree", "dp",
                                  "algorithm", "recursion", "linked", "stack",
                                  "queue", "search", "sort"]):
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
        pool = [
            q for q in cluster_questions.get(cid, [])
            if q not in selected_questions
            and category_bucket(q_to_cat.get(q, "")) == "coding"
        ]
        random.shuffle(pool)
        for q in pool:
            selected_questions.append(q)
            if len(selected_questions) >= max_questions:
                return selected_questions[:max_questions]

    return selected_questions[:max_questions]


# -------------------------------
# MODULE-LEVEL DATA LOAD
# -------------------------------
@st.cache_data
def load_jobs_df():
    df = fetch_df("""
    SELECT *
    FROM JOBPREP_DB.MARTS.MART_QUESTION_BANK
    """)
    df = df.drop_duplicates(
        subset=["company_name", "role_name", "interview_question"]
    )
    df = df.dropna(
        subset=["company_name", "role_name", "interview_question", "question_category_enhanced"]
    )
    return df

jobs_df = load_jobs_df()
summary_df = load_summary_df()
cluster_questions = load_cluster_questions()


def generate_single_question(topic: str, q_type: str, company: str = "", role: str = "") -> dict:
    """
    Generate a single question for Quick Practice mode.
    q_type: 'coding', 'system_design', or 'behavioral'
    topic: specific topic like 'Arrays & Strings', 'Dynamic Programming', etc.
    """
    variety_seed = int(time.time()) % 1000

    if q_type == "coding":
        prompt = f"""
You are a technical interviewer generating a single coding question.

Topic: {topic}
Company context: {company or "a top tech company"}
Role context: {role or "Software Engineer"}
Variety seed: {variety_seed}

Generate ONE coding question on the topic: {topic}

Rules:
- Must be a pure DSA problem related to {topic}
- Include a complete problem statement
- Include example input/output
- Include constraints
- Include 3 test cases with verified expected outputs
- Include starter code for python, java, and cpp
- Difficulty should be Medium

Return ONLY valid JSON. No explanation. No extra text.

{{
  "title": "",
  "difficulty": "Medium",
  "topic": "{topic}",
  "problem_statement": "",
  "example_input_output": "",
  "constraints": "",
  "test_cases": [{{"input": "", "expected_output": ""}}],
  "starter_code": {{
    "python": "",
    "java": "",
    "cpp": ""
  }}
}}
"""

    elif q_type == "system_design":
        prompt = f"""
You are a technical interviewer generating a single system design question.

Variety seed: {variety_seed}

Generate ONE realistic system design question.

Rules:
- Must be a real-world system (e.g. URL shortener, notification system, ride sharing)
- Include use case, functional requirements, non-functional requirements, key discussion points
- Must be different each time

Return ONLY valid JSON. No explanation.

{{
  "title": "",
  "use_case": "",
  "functional_requirements": [],
  "non_functional_requirements": [],
  "key_discussion_points": []
}}
"""

    elif q_type == "behavioral":
        prompt = f"""
You are a technical interviewer generating a single behavioral question.

Variety seed: {variety_seed}

Generate ONE behavioral interview question.

Rules:
- Must be a real HR-style question
- Must be different each time
- Examples: "Tell me about a time you handled conflict", "Describe a challenging project"

Return ONLY valid JSON. No explanation.

{{
  "question": ""
}}
"""
    else:
        return {}

    response = llm(prompt, model="llama3.3-70b")
    parsed = parse_json_response(response)
    if isinstance(parsed, dict):
        parsed["q_type"] = q_type
        return parsed
    return {}

def run_graphrag_interview(company, role, job_description, top_k=8):
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