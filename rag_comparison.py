from graphrag_pipeline import run_graphrag_interview
from vector_rag_pipeline import run_vector_rag_interview
from graphrag_pipeline import clean_llm_output, llm

import json
import re
import pandas as pd

# ---------------------------------------------------------------------------
# 3 test cases — generic → moderate → niche
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "generic_swe",
        "label": "Generic SWE (Google)",
        "company": "Google",
        "role": "Software Engineer",
        "job_description": """
Strong knowledge of data structures and algorithms including arrays, trees,
graphs, dynamic programming, and sorting. Experience with object-oriented
design, debugging, and writing clean, testable code. Ability to work
collaboratively in a team environment.
"""
    },
    {
        "id": "data_engineer",
        "label": "Data Engineer (Snowflake)",
        "company": "Snowflake",
        "role": "Data Engineer",
        "job_description": """
Experience building and maintaining scalable data pipelines using Apache Spark,
Airflow, and dbt. Proficiency in SQL and Python. Familiarity with data
warehouse design, ETL/ELT patterns, and cloud platforms such as AWS or GCP.
Strong understanding of data modeling, partitioning strategies, and
pipeline observability.
"""
    },
    {
        "id": "distributed_swe",
        "label": "Distributed Systems SWE (Google)",
        "company": "Google",
        "role": "Software Engineer",
        "job_description": """
Experience with consensus algorithms (Raft, Paxos),
distributed transaction management across microservices,
low-latency systems under high write throughput,
cross-region failover design, and strong debugging skills
for distributed tracing and observability.
"""
    },
]

RETRIEVAL_CRITERIA = [
    "jd_relevance",
    "noise_ratio",
    "question_completeness",
    "jd_requirement_extraction",
    "gap_awareness",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_json_response(text):
    text = clean_llm_output(text)
    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)
        except Exception:
            pass
    text = re.sub(r"^```(?:json)?\s*", "", str(text).strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", str(text).strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                return parsed
            except Exception:
                return None
    return None

# ---------------------------------------------------------------------------
# Retrieval quality scorer
# ---------------------------------------------------------------------------

def score_retrieval_quality(
    label, retrieved_questions, job_description,
    company, role, jd_requirements=None, missing_requirements=None,
):
    questions_text = "\n".join(
        f"  {i+1}. {q[:400]}" for i, q in enumerate(retrieved_questions[:12])
    )

    if jd_requirements and isinstance(jd_requirements, dict):
        jd_req_text = json.dumps(jd_requirements, indent=2)
        extraction_note = (
            f"This pipeline extracted the following structured JD requirements "
            f"before retrieval:\n{jd_req_text}"
        )
    else:
        extraction_note = (
            "This pipeline did NOT extract structured JD requirements. "
            "It embedded the raw JD text directly."
        )

    if missing_requirements:
        gap_note = (
            "This pipeline detected these missing requirements after retrieval:\n"
            + "\n".join(f"  - {m}" for m in missing_requirements)
        )
    else:
        gap_note = (
            "This pipeline has NO gap detection. "
            "It does not report which JD requirements were missed."
        )

    prompt = f"""You are evaluating the RETRIEVAL stage of a RAG pipeline for technical interview generation.
Do NOT evaluate any generated interview. Only evaluate what is shown below.

Company: {company}
Role: {role}

Job Description:
{job_description}

--- PIPELINE: {label} ---

{extraction_note}

Retrieved questions:
{questions_text}

{gap_note}

---

Score each criterion from 1 (very poor) to 5 (excellent). Be strict and cite specific evidence.

CRITERIA:

1. jd_relevance (1-5)
   Do the retrieved questions directly relate to the specific JD requirements?
   Score 1 if most are off-topic. Score 5 if most are directly relevant.

2. noise_ratio (1-5)  INVERTED: high score = low noise
   What fraction of retrieved questions are irrelevant or off-topic for this JD?
   Score 1 if nearly all are noise. Score 5 if nearly all are relevant.

3. question_completeness (1-5)
   Are the questions complete with problem statement, example, and constraints?
   Score 1 for bare stubs. Score 5 for fully formed questions with context.

4. jd_requirement_extraction (1-5)
   Did the pipeline explicitly extract and structure the JD key requirements?
   Score 1 if no extraction. Score 5 if extraction is precise and complete.

5. gap_awareness (1-5)
   Did the pipeline detect which JD requirements were NOT covered and report them?
   Score 1 if no gap detection. Score 5 if gaps are precisely identified.

Return ONLY valid JSON, no markdown:
{{
  "jd_relevance":              {{"score": 0, "reason": "cite specific questions"}},
  "noise_ratio":               {{"score": 0, "reason": "count noisy questions explicitly"}},
  "question_completeness":     {{"score": 0, "reason": "describe whether questions have full context"}},
  "jd_requirement_extraction": {{"score": 0, "reason": "describe what was extracted or not"}},
  "gap_awareness":             {{"score": 0, "reason": "describe gap detection quality"}}
}}"""

    raw = llm(prompt, model="llama3.3-70b")
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        print(f"  [Score {label}] Parse failed. Raw:\n{raw[:300]}")
        return {c: {"score": 0, "reason": "parse error"} for c in RETRIEVAL_CRITERIA}
    return parsed

# ---------------------------------------------------------------------------
# Run one test case
# ---------------------------------------------------------------------------

def run_test_case(tc):
    company        = tc["company"]
    role           = tc["role"]
    job_description = tc["job_description"]

    print(f"\n  Running Vector RAG...")
    vector_retrieved_df, vector_output = run_vector_rag_interview(
        company=company, role=role, job_description=job_description
    )

    print(f"  Running GraphRAG...")
    graph_clusters, graph_questions, graph_output, jd_requirements, missing_requirements = run_graphrag_interview(
        company=company, role=role, job_description=job_description
    )

    print(f"  Scoring Vector RAG retrieval...")
    vector_scores = score_retrieval_quality(
        label="Vector RAG",
        retrieved_questions=vector_retrieved_df["interview_question"].tolist(),
        job_description=job_description,
        company=company, role=role,
        jd_requirements=None,
        missing_requirements=None,
    )

    print(f"  Scoring GraphRAG retrieval...")
    graph_scores = score_retrieval_quality(
        label="GraphRAG",
        retrieved_questions=graph_questions,
        job_description=job_description,
        company=company, role=role,
        jd_requirements=jd_requirements,
        missing_requirements=missing_requirements,
    )

    return {
        "tc":                   tc,
        "vector_retrieved_df":  vector_retrieved_df,
        "graph_questions":      graph_questions,
        "graph_clusters":       graph_clusters,
        "jd_requirements":      jd_requirements,
        "missing_requirements": missing_requirements,
        "vector_scores":        vector_scores,
        "graph_scores":         graph_scores,
    }

# ---------------------------------------------------------------------------
# Print results for one test case
# ---------------------------------------------------------------------------

def print_case_result(result):
    tc  = result["tc"]
    vrs = result["vector_scores"]
    grs = result["graph_scores"]

    print("\n" + "=" * 72)
    print(f"TEST CASE: {tc['label']}")
    print("=" * 72)
    print(f"  {'Criterion':<30} {'Vector':>8} {'GraphRAG':>10}  Winner")
    print(f"  {'-'*30} {'-'*8} {'-'*10}  {'-'*10}")

    vr_total, gr_total = 0, 0
    for c in RETRIEVAL_CRITERIA:
        vs = (vrs.get(c) or {}).get("score", "?")
        gs = (grs.get(c) or {}).get("score", "?")
        if isinstance(vs, int) and isinstance(gs, int):
            winner = "Vector RAG" if vs > gs else ("GraphRAG" if gs > vs else "Tie")
            vr_total += vs
            gr_total += gs
        else:
            winner = "?"
        print(f"  {c:<30} {str(vs):>8} {str(gs):>10}  {winner}")

    overall = "Vector RAG" if vr_total > gr_total else ("GraphRAG" if gr_total > vr_total else "Tie")
    print(f"\n  {'TOTAL (out of 25)':<30} {vr_total:>8} {gr_total:>10}  {overall}")

    print("\n  Reasons:")
    for c in RETRIEVAL_CRITERIA:
        vs = (vrs.get(c) or {}).get("score", "?")
        gs = (grs.get(c) or {}).get("score", "?")
        vr = (vrs.get(c) or {}).get("reason", "")
        gr = (grs.get(c) or {}).get("reason", "")
        print(f"\n  [{c.upper()}]  Vector={vs}  GraphRAG={gs}")
        print(f"    Vector  : {vr}")
        print(f"    GraphRAG: {gr}")

    print("\n  Missing requirements (GraphRAG):")
    for m in result["missing_requirements"]:
        print(f"    • {m}")

    print("\n  Vector RAG retrieved questions:")
    for i, q in enumerate(result["vector_retrieved_df"]["interview_question"].tolist()[:5], 1):
        print(f"    {i}. {q[:90]}")

    print("\n  GraphRAG grounding questions:")
    for i, q in enumerate(result["graph_questions"][:5], 1):
        print(f"    {i}. {q.split(chr(10))[0][:90]}")

# ---------------------------------------------------------------------------
# Aggregate across all test cases
# ---------------------------------------------------------------------------

def print_aggregate_summary(all_results):
    print("\n" + "=" * 72)
    print("AGGREGATE SUMMARY — ACROSS ALL 3 TEST CASES")
    print("=" * 72)

    agg = {c: {"vector": [], "graph": []} for c in RETRIEVAL_CRITERIA}

    for r in all_results:
        for c in RETRIEVAL_CRITERIA:
            vs = (r["vector_scores"].get(c) or {}).get("score", None)
            gs = (r["graph_scores"].get(c) or {}).get("score", None)
            if isinstance(vs, int):
                agg[c]["vector"].append(vs)
            if isinstance(gs, int):
                agg[c]["graph"].append(gs)

    print(f"\n  {'Criterion':<30} {'Vector avg':>12} {'GraphRAG avg':>14}  Winner")
    print(f"  {'-'*30} {'-'*12} {'-'*14}  {'-'*10}")

    v_grand, g_grand = 0.0, 0.0
    for c in RETRIEVAL_CRITERIA:
        v_scores = agg[c]["vector"]
        g_scores = agg[c]["graph"]
        v_avg = round(sum(v_scores) / len(v_scores), 2) if v_scores else 0
        g_avg = round(sum(g_scores) / len(g_scores), 2) if g_scores else 0
        winner = "Vector RAG" if v_avg > g_avg else ("GraphRAG" if g_avg > v_avg else "Tie")
        v_grand += v_avg
        g_grand += g_avg
        print(f"  {c:<30} {str(v_avg):>12} {str(g_avg):>14}  {winner}")

    overall = "Vector RAG" if v_grand > g_grand else ("GraphRAG" if g_grand > v_grand else "Tie")
    print(f"\n  {'GRAND TOTAL (avg out of 25)':<30} {round(v_grand,2):>12} {round(g_grand,2):>14}  {overall}")

    print("\n  Per-case totals:")
    for r in all_results:
        vt = sum(
            (r["vector_scores"].get(c) or {}).get("score", 0)
            for c in RETRIEVAL_CRITERIA
            if isinstance((r["vector_scores"].get(c) or {}).get("score", 0), int)
        )
        gt = sum(
            (r["graph_scores"].get(c) or {}).get("score", 0)
            for c in RETRIEVAL_CRITERIA
            if isinstance((r["graph_scores"].get(c) or {}).get("score", 0), int)
        )
        winner = "Vector RAG" if vt > gt else ("GraphRAG" if gt > vt else "Tie")
        print(f"    {r['tc']['label']:<38} Vector={vt}  GraphRAG={gt}  → {winner}")

    # Criterion-level win counts
    print("\n  Criterion win counts across 3 cases:")
    win_counts = {"Vector RAG": 0, "GraphRAG": 0, "Tie": 0}
    for c in RETRIEVAL_CRITERIA:
        for r in all_results:
            vs = (r["vector_scores"].get(c) or {}).get("score", 0)
            gs = (r["graph_scores"].get(c) or {}).get("score", 0)
            if isinstance(vs, int) and isinstance(gs, int):
                if vs > gs:
                    win_counts["Vector RAG"] += 1
                elif gs > vs:
                    win_counts["GraphRAG"] += 1
                else:
                    win_counts["Tie"] += 1
    total_decisions = sum(win_counts.values())
    for k, v in win_counts.items():
        pct = round(100 * v / total_decisions) if total_decisions else 0
        print(f"    {k:<15}: {v}/{total_decisions} ({pct}%)")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = []

    for tc in TEST_CASES:
        print(f"\n{'='*72}")
        print(f"Running test case: {tc['label']}")
        print(f"{'='*72}")
        result = run_test_case(tc)
        all_results.append(result)
        print_case_result(result)

    print_aggregate_summary(all_results)