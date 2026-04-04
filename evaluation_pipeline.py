import json
from graphrag_pipeline import llm


# -------------------------------
# SAFE JSON PARSER (ROBUST)
# -------------------------------
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        if isinstance(text, str):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except:
                    pass
    return {}  # always dict fallback


# -------------------------------
# IDEAL ANSWER GENERATION
# -------------------------------
def generate_ideal_answer(question, role, company):
    prompt = f"""
You are an expert interviewer.

Question: {question}
Role: {role}
Company: {company}

Generate:
- Ideal answer
- Key concepts
- Common mistakes

Return STRICT JSON ONLY.

{{
  "ideal_answer": "",
  "key_concepts": [],
  "common_mistakes": []
}}
"""

    response = llm(prompt + "\nONLY RETURN JSON.")

    parsed = safe_json_parse(response)

    if not isinstance(parsed, dict):
        parsed = {}

    return {
        "ideal_answer": parsed.get("ideal_answer", ""),
        "key_concepts": parsed.get("key_concepts", []),
        "common_mistakes": parsed.get("common_mistakes", [])
    }


# -------------------------------
# ANSWER EVALUATION (FULL SAFE)
# -------------------------------
def evaluate_answer(question, user_answer, ideal_data):
    if not isinstance(ideal_data, dict):
        ideal_data = {}

    prompt = f"""
You are an expert technical interviewer.

Evaluate the candidate answer.

Question:
{question}

User Answer:
{user_answer}

Ideal Answer:
{ideal_data.get('ideal_answer', '')}

Key Concepts:
{ideal_data.get('key_concepts', [])}

----------------------------------

SCORING RULES:

- Accuracy (0–10)
- Depth (0–10)
- Communication (0–10)
- Completeness (0–10)

IMPORTANT:
- If correct → accuracy >= 8
- DO NOT return all 5 scores
- Be realistic

----------------------------------

Return STRICT JSON ONLY:

{{
  "scores": {{
    "accuracy": 0,
    "depth": 0,
    "communication": 0,
    "completeness": 0
  }},
  "strengths": [],
  "weaknesses": [],
  "missing_concepts": []
}}
"""

    response = llm(prompt + "\nONLY RETURN JSON.")

    # 🔥 DEBUG (optional)
    # print("RAW:", response)

    parsed = safe_json_parse(response)

    # 🚨 CRITICAL FIX: ensure dict
    if not isinstance(parsed, dict):
        parsed = {}

    scores = parsed.get("scores")

    # 🚨 FINAL SAFETY CHECK
    if not isinstance(scores, dict):
        return {
            "scores": {
                "accuracy": 5,
                "depth": 5,
                "communication": 5,
                "completeness": 5
            },
            "strengths": [],
            "weaknesses": ["LLM returned invalid format"],
            "missing_concepts": []
        }

    return {
        "scores": {
            "accuracy": scores.get("accuracy", 5),
            "depth": scores.get("depth", 5),
            "communication": scores.get("communication", 5),
            "completeness": scores.get("completeness", 5),
        },
        "strengths": parsed.get("strengths", []),
        "weaknesses": parsed.get("weaknesses", []),
        "missing_concepts": parsed.get("missing_concepts", [])
    }


# -------------------------------
# INTERVIEW EVALUATION
# -------------------------------
def evaluate_interview(company, role, qa_pairs):
    results = []

    for qa in qa_pairs:
        answer = qa.get("answer", "")

        if not isinstance(answer, str) or not answer.strip():
            continue

        question = qa.get("question", "")

        ideal = generate_ideal_answer(question, role, company)

        evaluation = evaluate_answer(
            question,
            answer,
            ideal
        )

        results.append({
            "question": question,
            "evaluation": evaluation,
            "ideal": ideal
        })

    return results


# -------------------------------
# LEARNING PATH (SMART)
# -------------------------------
def generate_learning_path(results, role, company):
    missing = []

    for r in results:
        missing.extend(
            r.get("evaluation", {}).get("missing_concepts", [])
        )

    missing = list(set(missing))

    prompt = f"""
You are a career coach.

Weak areas:
{missing}

Role: {role}
Company: {company}

RULES:
- Focus ONLY on weak areas
- NO generic topics
- Be specific (DFS, Binary Search, Caching, etc.)

Generate a 7-day plan:

Day 1:
- Topic:
- Focus:
- Practice:

Keep it short and useful.
"""

    response = llm(prompt)

    return response if response else "No learning plan generated."