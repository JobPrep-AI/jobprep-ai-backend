import json
import re
from snowflake_utils import llm
from graphrag_pipeline import parse_json_response
import logging

logger = logging.getLogger(__name__)

# -------------------------------
# SAFE JSON PARSER
# -------------------------------
def safe_json_parse(text):
    if not isinstance(text, str):
        return {}
    result = parse_json_response(text)
    return result if isinstance(result, dict) else {}


def repair_json_with_llm(bad_text):
    logger.warning("repair_json_with_llm: attempting single LLM repair...")
    repair_prompt = f"""
Fix this into VALID JSON.

Return ONLY JSON.
No explanation.

Input:
{bad_text}
"""
    fixed = llm(repair_prompt)
    # Use basic JSON parsing only — no further LLM calls
    if not isinstance(fixed, str):
        return {}
    try:
        start = fixed.find("{")
        end = fixed.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(fixed[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass
    return {}


# -------------------------------
# IDEAL ANSWER GENERATION
# -------------------------------
def generate_ideal_answer(question, role, company):
    prompt = f"""
You are an expert interviewer at {company} for the role of {role}.

Question: {question}

Generate an ideal answer with key concepts and common mistakes.

Return ONLY raw JSON with NO explanation, NO markdown, NO code blocks:
{{
  "ideal_answer": "string",
  "key_concepts": ["string"],
  "common_mistakes": ["string"]
}}
"""
    response = llm(prompt)
    parsed = safe_json_parse(response)

    return {
        "ideal_answer": parsed.get("ideal_answer", ""),
        "key_concepts": parsed.get("key_concepts", []),
        "common_mistakes": parsed.get("common_mistakes", [])
    }


# -------------------------------
# QUESTION TYPE DETECTION
# -------------------------------
def _detect_question_type(question: str) -> str:
    q = question.lower()

    behavioral_signals = [
        "tell me about", "describe a time", "how do you handle",
        "give an example", "what would you do", "have you ever",
        "walk me through", "share an experience", "how have you",
        "what is your approach to working", "describe your"
    ]
    system_design_signals = [
        "design a system", "design an", "architect a", "architect an",
        "how would you build a", "how would you design a",
        "design the backend", "url shortener", "chat system",
        "notification system", "ride sharing", "news feed",
        "design twitter", "design uber", "design netflix",
        "design youtube", "design instagram", "design whatsapp",
        "key discussion points", "non-functional requirements",
        "scalability of", "distributed system for"
    ]
    coding_signals = [
        "given an array", "given a string", "given a list",
        "given a binary tree", "given a linked list",
        "implement a function", "write a function", "write code",
        "find the", "return the", "solve the following",
        "leetcode", "algorithm", "data structure",
        "time complexity", "space complexity",
        "two pointers", "sliding window", "dynamic programming",
        "depth first", "breadth first", "binary search"
    ]

    # Score each type by counting matching signals
    behavioral_score = sum(1 for s in behavioral_signals if s in q)
    system_score = sum(1 for s in system_design_signals if s in q)
    coding_score = sum(1 for s in coding_signals if s in q)

    # Pick the highest scoring type
    scores = {
        "behavioral": behavioral_score,
        "system_design": system_score,
        "coding": coding_score
    }

    best = max(scores, key=scores.get)

    # Only classify as non-coding if it has at least 1 clear signal
    # Otherwise default to coding since that's most common
    if scores[best] == 0:
        return "coding"

    return best


# -------------------------------
# PROMPTS BY QUESTION TYPE
# -------------------------------
def _coding_prompt(question, user_answer, ideal_data, test_results=None):
    # Build test execution summary
    test_summary = ""
    if test_results:
        passed = sum(1 for r in test_results if r.get("passed"))
        total = len(test_results)
        lines = [f"Test Execution Results: {passed}/{total} passed"]
        for r in test_results:
            status = "PASSED" if r.get("passed") else "FAILED"
            lines.append(
                f"  - Test {r.get('case')}: Input={r.get('input')} "
                f"| Expected={r.get('expected')} "
                f"| Got={r.get('actual')} "
                f"| {status}"
            )
        test_summary = "\n".join(lines)
    else:
        test_summary = "Test Execution Results: Not available"

    return f"""
You are an expert technical interviewer evaluating a coding answer.

Question:
{question}

Candidate Answer:
{user_answer}

Ideal Answer:
{ideal_data.get('ideal_answer', '')}

ACTUAL TEST EXECUTION RESULTS (trust these over your own analysis):
{test_summary}

Score each dimension from 0 to 10:
- correctness: base this STRICTLY on test execution results above
  * 3/3 passed = 9-10
  * 2/3 passed = 6-7
  * 1/3 passed = 3-4
  * 0/3 passed = 0-2
- time_complexity: is the time complexity optimal for this problem
- space_complexity: is the space complexity optimal
- code_quality: clean code, meaningful names, handles edge cases

If correctness >= 8 AND time_complexity >= 7:
- set is_optimized to true
- strengths, weaknesses, optimized_approach can be empty

If NOT optimal:
- set is_optimized to false
- give specific strengths and weaknesses
- in optimized_approach briefly explain the best approach to solve this (algorithm, data structure, complexity)

Return ONLY this JSON. No explanation. No markdown.

{{
  "scores": {{
    "correctness": 0,
    "time_complexity": 0,
    "space_complexity": 0,
    "code_quality": 0
  }},
  "is_optimized": false,
  "strengths": ["string"],
  "weaknesses": ["string"],
  "optimized_approach": ""
}}
"""


def _system_design_prompt(question, user_answer, ideal_data):
    return f"""
You are an expert system design interviewer evaluating a system design answer.

Question:
{question}

Candidate Answer:
{user_answer}

Ideal Answer:
{ideal_data.get('ideal_answer', '')}

Score each dimension from 0 to 10:
- scalability: does the design handle scale, load, and growth effectively
- completeness: are all key components covered (API, DB, cache, queues, etc.)
- trade_offs: does the answer discuss trade-offs and justify design choices
- clarity: is the design clearly explained and structured

If scalability >= 7 AND completeness >= 7:
- set is_optimized to true
- strengths, weaknesses, optimized_approach can be empty

If NOT optimal:
- set is_optimized to false
- give specific strengths and weaknesses
- in optimized_approach briefly describe the ideal system design covering components, scaling strategy, and key trade-offs

Return ONLY this JSON. No explanation. No markdown.

{{
  "scores": {{
    "scalability": 0,
    "completeness": 0,
    "trade_offs": 0,
    "clarity": 0
  }},
  "is_optimized": false,
  "strengths": ["string"],
  "weaknesses": ["string"],
  "optimized_approach": ""
}}
"""


def _behavioral_prompt(question, user_answer, ideal_data):
    return f"""
You are an expert interviewer evaluating a behavioral answer.

Question:
{question}

Candidate Answer:
{user_answer}

Score each dimension from 0 to 10:
- relevance: how well the answer addresses the question
- clarity: how clearly and structurally the answer is communicated

If relevance >= 8 AND clarity >= 7:
- set is_optimized to true
- strengths, weaknesses, optimized_approach can be empty

If NOT optimal:
- set is_optimized to false
- give specific strengths and weaknesses
- in optimized_approach briefly describe what a strong answer to this question would look like

Return ONLY this JSON. No explanation. No markdown.

{{
  "scores": {{
    "relevance": 0,
    "clarity": 0
  }},
  "is_optimized": false,
  "strengths": ["string"],
  "weaknesses": ["string"],
  "optimized_approach": ""
}}
"""


def _default_scores(q_type):
    if q_type == "behavioral":
        return {"relevance": 5, "clarity": 5}
    if q_type == "system_design":
        return {"scalability": 5, "completeness": 5, "trade_offs": 5, "clarity": 5}
    return {"correctness": 5, "time_complexity": 5, "space_complexity": 5, "code_quality": 5}


def _check_optimized(q_type, scores):
    if q_type == "coding":
        return scores.get("correctness", 0) >= 8 and scores.get("time_complexity", 0) >= 7
    if q_type == "system_design":
        return scores.get("scalability", 0) >= 7 and scores.get("completeness", 0) >= 7
    if q_type == "behavioral":
        return scores.get("relevance", 0) >= 8 and scores.get("clarity", 0) >= 7
    return False


# -------------------------------
# ANSWER EVALUATION
# -------------------------------
def evaluate_answer(question, user_answer, ideal_data, test_results=None):
    if not isinstance(ideal_data, dict):
        ideal_data = {}

    q_type = _detect_question_type(question)

    if q_type == "behavioral":
        prompt = _behavioral_prompt(question, user_answer, ideal_data)
    elif q_type == "system_design":
        prompt = _system_design_prompt(question, user_answer, ideal_data)
    else:
        prompt = _coding_prompt(question, user_answer, ideal_data, test_results)

    response = llm(prompt)
    parsed = safe_json_parse(response)

    if not parsed or "scores" not in parsed:
        parsed = repair_json_with_llm(response)

    if not isinstance(parsed, dict) or "scores" not in parsed:
        return {
            "scores": _default_scores(q_type),
            "is_optimized": False,
            "strengths": [],
            "weaknesses": ["LLM returned invalid format"],
            "optimized_approach": "",
            "question_type": q_type
        }

    scores = parsed.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}

    is_optimized = parsed.get("is_optimized", _check_optimized(q_type, scores))
    default = _default_scores(q_type)

    return {
        "scores": {k: scores.get(k, default[k]) for k in default},
        "is_optimized": bool(is_optimized),
        "strengths": parsed.get("strengths", []),
        "weaknesses": parsed.get("weaknesses", []),
        "optimized_approach": parsed.get("optimized_approach", ""),
        "question_type": q_type
    }


# -------------------------------
# INTERVIEW EVALUATION
# -------------------------------
def evaluate_interview(company, role, qa_pairs):
    from code_executor import run_test_cases, reverify_with_user_code
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Filter out empty answers first
    valid_pairs = [
        qa for qa in qa_pairs
        if isinstance(qa.get("answer", ""), str) and qa.get("answer", "").strip()
    ]

    if not valid_pairs:
        return []

    def evaluate_single(qa):
        """Evaluate a single question-answer pair."""
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        lang = qa.get("lang", "python")
        test_cases = qa.get("test_cases", [])

        try:
            ideal = generate_ideal_answer(question, role, company)
            q_type = _detect_question_type(question)

            # Run code against test cases for coding questions
            test_results = None
            if q_type == "coding" and test_cases and answer.strip():
                try:
                    corrected = reverify_with_user_code(answer, lang, test_cases)
                    test_results = run_test_cases(answer, lang, corrected)
                    passed = sum(1 for r in test_results if r.get("passed"))
                    logger.info(
                        f"Test execution for Q: {question[:60]}... "
                        f"Passed: {passed}/{len(test_results)}"
                    )
                except Exception as e:
                    logger.warning(f"Test execution failed: {e}")
                    test_results = None

            evaluation = evaluate_answer(question, answer, ideal, test_results)

            return {
                "question": question,
                "evaluation": evaluation,
                "ideal": ideal,
                "test_results": test_results,
                "_order": qa.get("_order", 0)
            }

        except Exception as e:
            logger.error(f"evaluate_single failed for Q: {question[:60]}... Error: {e}")
            q_type = _detect_question_type(question)
            return {
                "question": question,
                "evaluation": {
                    "scores": _default_scores(q_type),
                    "is_optimized": False,
                    "strengths": [],
                    "weaknesses": [f"Evaluation failed: {str(e)}"],
                    "optimized_approach": "",
                    "question_type": q_type
                },
                "ideal": {},
                "test_results": None,
                "_order": qa.get("_order", 0)
            }

    # Tag each qa with its original order
    for idx, qa in enumerate(valid_pairs):
        qa["_order"] = idx

    results = [None] * len(valid_pairs)

    # Run all evaluations in parallel
    # max_workers=3 to avoid overwhelming Snowflake Cortex
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_idx = {
            executor.submit(evaluate_single, qa): qa.get("_order", i)
            for i, qa in enumerate(valid_pairs)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[result.get("_order", idx)] = result
            except Exception as e:
                logger.error(f"Future failed for index {idx}: {e}")

    # Filter out any None results (failed futures)
    results = [r for r in results if r is not None]

    # Clean up _order key from results
    for r in results:
        r.pop("_order", None)

    return results


# -------------------------------
# HINT GENERATION
# -------------------------------
def generate_hint(question, level):
    instructions = {
        1: "Give only a vague directional hint. Do NOT name any algorithm or data structure. Just point the candidate in the right direction.",
        2: "Name the specific algorithm or data structure needed. Do NOT show any code or pseudocode.",
        3: "Provide a clear step-by-step pseudocode approach. Do NOT write actual runnable code.",
    }
    prompt = f"""
You are a technical interviewer giving a structured hint to a candidate who is stuck.

Question:
{question}

Hint Level {level}/3 instruction:
{instructions.get(level, instructions[1])}

Return ONLY the hint text. No preamble. No label. No extra explanation.
"""
    return llm(prompt)


# -------------------------------
# LEARNING PATH
# -------------------------------
def generate_learning_path(results, role, company, job_description="", jd_requirements=None):

    # --- Collect performance data from evaluation ---
    weak_areas = []
    strong_areas = []
    per_question_summary = []

    for r in results:
        evaluation = r.get("evaluation", {})
        q_type = evaluation.get("question_type", "coding")

        if q_type == "behavioral":
            continue

        scores = evaluation.get("scores", {})
        is_optimized = evaluation.get("is_optimized", False)
        question = r.get("question", "")[:120]

        low = [k.replace("_", " ") for k, v in scores.items() if v < 7]
        high = [k.replace("_", " ") for k, v in scores.items() if v >= 8]
        weaknesses = evaluation.get("weaknesses", [])

        weak_areas.extend(low)
        weak_areas.extend(weaknesses)
        strong_areas.extend(high)

        per_question_summary.append({
            "question": question,
            "type": q_type,
            "is_optimized": is_optimized,
            "low_scores": low,
            "weaknesses": weaknesses
        })

    weak_areas = list(set(weak_areas))
    strong_areas = list(set(strong_areas))

    # --- JD requirements summary ---
    jd_tech = []
    jd_systems = []
    jd_behavioral = []
    if isinstance(jd_requirements, dict):
        jd_tech = jd_requirements.get("technical_skills", [])
        jd_systems = jd_requirements.get("system_topics", [])
        jd_behavioral = jd_requirements.get("behavioral_traits", [])

    prompt = f"""
You are an expert technical career coach helping a candidate prepare for a {role} role at {company}.

---

JOB DESCRIPTION:
{job_description or "Not provided"}

---

WHAT THE JOB REQUIRES:
- Technical Skills: {jd_tech}
- System Topics: {jd_systems}
- Behavioral Traits: {jd_behavioral}

---

CANDIDATE PERFORMANCE SUMMARY:
Strong areas (scored >= 8): {strong_areas if strong_areas else "None identified"}
Weak areas (scored < 7 or flagged): {weak_areas if weak_areas else "None identified"}

Per-question breakdown:
{json.dumps(per_question_summary, indent=2)}

---

Your task:

SECTION 1 — GAP ANALYSIS
- List what the job requires
- List what the candidate demonstrated well
- List the specific gaps (things required by JD that the candidate is weak in)
- Be specific and honest

SECTION 2 — PERSONALIZED LEARNING PLAN (14 days)
- Cover ONLY the gaps identified
- Each day must be specific: exact topic, why it matters for this role, and a concrete practice task
- Prioritize by importance to the role
- Mix coding, system design, and concepts as needed

Format EXACTLY as:

## Gap Analysis

### What This Role Requires
- ...

### What You Demonstrated Well
- ...

### Your Gaps
- ...

## 14-Day Learning Plan

### Day 1 — <Topic>
- **Why it matters:** ...
- **Focus:** ...
- **Practice:** ...

### Day 2 — <Topic>
...

Return plain text with markdown formatting only. No JSON.
"""

    response = llm(prompt)

    if isinstance(response, str):
        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            try:
                response = json.loads(response)
            except Exception:
                pass
        response = response.replace("\\n", "\n")

    return response if response else "No learning plan generated."