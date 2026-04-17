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
  * If correctness = 0 (completely wrong solution): time_complexity MUST be <= 3
  * If correctness >= 1 but code has inefficiencies: score normally
  * A completely wrong solution cannot have a valid time complexity
- space_complexity: is the space complexity optimal
  * If correctness = 0 (completely wrong solution): space_complexity MUST be <= 3
  * If correctness >= 1: score normally based on actual space used
- code_quality: clean code, meaningful names, handles edge cases
  * If correctness = 0: code_quality MUST be <= 3
  * If correctness >= 1 but has minor issues: score between 4-7
  * Only score 8+ if code is both correct AND clean

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
# SKILL GROUP DEFINITIONS
# OR-skills: demonstrating any one satisfies the requirement
# -------------------------------
LANGUAGE_GROUP = [
    "java", "python", "c++", "c#", "go", "golang",
    "javascript", "typescript", "kotlin", "swift", "rust", "scala"
]

CLOUD_GROUP = [
    "aws", "gcp", "azure", "cloud platforms", "cloud", "google cloud"
]

DB_GROUP = [
    "mysql", "postgresql", "mongodb", "dynamodb", "redis",
    "cassandra", "database", "databases", "sql", "nosql"
]

OR_SKILL_GROUPS = [LANGUAGE_GROUP, CLOUD_GROUP, DB_GROUP]

# Skills implicitly assessed through coding questions
CODING_IMPLICIT_SKILLS = [
    "data structures", "algorithms", "problem-solving", "problem solving",
    "analytical skills", "coding", "programming", "data structure",
    "algorithm", "computational thinking", "logical reasoning",
    "software engineering", "computer science fundamentals"
]

# Skills implicitly assessed through system design questions
SYSTEM_DESIGN_IMPLICIT_SKILLS = [
    "system design", "distributed systems", "scalability", "architecture",
    "backend", "scalable backend", "distributed", "microservices",
    "high availability", "fault tolerance"
]

# Skills implicitly assessed through behavioral questions
BEHAVIORAL_IMPLICIT_SKILLS = [
    "communication", "teamwork", "collaboration", "leadership",
    "problem-solving", "adaptability", "analytical skills",
    "interpersonal skills", "stakeholder management"
]

# Skills that need explicit demonstration in code — not implicitly covered
OOP_SKILLS = [
    "object-oriented programming", "oop", "object oriented",
    "design patterns", "inheritance", "encapsulation",
    "polymorphism", "abstraction", "solid principles"
]


def _normalize(text: str) -> str:
    return text.lower().strip()


def _is_or_skill(skill: str) -> bool:
    """Check if a skill belongs to an OR group."""
    s = _normalize(skill)
    for group in OR_SKILL_GROUPS:
        if any(s in g or g in s for g in group):
            return True
    return False


def _get_or_group(skill: str) -> list:
    """Get the OR group a skill belongs to."""
    s = _normalize(skill)
    for group in OR_SKILL_GROUPS:
        if any(s in g or g in s for g in group):
            return group
    return []


def _detect_not_assessed(
    jd_requirements: dict,
    eval_results: list,
    user_answers: list = None
) -> list:
    """
    Detect JD skills that were never assessed in the interview.
    Handles OR-skill groups, implicit skill coverage, and OOP detection.
    """
    # Check what question types were actually present
    question_types_assessed = set()
    for r in eval_results:
        qt = r.get("evaluation", {}).get("question_type", "")
        if qt:
            question_types_assessed.add(qt)

    # Collect all score keys from evaluation
    assessed_keys = set()
    for r in eval_results:
        scores = r.get("evaluation", {}).get("scores", {})
        for k in scores.keys():
            assessed_keys.add(_normalize(k.replace("_", " ")))

    # Add languages actually used by the user
    used_languages = set()
    if user_answers:
        for qa in user_answers:
            lang = qa.get("lang", "")
            if lang:
                used_languages.add(_normalize(lang))

    # Check if OOP was actually demonstrated in any code answer
    oop_demonstrated = False
    if user_answers:
        oop_signals = [
            "class ", "self.", "extends ", "implements ",
            "public class", "private ", "protected ",
            "def __init__", "super()", "override",
            "@override", "abstract", "interface "
        ]
        for qa in user_answers:
            answer = qa.get("answer", "")
            if any(signal in answer for signal in oop_signals):
                oop_demonstrated = True
                break

    # Collect all JD skills
    all_jd_skills = []
    for group in ["technical_skills", "system_topics",
                  "behavioral_traits", "priority_requirements"]:
        for skill in jd_requirements.get(group, []):
            all_jd_skills.append(_normalize(skill))

    not_assessed = []
    satisfied_or_groups = set()

    for skill in all_jd_skills:
        skill_n = _normalize(skill)

        # OOP — only satisfied if explicitly demonstrated in code
        if any(oop in skill_n or skill_n in oop
               for oop in OOP_SKILLS):
            if not oop_demonstrated:
                not_assessed.append(skill)
            continue

        # Implicit coverage via coding questions
        if "coding" in question_types_assessed or \
           "system_design" in question_types_assessed:
            if any(imp in skill_n or skill_n in imp
                   for imp in CODING_IMPLICIT_SKILLS):
                continue

        # Implicit coverage via system design questions
        if "system_design" in question_types_assessed:
            if any(imp in skill_n or skill_n in imp
                   for imp in SYSTEM_DESIGN_IMPLICIT_SKILLS):
                continue

        # Implicit coverage via behavioral questions
        if "behavioral" in question_types_assessed:
            if any(imp in skill_n or skill_n in imp
                   for imp in BEHAVIORAL_IMPLICIT_SKILLS):
                continue

        # Check if it's an OR skill
        if _is_or_skill(skill):
            group = tuple(_get_or_group(skill))
            if group in satisfied_or_groups:
                continue
            group_satisfied = any(
                any(g in ak or ak in g for ak in assessed_keys)
                or any(g in ul or ul in g for ul in used_languages)
                for g in group
            )
            if group_satisfied:
                satisfied_or_groups.add(group)
            else:
                not_assessed.append(skill)
                satisfied_or_groups.add(group)
        else:
            # AND skill — check direct match in score keys
            is_assessed = any(
                skill_n in ak or ak in skill_n
                for ak in assessed_keys
            )
            if not is_assessed:
                not_assessed.append(skill)

    return list(set(not_assessed))[:6]


def _extract_concepts_per_question(eval_results: list) -> list:
    """
    For each coding question, extract the concepts it tests
    and whether the user demonstrated them correctly.
    Returns a list of dicts per question with concepts and pass/fail.
    """
    question_concepts = []

    for r in eval_results:
        evaluation = r.get("evaluation", {})
        q_type = evaluation.get("question_type", "coding")

        # Only process coding questions
        if q_type != "coding":
            continue

        question = r.get("question", "")
        correctness = evaluation.get("scores", {}).get("correctness", 0)
        weaknesses = evaluation.get("weaknesses", [])
        optimized_approach = evaluation.get("optimized_approach", "")

        # Ask LLM to extract concepts this question tests
        prompt = f"""
You are analyzing a coding interview question to identify the core concepts it tests.

Question:
{question[:500]}

Candidate weakness feedback:
{chr(10).join(weaknesses[:3])}

List the 2-4 core CS concepts this question primarily tests.
Examples: stack, dynamic programming, two pointers, binary search,
graph traversal, recursion, hash map, sliding window, tree traversal,
string manipulation, greedy algorithm, backtracking

Return ONLY valid JSON. No explanation.

{{
  "concepts": ["concept1", "concept2"]
}}
"""
        try:
            raw = llm(prompt, model="llama3.3-70b")
            parsed = safe_json_parse(raw)
            concepts = parsed.get("concepts", []) if parsed else []
        except Exception:
            concepts = []

        # Determine if user passed or failed this question
        passed = correctness >= 7
        completely_wrong = correctness == 0

        question_concepts.append({
            "question": question[:80],
            "concepts": [c.lower().strip() for c in concepts],
            "correctness": correctness,
            "passed": passed,
            "completely_wrong": completely_wrong,
            "weaknesses": weaknesses
        })

    return question_concepts

def _geval_gap_scoring(
    eval_results: list,
    role: str,
    company: str,
    jd_requirements: dict
) -> list:
    """
    Score each gap using G-Eval style + concept-level analysis.

    Two layers:
    1. Score-key gaps — correctness, time_complexity etc averaged per question type
    2. Concept gaps — cross-question concept analysis with critical/medium/light rules:
       - Concept wrong in 2+ questions → CRITICAL
       - Concept wrong in 1 question, correct in another → MEDIUM
       - Concept wrong in 1 question, never tested again → MEDIUM
       - Minor mistake (score 1-6, not 0) → LIGHT
    """
    # ── LAYER 1: Score-key gaps ───────────────────────────────────
    topic_scores = {}
    for r in eval_results:
        evaluation = r.get("evaluation", {})
        q_type = evaluation.get("question_type", "coding")
        if q_type == "behavioral":
            continue
        scores = evaluation.get("scores", {})
        for k, v in scores.items():
            if q_type == "system_design":
                topic = f"system design — {k.replace('_', ' ').lower()}"
            else:
                topic = k.replace("_", " ").lower()
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(v or 0)

    avg_scores = {
        topic: round(sum(vals) / len(vals), 1)
        for topic, vals in topic_scores.items()
    }

    # Only score-key gaps below 7
    score_key_gaps = {k: v for k, v in avg_scores.items() if v < 7}

    # ── LAYER 2: Concept-level gaps ───────────────────────────────
    question_concepts = _extract_concepts_per_question(eval_results)

    # Map concept → list of (passed, completely_wrong) across questions
    concept_results = {}
    for qc in question_concepts:
        for concept in qc["concepts"]:
            if concept not in concept_results:
                concept_results[concept] = []
            concept_results[concept].append({
                "passed":           qc["passed"],
                "completely_wrong": qc["completely_wrong"],
                "correctness":      qc["correctness"]
            })

    # Determine gap level per concept
    concept_gaps = {}
    for concept, results in concept_results.items():
        failed = [r for r in results if not r["passed"]]
        passed = [r for r in results if r["passed"]]
        completely_wrong = [r for r in results if r["completely_wrong"]]

        if not failed:
            continue  # No gap for this concept

        if len(completely_wrong) >= 2:
            # Wrong in 2+ questions → CRITICAL
            level = "critical"
        elif len(completely_wrong) == 1 and len(passed) == 0:
            # Completely wrong in 1 question, never demonstrated correctly → MEDIUM
            level = "medium"
        elif len(completely_wrong) == 1 and len(passed) > 0:
            # Completely wrong in 1 but correct in another → MEDIUM
            # (showed partial understanding)
            level = "medium"
        elif len(failed) >= 2:
            # Partial failures in 2+ questions → MEDIUM
            level = "medium"
        else:
            # Minor failure in 1 question → LIGHT
            level = "light"

        avg_correctness = round(
            sum(r["correctness"] for r in results) / len(results), 1
        )
        concept_gaps[concept] = {
            "avg_score":  avg_correctness,
            "level":      level,
            "times_failed":   len(failed),
            "times_passed":   len(passed),
            "times_zero":     len(completely_wrong)
        }

    # ── COMBINE both layers ───────────────────────────────────────
    # Ask LLM to score importance of all gaps
    all_gap_topics = list(score_key_gaps.keys()) + list(concept_gaps.keys())
    all_gap_topics = list(set(all_gap_topics))  # deduplicate

    if not all_gap_topics:
        return []

    tech     = ", ".join(jd_requirements.get("technical_skills", []))
    systems  = ", ".join(jd_requirements.get("system_topics", []))
    priority = ", ".join(jd_requirements.get("priority_requirements", []))

    gaps_text = "\n".join(f"- {t}" for t in all_gap_topics)

    prompt = f"""
You are evaluating skill gaps for a {role} interview at {company}.

Job requires: {tech}, {systems}
Priority requirements: {priority}

Identified gaps:
{gaps_text}

For each gap, score its IMPORTANCE for this specific role from 0.0 to 1.0.
0.0 = not important for this role
1.0 = critical for this role

Return ONLY valid JSON. No explanation.

{{
  "importance_scores": {{
    "gap_topic": 0.0
  }}
}}
"""
    try:
        raw = llm(prompt, model="llama3.3-70b")
        parsed = safe_json_parse(raw)
        importance_scores = parsed.get("importance_scores", {}) if parsed else {}
    except Exception:
        importance_scores = {}

    scored_gaps = []
    seen_topics = set()

    # Process concept gaps first — they are more specific
    for concept, data in concept_gaps.items():
        if concept in seen_topics:
            continue
        seen_topics.add(concept)

        avg_score  = data["avg_score"]
        level      = data["level"]
        importance = float(importance_scores.get(concept, 0.5))
        severity   = round((10 - avg_score) / 10, 2)
        priority_score = round(severity * importance, 3)

        # Enforce level minimums based on concept analysis
        if level == "critical":
            priority_score = max(priority_score, 0.7)
            color = "🔴"
        elif level == "medium":
            priority_score = max(priority_score, 0.35)
            color = "🟠"
        else:
            color = "🟡"

        scored_gaps.append({
            "topic":          concept,
            "avg_score":      avg_score,
            "severity":       severity,
            "importance":     importance,
            "priority_score": priority_score,
            "level":          level,
            "color":          color,
            "source":         "concept",
            "times_failed":   data["times_failed"],
            "times_zero":     data["times_zero"]
        })

    # Process score-key gaps (system design etc) not already covered
    for topic, avg_score in score_key_gaps.items():
        if topic in seen_topics:
            continue
        seen_topics.add(topic)

        importance     = float(importance_scores.get(topic, 0.5))
        severity       = round((10 - avg_score) / 10, 2)
        priority_score = round(severity * importance, 3)

        if priority_score >= 0.6:
            level = "critical"
            color = "🔴"
        elif priority_score >= 0.35:
            level = "medium"
            color = "🟠"
        else:
            level = "light"
            color = "🟡"

        scored_gaps.append({
            "topic":          topic,
            "avg_score":      avg_score,
            "severity":       severity,
            "importance":     importance,
            "priority_score": priority_score,
            "level":          level,
            "color":          color,
            "source":         "score_key",
            "times_failed":   0,
            "times_zero":     0
        })

    return sorted(scored_gaps, key=lambda x: x["priority_score"], reverse=True)


def _build_spaced_repetition_schedule(
    scored_gaps: list,
    not_assessed: list,
    role: str,
    company: str,
    jd_requirements: dict
) -> list:
    """
    Build a 14-day spaced repetition schedule.
    Critical gaps appear multiple times.
    Not-assessed skills fill remaining days.
    """
    critical = [g for g in scored_gaps if g["level"] == "critical"]
    medium   = [g for g in scored_gaps if g["level"] == "medium"]
    light    = [g for g in scored_gaps if g["level"] == "light"]

    # Spaced repetition slots
    # Critical gaps: Days 1, 4, 8, 13
    # Medium gaps:   Days 2, 6, 11
    # Light gaps:    Days 3, 9
    # Not assessed:  Days 5, 7, 10, 12, 14
    schedule_template = [
        {"day": 1,  "source": "critical", "idx": 0},
        {"day": 2,  "source": "medium",   "idx": 0},
        {"day": 3,  "source": "light",    "idx": 0},
        {"day": 4,  "source": "critical", "idx": 0},  # revisit
        {"day": 5,  "source": "not_assessed", "idx": 0},
        {"day": 6,  "source": "medium",   "idx": 1},
        {"day": 7,  "source": "not_assessed", "idx": 1},
        {"day": 8,  "source": "critical", "idx": 1},  # revisit
        {"day": 9,  "source": "light",    "idx": 1},
        {"day": 10, "source": "not_assessed", "idx": 2},
        {"day": 11, "source": "medium",   "idx": 2},
        {"day": 12, "source": "not_assessed", "idx": 3},
        {"day": 13, "source": "critical", "idx": 2},  # revisit
        {"day": 14, "source": "not_assessed", "idx": 4},
    ]

    source_map = {
        "critical":     critical,
        "medium":       medium,
        "light":        light,
        "not_assessed": [{"topic": s, "level": "not_assessed",
                          "color": "⚪"} for s in not_assessed]
    }

    days = []
    for slot in schedule_template:
        source_list = source_map[slot["source"]]
        idx = slot["idx"] % max(len(source_list), 1) if source_list else None

        if not source_list or idx is None:
            # Fall back to any available gap
            fallback = critical or medium or light
            if fallback:
                gap = fallback[slot["day"] % len(fallback)]
            else:
                gap = {"topic": "general review", "level": "light", "color": "🟡"}
        else:
            gap = source_list[idx]

        is_revisit = slot["source"] == "critical" and slot["day"] > 1

        days.append({
            "day": slot["day"],
            "topic": gap["topic"],
            "level": gap.get("level", "light"),
            "color": gap.get("color", "🟡"),
            "is_revisit": is_revisit,
            "avg_score": gap.get("avg_score", None),
            "priority_score": gap.get("priority_score", None)
        })

    return days


def _enrich_day_content(
    days: list,
    role: str,
    company: str,
    jd_requirements: dict
) -> list:
    """
    Use LLM to add focus, why_it_matters, and practice
    for each day in the schedule.
    """
    days_text = "\n".join(
        f"Day {d['day']} — {d['topic']} "
        f"({'revisit' if d['is_revisit'] else d['level']})"
        for d in days
    )

    tech = ", ".join(jd_requirements.get("technical_skills", []))
    systems = ", ".join(jd_requirements.get("system_topics", []))

    prompt = f"""
You are creating a personalized interview preparation plan for a {role} role at {company}.

Job requires: {tech}, {systems}

14-day schedule:
{days_text}

For each day provide:
- why_it_matters: 1 sentence specific to {company} and {role}
- focus: 2-3 specific concepts to study (not generic)
- practice: 1 concrete actionable task (specific problem type, not "write an essay")

Return ONLY valid JSON. No explanation.

{{
  "days": [
    {{
      "day": 1,
      "why_it_matters": "",
      "focus": "",
      "practice": ""
    }}
  ]
}}
"""
    try:
        raw = llm(prompt, model="llama3.3-70b")
        parsed = safe_json_parse(raw)
        enriched_days = parsed.get("days", []) if parsed else []
        enriched_map = {d["day"]: d for d in enriched_days}
    except Exception:
        enriched_map = {}

    for d in days:
        enriched = enriched_map.get(d["day"], {})
        d["why_it_matters"] = enriched.get("why_it_matters", "")
        d["focus"] = enriched.get("focus", "")
        d["practice"] = enriched.get("practice", "")

    return days


def generate_learning_path(
    results: list,
    role: str,
    company: str,
    job_description: str = "",
    jd_requirements: dict = None,
    user_answers: list = None
) -> dict:
    """
    Generate a structured personalized learning plan.
    Returns a dict (not a string) for proper UI rendering.

    Steps:
    1. G-Eval gap scoring — severity × importance
    2. Detect not-assessed JD skills (handles OR groups)
    3. Build spaced repetition schedule
    4. Enrich each day with specific content
    """
    if jd_requirements is None:
        jd_requirements = {}

    logger.info("Generating learning path with G-Eval scoring...")

    # Step 1 — Score gaps
    scored_gaps = _geval_gap_scoring(results, role, company, jd_requirements)
    logger.info(f"Scored {len(scored_gaps)} gaps.")

    # Step 2 — Detect not-assessed skills
    not_assessed = _detect_not_assessed(jd_requirements, results, user_answers)
    logger.info(f"Detected {len(not_assessed)} not-assessed skills.")

    # Step 3 — Build spaced repetition schedule
    days = _build_spaced_repetition_schedule(
        scored_gaps, not_assessed, role, company, jd_requirements
    )

    # Step 4 — Enrich with specific content
    days = _enrich_day_content(days, role, company, jd_requirements)

    return {
        "role": role,
        "company": company,
        "scored_gaps": scored_gaps,
        "not_assessed": not_assessed,
        "days": days
    }


# -------------------------------
# SINGLE QUESTION EVALUATION
# (for Quick Practice mode)
# -------------------------------
def evaluate_single_answer(question: str, answer: str, q_type: str) -> dict:
    """
    Evaluate a single answer for Quick Practice mode.
    No ideal answer generation — lightweight and fast.
    No session saving. No learning plan.
    """
    if not isinstance(answer, str) or not answer.strip():
        return {
            "scores": {},
            "is_optimized": False,
            "strengths": [],
            "weaknesses": ["No answer provided."],
            "optimized_approach": "",
            "question_type": q_type
        }

    # For coding — run test cases if available
    test_results = None

    # Generate ideal answer for context
    ideal = {
        "ideal_answer": "",
        "key_concepts": [],
        "common_mistakes": []
    }

    if q_type == "coding":
        prompt = _coding_prompt(question, answer, ideal, test_results)
    elif q_type == "system_design":
        prompt = _system_design_prompt(question, answer, ideal)
    else:
        prompt = _behavioral_prompt(question, answer, ideal)

    default = _default_scores(q_type)

    # Single model for quick practice — no need for consensus
    try:
        response = llm(prompt)
        parsed = safe_json_parse(response)
        if not parsed or "scores" not in parsed:
            parsed = repair_json_with_llm(response)
    except Exception as e:
        logger.warning(f"evaluate_single_answer failed: {e}")
        parsed = {}

    if not isinstance(parsed, dict) or "scores" not in parsed:
        return {
            "scores": default,
            "is_optimized": False,
            "strengths": [],
            "weaknesses": ["Could not evaluate answer. Please try again."],
            "optimized_approach": "",
            "question_type": q_type
        }

    scores = parsed.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}

    is_optimized = parsed.get("is_optimized", _check_optimized(q_type, scores))

    return {
        "scores": {k: scores.get(k, default[k]) for k in default},
        "is_optimized": bool(is_optimized),
        "strengths": parsed.get("strengths", []),
        "weaknesses": parsed.get("weaknesses", []),
        "optimized_approach": parsed.get("optimized_approach", ""),
        "question_type": q_type
    }