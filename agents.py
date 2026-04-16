"""
agents.py
---------
Multi-Agent Pipeline for JobPrep AI.

Agents:
1. JDAnalyzerAgent      — extracts requirements from job description
2. GraphRetrievalAgent  — retrieves relevant question clusters
3. QuestionGeneratorAgent — generates the mock interview
4. ReflectionAgent      — reviews and validates the generated interview
5. EvaluatorAgent       — scores user answers
6. LearningPathAgent    — generates personalized study plan
"""

import json
import logging
from guardrails import guardrails
from dataclasses import dataclass, field
from typing import Optional
from user_profile import load_weak_areas, save_user_session

from snowflake_utils import llm

logger = logging.getLogger(__name__)


# -------------------------------
# AGENT STATE
# Shared context passed between agents
# -------------------------------
@dataclass
class AgentState:
    # Inputs
    company: str = ""
    role: str = ""
    job_description: str = ""
    user_id: str = ""

    # JD Analyzer outputs
    jd_requirements: dict = field(default_factory=dict)

    # Graph Retrieval outputs
    top_clusters: object = None
    selected_questions: list = field(default_factory=list)
    missing_requirements: list = field(default_factory=list)

    # Question Generator outputs
    interview_raw: str = ""
    interview_parsed: dict = field(default_factory=dict)

    # Reflection outputs
    reflection_passed: bool = False
    reflection_feedback: str = ""
    reflection_attempts: int = 0

    # Evaluator outputs
    eval_results: list = field(default_factory=list)

    # Learning Path outputs
    learning_plan: str = ""

    # Pipeline metadata
    errors: list = field(default_factory=list)
    agent_logs: list = field(default_factory=list)

    def log(self, agent: str, message: str):
        entry = f"[{agent}] {message}"
        self.agent_logs.append(entry)
        logger.info(entry)


# -------------------------------
# BASE AGENT
# -------------------------------
class BaseAgent:
    name = "BaseAgent"

    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError


# -------------------------------
# AGENT 0 — INPUT VALIDATOR
# Runs before anything else
# -------------------------------
class InputValidatorAgent(BaseAgent):
    name = "InputValidatorAgent"

    def run(self, state: AgentState) -> AgentState:
        state.log(self.name, "Validating inputs...")

        result = guardrails.jd.validate(state.job_description)

        if not result.passed:
            state.errors.append(f"InputValidatorAgent: {result.reason}")
            state.log(self.name, f"FAILED: {result.reason}")
        else:
            state.log(self.name, "Input validation passed.")

        return state


# -------------------------------
# AGENT 1 — JD ANALYZER
# -------------------------------
class JDAnalyzerAgent(BaseAgent):
    name = "JDAnalyzerAgent"

    def run(self, state: AgentState) -> AgentState:
        state.log(self.name, f"Analyzing JD for {state.role} at {state.company}...")

        from graphrag_pipeline import extract_jd_requirements
        state.jd_requirements = extract_jd_requirements(
            state.company,
            state.role,
            state.job_description
        )

        tech = state.jd_requirements.get("technical_skills", [])
        systems = state.jd_requirements.get("system_topics", [])
        state.log(self.name, f"Extracted {len(tech)} technical skills, {len(systems)} system topics.")

        return state


# -------------------------------
# AGENT 2 — USER HISTORY
# -------------------------------
class UserHistoryAgent(BaseAgent):
    name = "UserHistoryAgent"

    def run(self, state: AgentState) -> AgentState:
        user_id = getattr(state, "user_id", None)

        if not user_id:
            state.log(self.name, "No user logged in — skipping history load.")
            return state

        state.log(self.name, f"Loading weak areas for user {user_id}...")

        weak_areas = load_weak_areas(user_id)

        if not weak_areas:
            state.log(self.name, "No weak areas found — first attempt or no history.")
            return state

        # Inject weak areas into jd_requirements so graph retrieval boosts them
        topics = [w["topic"] for w in weak_areas]
        state.log(self.name, f"Found {len(topics)} weak areas: {topics}")

        existing = state.jd_requirements.get("priority_requirements", [])
        merged = list(dict.fromkeys(topics + existing))
        state.jd_requirements["priority_requirements"] = merged[:6]

        # Also add to technical skills if not already there
        existing_tech = state.jd_requirements.get("technical_skills", [])
        extra = [t for t in topics if t not in existing_tech]
        state.jd_requirements["technical_skills"] = (existing_tech + extra)[:8]

        state.log(self.name, "Weak areas injected into JD requirements for personalized retrieval.")
        return state

# -------------------------------
# AGENT 3 — GRAPH RETRIEVAL
# -------------------------------
class GraphRetrievalAgent(BaseAgent):
    name = "GraphRetrievalAgent"

    def run(self, state: AgentState) -> AgentState:
        state.log(self.name, "Retrieving relevant clusters from knowledge graph...")

        from graphrag_pipeline import (
            retrieve_top_clusters,
            collect_relevant_questions,
            expand_questions,
            detect_missing_requirements,
            summary_df,
            jobs_df,
            cluster_questions
        )

        top_clusters = retrieve_top_clusters(
            company=state.company,
            role=state.role,
            jd_requirements=state.jd_requirements,
            summary_df=summary_df,
            top_k=5
        )

        selected_questions = collect_relevant_questions(
            top_clusters=top_clusters,
            jobs_df=jobs_df,
            cluster_questions=cluster_questions,
            company=state.company,
            role=state.role,
            max_questions=12
        )

        selected_questions = expand_questions(
            selected_questions,
            state.company,
            state.role
        )

        missing_requirements = detect_missing_requirements(
            jd_requirements=state.jd_requirements,
            selected_questions=selected_questions,
            top_clusters=top_clusters
        )

        state.top_clusters = top_clusters
        state.selected_questions = selected_questions
        state.missing_requirements = missing_requirements[:4]

        state.log(self.name, f"Retrieved {len(selected_questions)} questions from {len(top_clusters)} clusters.")
        state.log(self.name, f"Missing requirements: {state.missing_requirements}")

        return state


# -------------------------------
# AGENT 4 — QUESTION GENERATOR
# -------------------------------
class QuestionGeneratorAgent(BaseAgent):
    name = "QuestionGeneratorAgent"

    def run(self, state: AgentState) -> AgentState:
        state.log(self.name, "Generating mock interview questions...")

        from graphrag_pipeline import generate_mock_interview, coerce_interview_json

        interview_raw = generate_mock_interview(
            company=state.company,
            role=state.role,
            job_description=state.job_description,
            selected_questions=state.selected_questions,
            jd_requirements=state.jd_requirements,
            missing_requirements=state.missing_requirements
        )

        interview_parsed = coerce_interview_json(interview_raw)

        if not isinstance(interview_parsed, dict):
            state.errors.append("QuestionGeneratorAgent: Failed to parse interview JSON")
            state.log(self.name, "ERROR: Could not parse interview JSON")
            state.interview_parsed = {}
        else:
            state.interview_raw = interview_raw
            state.interview_parsed = interview_parsed
            coding_count = len(interview_parsed.get("coding_questions", []))
            state.log(self.name, f"Generated {coding_count} coding questions, 1 system design, 1 behavioral.")

        return state


# -------------------------------
# AGENT 5 — REFLECTION
# -------------------------------
class ReflectionAgent(BaseAgent):
    name = "ReflectionAgent"
    MAX_ATTEMPTS = 2

    def _validate(self, state: AgentState) -> tuple[bool, str]:
        """
        Validate the generated interview.
        Returns (passed, feedback).
        """
        parsed = state.interview_parsed
        issues = []

        if not isinstance(parsed, dict):
            return False, "Interview is not a valid dict."

        coding_questions = parsed.get("coding_questions", [])

        for i, q in enumerate(coding_questions):
            result = guardrails.content.validate_coding_question(q)
            if not result.passed:
                issues.append(f"Q{i+1} content issue: {result.reason}")

        behavioral_q = parsed.get("behavioral", {}).get("question", "")
        result = guardrails.content.validate_behavioral_question(behavioral_q)
        if not result.passed:
            issues.append(f"Behavioral issue: {result.reason}")        

        # Check 1 — must have 3 coding questions
        if len(coding_questions) < 3:
            issues.append(f"Only {len(coding_questions)} coding questions generated, need 3.")

        # Check 2 — difficulties must be Easy, Medium, Hard in order
        expected_difficulties = ["Easy", "Medium", "Hard"]
        for i, q in enumerate(coding_questions[:3]):
            diff = q.get("difficulty", "")
            if diff != expected_difficulties[i]:
                issues.append(
                    f"Q{i+1} difficulty is '{diff}', expected '{expected_difficulties[i]}'."
                )

        # Check 3 — no SQL or database questions in coding
        bad_keywords = ["sql", "database query", "join", "select from"]
        for i, q in enumerate(coding_questions):
            problem = q.get("problem_statement", "").lower()
            if any(kw in problem for kw in bad_keywords):
                issues.append(f"Q{i+1} appears to be a SQL/database question, not DSA.")

        # Check 4 — system design must have all fields
        sd = parsed.get("system_design", {})
        for field in ["title", "use_case", "functional_requirements",
                      "non_functional_requirements", "key_discussion_points"]:
            if not sd.get(field):
                issues.append(f"System design missing field: {field}")

        # Check 5 — behavioral must have a question
        behavioral = parsed.get("behavioral", {})
        if not behavioral.get("question"):
            issues.append("Behavioral question is empty.")

        # Check 6 — each coding question must have test cases
        for i, q in enumerate(coding_questions[:3]):
            if not q.get("test_cases"):
                issues.append(f"Q{i+1} has no test cases.")

        # Check 7 — each coding question must have starter_code
        for i, q in enumerate(coding_questions[:3]):
            starter = q.get("starter_code", {})
            if not isinstance(starter, dict) or not starter.get("python"):
                issues.append(f"Q{i+1} missing starter code.")

        if issues:
            feedback = "\n".join(f"- {issue}" for issue in issues)
            return False, feedback

        return True, "All checks passed."

    def run(self, state: AgentState) -> AgentState:
        state.log(self.name, "Reflecting on generated interview...")

        from graphrag_pipeline import generate_mock_interview, coerce_interview_json

        for attempt in range(self.MAX_ATTEMPTS):
            state.reflection_attempts = attempt + 1
            passed, feedback = self._validate(state)

            if passed:
                state.reflection_passed = True
                state.reflection_feedback = feedback
                state.log(self.name, f"Reflection passed on attempt {attempt + 1}.")
                return state

            state.log(self.name, f"Attempt {attempt + 1} failed:\n{feedback}")
            state.log(self.name, "Asking LLM to fix the issues...")

            # Ask LLM to fix specific issues
            fix_prompt = f"""
You generated a mock interview but it has issues.

Issues found:
{feedback}

Original interview JSON:
{json.dumps(state.interview_parsed, indent=2)[:3000]}

Fix ALL the issues listed above and return the corrected interview as valid JSON.
Follow the exact same JSON structure.
Return ONLY JSON. No explanation.
"""
            fixed_raw = llm(fix_prompt, model="llama3.3-70b")
            fixed_parsed = coerce_interview_json(fixed_raw)

            if isinstance(fixed_parsed, dict):
                state.interview_raw = fixed_raw
                state.interview_parsed = fixed_parsed
                state.log(self.name, f"LLM applied fixes, re-validating...")
            else:
                state.log(self.name, "LLM fix attempt returned invalid JSON.")

        # If still failing after MAX_ATTEMPTS, pass anyway with feedback
        state.reflection_passed = False
        state.reflection_feedback = feedback
        state.log(
            self.name,
            f"Reflection failed after {self.MAX_ATTEMPTS} attempts. "
            f"Proceeding with best available interview."
        )
        return state


# -------------------------------
# AGENT 6 — EVALUATOR
# -------------------------------
class EvaluatorAgent(BaseAgent):
    name = "EvaluatorAgent"

    def run(self, state: AgentState, qa_pairs: list) -> AgentState:
        state.log(self.name, f"Evaluating {len(qa_pairs)} answers...")

        from evaluation_pipeline import evaluate_interview
        state.eval_results = evaluate_interview(
            state.company,
            state.role,
            qa_pairs
        )

        optimized = sum(
            1 for r in state.eval_results
            if r.get("evaluation", {}).get("is_optimized", False)
        )
        state.log(self.name, f"Evaluation complete. {optimized}/{len(state.eval_results)} optimized.")

        return state


# -------------------------------
# AGENT 7 — LEARNING PATH
# -------------------------------
class LearningPathAgent(BaseAgent):
    name = "LearningPathAgent"

    def run(self, state: AgentState, job_description: str = "") -> AgentState:
        state.log(self.name, "Generating personalized learning plan...")

        from evaluation_pipeline import generate_learning_path
        state.learning_plan = generate_learning_path(
            results=state.eval_results,
            role=state.role,
            company=state.company,
            job_description=job_description,
            jd_requirements=state.jd_requirements
        )

        state.log(self.name, "Learning plan generated.")
        return state


# -------------------------------
# PIPELINE ORCHESTRATOR
# -------------------------------
class InterviewPipeline:
    """
    Orchestrates all agents in sequence.
    Replaces run_graphrag_interview().
    """

    def __init__(self):
        self.input_validator = InputValidatorAgent()
        self.jd_analyzer = JDAnalyzerAgent()
        self.user_history = UserHistoryAgent()
        self.graph_retrieval = GraphRetrievalAgent()
        self.question_generator = QuestionGeneratorAgent()
        self.reflection = ReflectionAgent()
        self.evaluator = EvaluatorAgent()
        self.learning_path = LearningPathAgent()

    def generate_interview(self, company: str, role: str,
                        job_description: str, user_id: str = "") -> AgentState:
        state = AgentState(
            company=company,
            role=role,
            job_description=job_description,
            user_id=user_id
        )

        state.log("Pipeline", "Starting interview generation pipeline...")

        # Agent 0 — Validate inputs
        state = self.input_validator.run(state)
        if state.errors:
            state.log("Pipeline", "Pipeline stopped due to input validation failure.")
            return state

        # Agent 1 — Analyze JD
        state = self.jd_analyzer.run(state)

        # Agent 2 — Load user history and inject weak areas
        state = self.user_history.run(state)

        # Agent 3 — Retrieve from Graph (now boosted by weak areas)
        state = self.graph_retrieval.run(state)

        # Agent 3 — Generate Questions
        state = self.question_generator.run(state)

        # Agent 4 — Reflect and Validate
        state = self.reflection.run(state)

        state.log("Pipeline", "Interview generation complete.")
        return state

    def evaluate(self, state: AgentState, qa_pairs: list) -> AgentState:
        """Run evaluation pipeline."""
        state.log("Pipeline", "Starting evaluation pipeline...")

        # Agent 5 — Evaluate answers
        state = self.evaluator.run(state, qa_pairs)

        # Agent 6 — Generate learning plan
        state = self.learning_path.run(state, state.job_description)

        # Save session to user profile if logged in
        if state.user_id and state.eval_results:
            try:
                save_user_session(
                    user_id=state.user_id,
                    company=state.company,
                    role=state.role,
                    results=state.eval_results
                )
                state.log("Pipeline", f"Session saved for user {state.user_id}.")
            except Exception as e:
                state.log("Pipeline", f"Session save failed: {e}")

        state.log("Pipeline", "Evaluation pipeline complete.")
        return state