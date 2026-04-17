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
from typing import Optional, Literal
from user_profile import load_weak_areas, save_user_session
from snowflake_utils import llm
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from langsmith.wrappers import wrap_openai

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
    mode: str = "deep"
    weak_areas: list = field(default_factory=list)

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
    user_answers: list = field(default_factory=list)

    # Learning Path outputs
    learning_plan: str = ""

    # Pipeline metadata
    errors: list = field(default_factory=list)
    agent_logs: list = field(default_factory=list)

    def log(self, agent: str, message: str):
        entry = f"[{agent}] {message}"
        self.agent_logs.append(entry)
        logger.info(entry)

    def to_dict(self) -> dict:
        """Convert to dict for LangGraph state passing."""
        import pandas as pd
        # Serialize top_clusters DataFrame to JSON string
        top_clusters_serial = None
        if self.top_clusters is not None and isinstance(self.top_clusters, pd.DataFrame):
            # Drop embedding column — numpy arrays not serializable
            df = self.top_clusters.copy()
            if "embedding" in df.columns:
                df = df.drop(columns=["embedding"])
            top_clusters_serial = df.to_json(orient="records")
        return {
            "company": self.company,
            "role": self.role,
            "job_description": self.job_description,
            "user_id": self.user_id,
            "mode": self.mode,
            "weak_areas": self.weak_areas,
            "jd_requirements": self.jd_requirements,
            "top_clusters": top_clusters_serial,
            "selected_questions": self.selected_questions,
            "missing_requirements": self.missing_requirements,
            "interview_raw": self.interview_raw,
            "interview_parsed": self.interview_parsed,
            "reflection_passed": self.reflection_passed,
            "reflection_feedback": self.reflection_feedback,
            "reflection_attempts": self.reflection_attempts,
            "eval_results": self.eval_results,
            "user_answers": self.user_answers,
            "learning_plan": self.learning_plan,
            "errors": self.errors,
            "agent_logs": self.agent_logs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentState":
        """Reconstruct from LangGraph state dict."""
        import pandas as pd
        state = cls()
        for key, value in d.items():
            if not hasattr(state, key):
                continue
            # Deserialize top_clusters from JSON string back to DataFrame
            if key == "top_clusters" and isinstance(value, str):
                try:
                    setattr(state, key, pd.read_json(value, orient="records"))
                except Exception:
                    setattr(state, key, None)
            else:
                setattr(state, key, value)
        return state

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

        state.weak_areas = weak_areas
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
            missing_requirements=state.missing_requirements,
            weak_areas=state.weak_areas if state.weak_areas else None
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

        # Check 1 — must have 4 coding questions
        if len(coding_questions) < 4:
            issues.append(f"Only {len(coding_questions)} coding questions generated, need 4.")

        # Check 2 — difficulties must be Easy, Medium, Hard in order
        expected_difficulties = ["Easy", "Medium", "Hard", "Hard"]
        for i, q in enumerate(coding_questions[:4]):
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

        # Check 4 — system design questions must be a list with 2 items
        sd_questions = parsed.get("system_design_questions", [])
        if len(sd_questions) < 2:
            issues.append(f"Only {len(sd_questions)} system design questions, need 2.")
        for j, sd in enumerate(sd_questions[:2]):
            for f in ["title", "use_case", "functional_requirements",
                      "non_functional_requirements", "key_discussion_points"]:
                if not sd.get(f):
                    issues.append(f"System design {j+1} missing field: {f}")

        # Check 5 — behavioral must have a question
        behavioral = parsed.get("behavioral", {})
        if not behavioral.get("question"):
            issues.append("Behavioral question is empty.")

        # Check 6 — each coding question must have test cases
        for i, q in enumerate(coding_questions[:4]):
            if not q.get("test_cases"):
                issues.append(f"Q{i+1} has no test cases.")

        # Check 7 — each coding question must have starter_code
        for i, q in enumerate(coding_questions[:4]):
            starter = q.get("starter_code", {})
            if not isinstance(starter, dict) or not starter.get("python"):
                issues.append(f"Q{i+1} missing starter code.")

        if issues:
            feedback = "\n".join(f"- {issue}" for issue in issues)
            return False, feedback

        return True, "All checks passed."

    def run(self, state: AgentState) -> AgentState:
        state.log(self.name, "Reflecting on generated interview...")

        state.reflection_attempts += 1
        passed, feedback = self._validate(state)

        if passed:
            state.reflection_passed = True
            state.reflection_feedback = feedback
            state.log(self.name, f"Reflection passed on attempt {state.reflection_attempts}.")
            return state

        # Validation failed — LangGraph will route back to question_generator
        state.reflection_passed = False
        state.reflection_feedback = feedback
        state.log(self.name, f"Attempt {state.reflection_attempts} failed:\n{feedback}")
        state.log(self.name, "LangGraph will retry via question_generator node.")
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
            jd_requirements=state.jd_requirements,
            user_answers=state.user_answers
        )
        state.log(self.name, "Learning plan generated.")
        return state


# -------------------------------
# LANGGRAPH NODE FUNCTIONS
# Thin wrappers around agent classes
# -------------------------------
_input_validator   = InputValidatorAgent()
_jd_analyzer       = JDAnalyzerAgent()
_user_history      = UserHistoryAgent()
_graph_retrieval   = GraphRetrievalAgent()
_question_gen      = QuestionGeneratorAgent()
_reflection        = ReflectionAgent()
_evaluator         = EvaluatorAgent()
_learning_path     = LearningPathAgent()


@traceable(name="InputValidator")
def node_input_validator(state: dict) -> dict:
    s = AgentState.from_dict(state)
    s = _input_validator.run(s)
    return s.to_dict()

@traceable(name="JDAnalyzer")
def node_jd_analyzer(state: dict) -> dict:
    s = AgentState.from_dict(state)
    s = _jd_analyzer.run(s)
    return s.to_dict()

@traceable(name="UserHistory")
def node_user_history(state: dict) -> dict:
    s = AgentState.from_dict(state)
    s = _user_history.run(s)
    return s.to_dict()

@traceable(name="GraphRetrieval")
def node_graph_retrieval(state: dict) -> dict:
    s = AgentState.from_dict(state)
    s = _graph_retrieval.run(s)
    return s.to_dict()

@traceable(name="QuestionGenerator")
def node_question_generator(state: dict) -> dict:
    s = AgentState.from_dict(state)
    s = _question_gen.run(s)
    return s.to_dict()

@traceable(name="ReflectionAgent")
def node_reflection(state: dict) -> dict:
    s = AgentState.from_dict(state)
    s = _reflection.run(s)
    return s.to_dict()

@traceable(name="Evaluator")
def node_evaluator(state: dict, qa_pairs: list) -> dict:
    s = AgentState.from_dict(state)
    s = _evaluator.run(s, qa_pairs)
    return s.to_dict()

@traceable(name="LearningPath")
def node_learning_path(state: dict) -> dict:
    s = AgentState.from_dict(state)
    s = _learning_path.run(s, s.job_description)
    return s.to_dict()

@traceable(name="SaveSession")
def node_save_session(state: dict) -> dict:
    s = AgentState.from_dict(state)
    if s.user_id and s.eval_results:
        try:
            save_user_session(
                user_id=s.user_id,
                company=s.company,
                role=s.role,
                results=s.eval_results
            )
            s.log("Pipeline", f"Session saved for user {s.user_id}.")
        except Exception as e:
            s.log("Pipeline", f"Session save failed: {e}")
    return s.to_dict()

# -------------------------------
# ROUTING FUNCTIONS
# Conditional edges for LangGraph
# -------------------------------
def route_after_validation(state: dict) -> Literal["jd_analyzer", "__end__"]:
    """Stop pipeline if input validation failed."""
    if state.get("errors"):
        return END
    return "jd_analyzer"

def route_after_reflection(state: dict) -> Literal["question_generator", "__end__"]:
    """
    If reflection failed and attempts < MAX_ATTEMPTS → retry question generation.
    If reflection passed or max attempts reached → end generation pipeline.
    """
    reflection_passed   = state.get("reflection_passed", False)
    reflection_attempts = state.get("reflection_attempts", 0)

    if not reflection_passed and reflection_attempts < 2:
        return "question_generator"
    return END


# -------------------------------
# BUILD LANGGRAPH PIPELINES
# -------------------------------
def _build_generation_graph() -> StateGraph:
    """Build the interview generation graph."""
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("input_validator",    node_input_validator)
    graph.add_node("jd_analyzer",        node_jd_analyzer)
    graph.add_node("user_history",       node_user_history)
    graph.add_node("graph_retrieval",    node_graph_retrieval)
    graph.add_node("question_generator", node_question_generator)
    graph.add_node("reflection",         node_reflection)

    # Entry point
    graph.set_entry_point("input_validator")

    # Fixed edges
    graph.add_edge("jd_analyzer",        "user_history")
    graph.add_edge("user_history",        "graph_retrieval")
    graph.add_edge("graph_retrieval",     "question_generator")
    graph.add_edge("question_generator",  "reflection")

    # Conditional edges
    graph.add_conditional_edges(
        "input_validator",
        route_after_validation,
        {
            "jd_analyzer": "jd_analyzer",
            END: END
        }
    )

    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "question_generator": "question_generator",
            END: END
        }
    )

    return graph


# -------------------------------
# INTERVIEW PIPELINE
# Same interface as before —
# streamlit_app.py needs zero changes
# -------------------------------
class InterviewPipeline:
    """
    LangGraph-orchestrated interview pipeline.
    Exposes same interface as previous custom pipeline.
    """

    def __init__(self):
        self.checkpointer = MemorySaver()
        generation_graph  = _build_generation_graph()
        self.generation_app = generation_graph.compile(
            checkpointer=self.checkpointer
        )
        logger.info("LangGraph InterviewPipeline initialized.")

    @traceable(name="InterviewPipeline.generate_interview")
    def generate_interview(self, company: str, role: str,
                           job_description: str, user_id: str = "",
                           mode: str = "deep") -> AgentState:
        """
        Run interview generation pipeline via LangGraph.
        Returns AgentState — same as before.
        """
        initial_state = AgentState(
            company=company,
            role=role,
            job_description=job_description,
            user_id=user_id,
            mode=mode
        ).to_dict()

        config = {
            "configurable": {
                "thread_id": f"{user_id}_{company}_{role}"
            }
        }

        logger.info("Starting LangGraph generation pipeline...")
        final_state = self.generation_app.invoke(
            initial_state,
            config=config
        )

        return AgentState.from_dict(final_state)

    @traceable(name="InterviewPipeline.evaluate")
    def evaluate(self, state: AgentState, qa_pairs: list) -> AgentState:
        """
        Run evaluation pipeline.
        Evaluation is sequential so runs directly without LangGraph.
        """
        state.log("Pipeline", "Starting evaluation pipeline...")

        state = _evaluator.run(state, qa_pairs)
        state = _learning_path.run(state, state.job_description)

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