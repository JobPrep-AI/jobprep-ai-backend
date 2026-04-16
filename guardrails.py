"""
guardrails.py
-------------
Guardrails for JobPrep AI pipeline.

1. JDGuardrail          — validates job description input
2. AnswerGuardrail      — validates user answers before evaluation
3. ContentGuardrail     — checks generated questions for quality
4. CodeExecutionGuardrail — detects dangerous code before execution
"""

import re
import logging

logger = logging.getLogger(__name__)


# -------------------------------
# GUARDRAIL RESULT
# -------------------------------
class GuardrailResult:
    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return f"GuardrailResult(passed={self.passed}, reason='{self.reason}')"


# -------------------------------
# 1. JD GUARDRAIL
# Input validation for job description
# -------------------------------
class JDGuardrail:
    MIN_LENGTH = 50
    MAX_LENGTH = 10000

    JOB_SIGNALS = [
        "engineer", "developer", "scientist", "analyst",
        "manager", "designer", "architect", "intern",
        "experience", "skills", "requirements", "responsibilities",
        "qualification", "degree", "bachelor", "master",
        "python", "java", "sql", "cloud", "aws", "azure",
        "team", "collaborate", "communication", "problem",
        "role", "position", "job", "work", "company",
        "salary", "benefits", "full-time", "part-time",
        "remote", "hybrid", "onsite", "office"
    ]

    PROMPT_INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"ignore all instructions",
        r"you are now",
        r"forget everything",
        r"system prompt",
        r"jailbreak",
        r"act as",
        r"pretend you are",
        r"disregard",
    ]

    def validate(self, job_description: str) -> GuardrailResult:
        if not job_description or not isinstance(job_description, str):
            return GuardrailResult(False, "Job description is empty.")

        text = job_description.strip()

        # Check minimum length
        if len(text) < self.MIN_LENGTH:
            return GuardrailResult(
                False,
                f"Job description is too short ({len(text)} chars). "
                f"Please provide a more detailed JD (min {self.MIN_LENGTH} chars)."
            )

        # Check maximum length
        if len(text) > self.MAX_LENGTH:
            return GuardrailResult(
                False,
                f"Job description is too long ({len(text)} chars). "
                f"Please shorten it to under {self.MAX_LENGTH} chars."
            )

        # Check if it looks like an actual job description
        lower = text.lower()
        signal_count = sum(1 for s in self.JOB_SIGNALS if s in lower)
        if signal_count < 3:
            return GuardrailResult(
                False,
                "This does not look like a job description. "
                "Please paste an actual job description with skills, "
                "requirements, and responsibilities."
            )

        # Check for prompt injection attempts
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lower):
                return GuardrailResult(
                    False,
                    "Invalid input detected in job description. "
                    "Please paste a real job description."
                )

        return GuardrailResult(True, "JD validation passed.")


# -------------------------------
# 2. ANSWER GUARDRAIL
# Validates user answers before evaluation
# -------------------------------
class AnswerGuardrail:
    MIN_ANSWER_LENGTH = 10

    GIBBERISH_PATTERNS = [
        r"^[a-z]{1,3}$",           # too short like "ab"
        r"^(.)\1+$",                # repeated single char like "aaaaaaa"
        r"^[^a-zA-Z0-9]+$",        # only special characters
    ]

    def validate_coding_answer(self, answer: str) -> GuardrailResult:
        if not answer or not isinstance(answer, str):
            return GuardrailResult(False, "Answer is empty.")

        text = answer.strip()

        if len(text) < self.MIN_ANSWER_LENGTH:
            return GuardrailResult(
                False,
                f"Answer is too short to evaluate meaningfully."
            )

        # Check for gibberish
        for pattern in self.GIBBERISH_PATTERNS:
            if re.match(pattern, text):
                return GuardrailResult(
                    False,
                    "Answer does not appear to be valid code."
                )

        # Check for at least some code-like content
        code_signals = [
            "def ", "class ", "return ", "for ", "while ",
            "if ", "import ", "int ", "string ", "void ",
            "public ", "private ", "#include", "function",
            "{", "}", "()", "[]"
        ]
        has_code = any(s in text for s in code_signals)
        if not has_code:
            return GuardrailResult(
                False,
                "Answer does not appear to contain code. "
                "Please write an actual code solution."
            )

        return GuardrailResult(True, "Answer validation passed.")

    def validate_text_answer(self, answer: str) -> GuardrailResult:
        if not answer or not isinstance(answer, str):
            return GuardrailResult(False, "Answer is empty.")

        text = answer.strip()

        if len(text) < self.MIN_ANSWER_LENGTH:
            return GuardrailResult(
                False,
                "Answer is too short to evaluate meaningfully."
            )

        return GuardrailResult(True, "Answer validation passed.")


# -------------------------------
# 3. CONTENT GUARDRAIL
# Checks generated questions for quality and appropriateness
# -------------------------------
class ContentGuardrail:

    INAPPROPRIATE_PATTERNS = [
        r"racist", r"sexist", r"offensive", r"discriminat",
        r"illegal", r"hack", r"exploit", r"malware",
        r"violence", r"weapon", r"drug"
    ]

    NON_DSA_PATTERNS = [
        r"select .* from",
        r"insert into",
        r"update .* set",
        r"delete from",
        r"create table",
        r"drop table",
        r"sql query",
        r"database query",
    ]

    def validate_coding_question(self, question: dict) -> GuardrailResult:
        problem = question.get("problem_statement", "").lower()
        title = question.get("title", "").lower()
        combined = f"{title} {problem}"

        # Check for inappropriate content
        for pattern in self.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, combined):
                return GuardrailResult(
                    False,
                    f"Question contains inappropriate content: '{pattern}'"
                )

        # Check for SQL/non-DSA content
        for pattern in self.NON_DSA_PATTERNS:
            if re.search(pattern, combined):
                return GuardrailResult(
                    False,
                    f"Coding question appears to be SQL/database, not DSA: '{pattern}'"
                )

        # Must have problem statement
        if len(problem) < 20:
            return GuardrailResult(
                False,
                "Problem statement is too short or empty."
            )

        return GuardrailResult(True, "Content validation passed.")

    def validate_behavioral_question(self, question: str) -> GuardrailResult:
        if not question or len(question.strip()) < 10:
            return GuardrailResult(False, "Behavioral question is empty.")

        lower = question.lower()

        # Check for inappropriate content
        for pattern in self.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, lower):
                return GuardrailResult(
                    False,
                    f"Behavioral question contains inappropriate content."
                )

        return GuardrailResult(True, "Behavioral question validation passed.")


# -------------------------------
# 4. CODE EXECUTION GUARDRAIL
# Detects dangerous code before sending to executor
# -------------------------------
class CodeExecutionGuardrail:

    DANGEROUS_PATTERNS = {
        "python": [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"import\s+socket",
            r"import\s+shutil",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"os\.system",
            r"os\.popen",
            r"subprocess\.",
            r"requests\.",
            r"urllib",
            r"rmdir",
            r"remove\s*\(",
            r"while\s+True\s*:",
        ],
        "java": [
            r"Runtime\.getRuntime",
            r"ProcessBuilder",
            r"System\.exit",
            r"FileWriter",
            r"FileReader",
            r"BufferedWriter",
            r"Socket\s*\(",
            r"ServerSocket",
            r"while\s*\(\s*true\s*\)",
        ],
        "cpp": [
            r"system\s*\(",
            r"popen\s*\(",
            r"fork\s*\(",
            r"exec\s*\(",
            r"fopen\s*\(",
            r"remove\s*\(",
            r"socket\s*\(",
            r"while\s*\(\s*1\s*\)",
            r"while\s*\(\s*true\s*\)",
        ]
    }

    # Patterns that are safe even if they match dangerous ones
    SAFE_EXCEPTIONS = {
        "python": [
            r"import\s+sys\s+as\s+_sys",  # our own injected validator
        ]
    }

    def validate(self, code: str, lang: str) -> GuardrailResult:
        if not code or not isinstance(code, str):
            return GuardrailResult(False, "Code is empty.")

        patterns = self.DANGEROUS_PATTERNS.get(lang, [])
        safe_exceptions = self.SAFE_EXCEPTIONS.get(lang, [])

        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                # Check if it matches a safe exception
                is_exception = any(
                    re.search(exc, code, re.IGNORECASE)
                    for exc in safe_exceptions
                )
                if not is_exception:
                    logger.warning(
                        f"CodeExecutionGuardrail blocked pattern: {pattern}"
                    )
                    return GuardrailResult(
                        False,
                        f"Code contains potentially unsafe operations. "
                        f"Please avoid system calls, file operations, "
                        f"and network requests in your solution."
                    )

        return GuardrailResult(True, "Code safety check passed.")


# -------------------------------
# GUARDRAIL REGISTRY
# Single access point for all guardrails
# -------------------------------
class GuardrailRegistry:
    def __init__(self):
        self.jd = JDGuardrail()
        self.answer = AnswerGuardrail()
        self.content = ContentGuardrail()
        self.code_execution = CodeExecutionGuardrail()


# Singleton instance
guardrails = GuardrailRegistry()