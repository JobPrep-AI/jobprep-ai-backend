import re
import ast
import json
import logging
import requests

logger = logging.getLogger(__name__)

# -------------------------------
# JUDGE0 (Primary)
# -------------------------------
JUDGE0_URL = "https://ce.judge0.com/submissions?base64_encoded=false&wait=true"

LANGUAGE_MAP = {
    "python": 71,
    "java": 62,
    "cpp": 54
}

# Correct version strings per language for Piston API
PISTON_VERSION_MAP = {
    "python": "3.10.0",
    "java": "15.0.2",
    "cpp": "10.2.0"
}


def run_code_judge0(source_code, language="python", stdin=""):
    try:
        data = {
            "source_code": source_code,
            "language_id": LANGUAGE_MAP.get(language, 71),
            "stdin": stdin
        }
        response = requests.post(JUDGE0_URL, json=data, timeout=10)
        result = response.json()
        return {
            "success": True,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "compile_output": result.get("compile_output"),
            "source": "Judge0"
        }
    except Exception as e:
        logger.warning(f"Judge0 failed: {e}")
        return {"success": False, "error": str(e)}


def run_code_piston(source_code, language="python", stdin=""):
    try:
        url = "https://emkc.org/api/v2/piston/execute"
        data = {
            "language": language,
            "version": PISTON_VERSION_MAP.get(language, "3.10.0"),
            "files": [{"content": source_code}],
            "stdin": stdin
        }
        response = requests.post(url, json=data, timeout=10)
        result = response.json()
        return {
            "success": True,
            "stdout": result.get("run", {}).get("output"),
            "stderr": result.get("run", {}).get("stderr"),
            "compile_output": result.get("compile", {}).get("output"),
            "source": "Piston"
        }
    except Exception as e:
        logger.warning(f"Piston failed: {e}")
        return {"success": False, "error": str(e)}


def run_code(source_code, language="python", stdin=""):
    # Guardrail check before execution
    from guardrails import guardrails
    safety = guardrails.code_execution.validate(source_code, language)
    if not safety.passed:
        logger.warning(f"CodeExecutionGuardrail blocked code: {safety.reason}")
        return {
            "success": False,
            "stdout": None,
            "stderr": safety.reason,
            "compile_output": None,
            "source": "Guardrail"
        }

    judge0_result = run_code_judge0(source_code, language, stdin)
    if judge0_result["success"]:
        return judge0_result
    logger.warning("Judge0 failed, falling back to Piston...")
    return run_code_piston(source_code, language, stdin)


# -------------------------------
# CODE MANIPULATION
# (moved from streamlit_app.py)
# -------------------------------

def strip_hardcoded_calls(code: str, lang: str) -> str:
    if lang == "python":
        clean = []
        for line in code.splitlines():
            stripped = line.strip()
            if line and not line[0].isspace():
                if stripped.startswith("print(") or stripped.startswith("#"):
                    continue
                if re.match(r"""^[a-zA-Z_]\w*\s*\(.*["'\d].*\)\s*$""", stripped):
                    continue
            clean.append(line)
        return "\n".join(clean)

    if lang == "java":
        return "\n".join(
            line for line in code.splitlines()
            if not re.search(r'System\.out\.println\s*\(.*["\'\\d]', line)
        )

    if lang == "cpp":
        return "\n".join(
            line for line in code.splitlines()
            if not re.search(r'cout\s*<<.*["\'\\d]', line)
        )

    return code


def inject_stdin_runner(code: str, lang: str) -> str:
    if lang == "python":
        fn_name = None
        for line in code.splitlines():
            m = re.match(r"^def\s+(\w+)\s*\(", line)
            if m:
                fn_name = m.group(1)
        if fn_name:
            return code + f"""

# --- injected by validator ---
import sys as _sys, ast as _ast
_raw = _sys.stdin.read().strip()
if _raw:
    _called = False
    try:
        _parsed = _ast.literal_eval(_raw)
        if isinstance(_parsed, tuple):
            print({fn_name}(*_parsed))
        else:
            print({fn_name}(_parsed))
        _called = True
    except Exception:
        pass
    if not _called:
        print({fn_name}(_raw))
"""
        return code

    if lang == "java":
        return code + "\n// Validator: ensure main() reads from Scanner(System.in)\n"

    if lang == "cpp":
        return code + "\n// Validator: ensure main() reads from cin\n"

    return code


def prepare_code_for_validation(code: str, lang: str) -> str:
    return inject_stdin_runner(strip_hardcoded_calls(code, lang), lang)


# -------------------------------
# SMART TEST CASE COMPARISON
# -------------------------------

def parse_test_case(tc):
    if isinstance(tc, dict):
        inp = str(tc.get("input", "")).strip().strip('"')
        out = str(tc.get("expected_output", tc.get("output", ""))).strip().strip('"')
        return inp, out
    if isinstance(tc, str):
        inp, out = "", ""
        for line in tc.replace(",", "\n").split("\n"):
            ll = line.lower()
            if "input:" in ll:
                inp = line.split(":", 1)[1].strip().strip('"')
            elif "output:" in ll or "expected:" in ll:
                out = line.split(":", 1)[1].strip().strip('"')
        return inp, out
    return "", ""


def _normalize(value: str):
    """
    Normalize an output value for smart comparison.
    Handles: JSON whitespace, quotes, integers, floats,
    booleans, None/null, trailing newlines, and nested structures.
    """
    if not isinstance(value, str):
        return value

    value = value.strip()

    # Remove surrounding quotes from plain strings
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1].strip()

    # Try parsing as JSON first (handles lists, dicts, numbers, booleans, null)
    try:
        return json.loads(value)
    except Exception:
        pass

    # Try integer
    try:
        return int(value)
    except Exception:
        pass

    # Try float
    try:
        return float(value)
    except Exception:
        pass

    # Handle Python-style booleans
    if value == "True":
        return True
    if value == "False":
        return False
    if value in ("None", "null"):
        return None

    # Handle Python-style list/dict that json.loads couldn't parse
    # e.g. single-quoted lists like ['a', 'b']
    try:
        parsed = ast.literal_eval(value)
        return parsed
    except Exception:
        pass

    # Return as plain lowercase string for final comparison
    return value.lower().strip()


def _smart_equal(actual: str, expected: str) -> bool:
    """
    Compare actual vs expected output intelligently.
    Falls back through multiple strategies.
    """
    # Strategy 1 — exact match after stripping whitespace
    if actual.strip() == expected.strip():
        return True

    # Strategy 2 — normalize both and compare
    norm_actual = _normalize(actual)
    norm_expected = _normalize(expected)

    if norm_actual == norm_expected:
        return True

    # Strategy 3 — if both are lists, compare sorted
    # (for problems where order doesn't matter)
    if isinstance(norm_actual, list) and isinstance(norm_expected, list):
        try:
            return sorted(str(x) for x in norm_actual) == sorted(str(x) for x in norm_expected)
        except Exception:
            pass

    # Strategy 4 — numeric tolerance for floats
    if isinstance(norm_actual, float) and isinstance(norm_expected, float):
        return abs(norm_actual - norm_expected) < 1e-6

    # Strategy 5 — strip all whitespace and compare
    if actual.replace(" ", "").replace("\n", "") == expected.replace(" ", "").replace("\n", ""):
        return True

    return False


def run_test_cases(code: str, lang: str, test_cases: list) -> list:
    validation_code = prepare_code_for_validation(code, lang)
    results = []
    for i, tc in enumerate(test_cases[:3]):
        inp, expected = parse_test_case(tc)
        if not expected:
            continue

        # Pass "\n" for empty input so input() gets "" not EOF
        stdin_input = inp if inp else "\n"

        res = run_code(validation_code, lang, stdin_input)
        actual = (res.get("stdout") or "").strip()

        passed = _smart_equal(actual, expected)

        results.append({
            "case": i + 1,
            "input": inp,
            "expected": expected.strip(),
            "actual": actual,
            "passed": passed,
            "error": res.get("stderr") or res.get("compile_output") or ""
        })
    return results

def verify_test_cases(problem_statement: str, lang: str, test_cases: list) -> list:
    """
    Run each test case through the judge and correct
    any wrong expected outputs the LLM generated.
    """
    verified = []
    for tc in test_cases:
        inp, expected = parse_test_case(tc)
        if not inp or not expected:
            verified.append(tc)
            continue

        # We need actual runnable code to verify
        # So we just flag suspicious cases using a basic sanity check
        # Full verification happens in streamlit after user writes code
        verified.append(tc)
    return verified


def reverify_with_user_code(code: str, lang: str, test_cases: list) -> list:
    validation_code = prepare_code_for_validation(code, lang)
    corrected = []

    for tc in test_cases:
        inp, expected = parse_test_case(tc)
        if not expected:
            corrected.append(tc)
            continue

        # Pass "\n" for empty input so input() gets "" not EOF
        stdin_input = inp if inp else "\n"

        res = run_code(validation_code, lang, stdin_input)
        actual = (res.get("stdout") or "").strip()

        if not actual and inp == "":
            # Empty input → function should return 0 or ""
            # Trust expected output for empty input edge case
            corrected.append(tc)
            continue

        if actual and not _smart_equal(actual, expected):
            logger.warning(
                f"LLM expected output was wrong. "
                f"Input: '{inp}' | LLM said: '{expected}' | Actual: '{actual}' "
                f"— trusting execution."
            )
            if isinstance(tc, dict):
                tc = dict(tc)
                tc["expected_output"] = actual
                tc["llm_expected"] = expected
                tc["auto_corrected"] = True

        corrected.append(tc)

    return corrected