import requests

# -------------------------------
# JUDGE0 (Primary)
# -------------------------------
JUDGE0_URL = "https://ce.judge0.com/submissions?base64_encoded=false&wait=true"

LANGUAGE_MAP = {
    "python": 71,
    "java": 62,
    "cpp": 54
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
        return {
            "success": False,
            "error": str(e)
        }


# -------------------------------
# PISTON (Fallback)
# -------------------------------
def run_code_piston(source_code, language="python"):
    try:
        url = "https://emkc.org/api/v2/piston/execute"

        data = {
            "language": language,
            "version": "3.10.0",
            "files": [
                {"content": source_code}
            ]
        }

        response = requests.post(url, json=data, timeout=10)
        result = response.json()

        return {
            "success": True,
            "stdout": result.get("run", {}).get("output"),
            "stderr": None,
            "compile_output": None,
            "source": "Piston"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# -------------------------------
# HYBRID EXECUTOR
# -------------------------------
def run_code(source_code, language="python", stdin=""):
    # Try Judge0 first
    judge0_result = run_code_judge0(source_code, language, stdin)

    if judge0_result["success"]:
        return judge0_result

    # Fallback to Piston
    piston_result = run_code_piston(source_code, language)

    return piston_result