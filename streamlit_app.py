import json
import re
import time
import plotly.graph_objects as go
import streamlit as st
from graphrag_pipeline import coerce_interview_json, run_graphrag_interview
from evaluation_pipeline import evaluate_interview, generate_learning_path, generate_hint
from code_executor import run_code
from history_tracker import save_session, load_recent_sessions

st.set_page_config(page_title="JobPrep AI", layout="wide")

DIFFICULTY_COLORS = {"Easy": "#00C851", "Medium": "#FF8800", "Hard": "#FF4444"}
TOTAL_SECONDS = 75 * 60


# -------------------------------
# HELPERS
# -------------------------------
def is_valid_dsa(q):
    bad = ["sql query", "database query", "table", "join"]
    return not any(b in q.lower() for b in bad)


def render_radar_chart(results):
    metric_sums, metric_counts = {}, {}
    for r in results:
        for k, v in r.get("evaluation", {}).get("scores", {}).items():
            label = k.replace("_", " ").title()
            metric_sums[label] = metric_sums.get(label, 0) + (v or 0)
            metric_counts[label] = metric_counts.get(label, 0) + 1

    if not metric_sums:
        return None

    cats = list(metric_sums.keys())
    vals = [round(metric_sums[k] / metric_counts[k], 1) for k in cats]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(99,110,250,0.2)",
        line=dict(color="rgba(99,110,250,0.9)", width=2),
        name="Your Score"
    ))
    fig.add_trace(go.Scatterpolar(
        r=[10] * (len(cats) + 1), theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(200,200,200,0.08)",
        line=dict(color="rgba(200,200,200,0.4)", width=1, dash="dash"),
        name="Perfect Score"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True, title="Performance Radar",
        height=420, margin=dict(l=80, r=80, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def render_timer(start_time):
    remaining = max(0, TOTAL_SECONDS - int(time.time() - start_time))
    mins, secs = remaining // 60, remaining % 60

    if remaining <= 0:
        st.sidebar.error("⏰ Time's up!")
        return

    color = "🔴" if remaining < 300 else "🟠" if remaining < 900 else "🟢"
    st.sidebar.markdown(
        f"<div style='text-align:center;padding:12px;background:#1e1e2e;"
        f"border-radius:8px;margin-bottom:8px'>"
        f"<span style='font-size:32px;font-weight:bold'>"
        f"{color} {mins:02d}:{secs:02d}</span><br>"
        f"<small style='color:#888'>Time Remaining</small></div>",
        unsafe_allow_html=True
    )


def parse_test_case(tc):
    if isinstance(tc, dict):
        inp = str(tc.get("input", "")).strip()
        out = str(tc.get("expected_output", tc.get("output", ""))).strip()
        return inp, out
    if isinstance(tc, str):
        inp, out = "", ""
        for line in tc.replace(",", "\n").split("\n"):
            ll = line.lower()
            if "input:" in ll:
                inp = line.split(":", 1)[1].strip()
            elif "output:" in ll or "expected:" in ll:
                out = line.split(":", 1)[1].strip()
        return inp, out
    return "", ""


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


def run_test_cases(code, lang, test_cases):
    validation_code = prepare_code_for_validation(code, lang)
    results = []
    for i, tc in enumerate(test_cases[:3]):
        inp, expected = parse_test_case(tc)
        if not expected:
            continue
        res = run_code(validation_code, lang, inp)
        actual = (res.get("stdout") or "").strip()
        expected = expected.strip()
        results.append({
            "case": i + 1,
            "input": inp,
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
            "error": res.get("stderr") or res.get("compile_output") or ""
        })
    return results


# -------------------------------
# SESSION STATE
# -------------------------------

# -------------------------------
# SESSION STATE
# -------------------------------
for key in ["interview_data", "user_answers", "interview_start_time", "eval_results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# SIDEBAR — Timer + History
# -------------------------------
with st.sidebar:
    st.markdown("## ⏱️ Interview Timer")
    if st.session_state.interview_start_time:
        render_timer(st.session_state.interview_start_time)
        if st.button("↻ Refresh Timer", key="refresh_timer"):
            st.rerun()
    else:
        st.info("Timer starts when you generate an interview.")

    st.markdown("---")
    st.markdown("## 📋 Past Sessions")
    try:
        sessions = load_recent_sessions(5)
        if sessions:
            for s in sessions:
                with st.expander(f"🏢 {s['company']} — {s['role'][:20]}"):
                    st.caption(s["created_at"])
                    col1, col2 = st.columns(2)
                    col1.metric("Avg Score", f"{s['avg_score']}/10")
                    col2.metric("Optimized", f"{s['optimized']}/{s['total']}")
        else:
            st.caption("No past sessions yet.")
    except Exception as e:
        st.caption(f"History unavailable: {e}")

# -------------------------------
# MAIN
# -------------------------------
st.title("JobPrep AI — Mock Interview Generator")

company = st.text_input("Company Name", "Google")
role = st.text_input("Role Name", "Software Engineer")
job_description = st.text_area("Job Description", height=220)

# -------------------------------
# GENERATE INTERVIEW
# -------------------------------
if st.button("Generate Mock Interview", type="primary"):
    if not company or not role or not job_description.strip():
        st.warning("Please provide all inputs.")
    else:
        with st.spinner("Generating your interview..."):
            result = run_graphrag_interview(company, role, job_description)
        st.session_state.interview_data = result
        st.session_state.user_answers = []
        st.session_state.eval_results = None
        st.session_state.interview_start_time = time.time()
        for k in list(st.session_state.keys()):
            if k.startswith("hints_"):
                del st.session_state[k]
        st.rerun()

# -------------------------------
# DISPLAY INTERVIEW
# -------------------------------
if st.session_state.interview_data:

    _, _, interview, jd_requirements, _ = st.session_state.interview_data
    parsed = coerce_interview_json(interview)

    if not isinstance(parsed, dict):
        st.error("Invalid interview format. Please regenerate.")
        st.stop()

    user_answers = []

    # ----------------------------
    # CODING QUESTIONS
    # ----------------------------
    st.markdown("## Coding Questions")

    raw_questions = parsed.get("coding_questions", [])
    filtered = [q for q in raw_questions if is_valid_dsa(q.get("problem_statement", ""))]
    coding_questions = filtered if filtered else raw_questions

    for i, q in enumerate(coding_questions, start=1):
        difficulty = q.get("difficulty", "Medium")
        diff_color = DIFFICULTY_COLORS.get(difficulty, "#888")

        with st.expander(f"Q{i}: {q.get('title', 'Untitled')}"):

            st.markdown(
                f'<span style="background:{diff_color};color:white;padding:2px 10px;'
                f'border-radius:12px;font-size:13px;font-weight:600">{difficulty}</span>',
                unsafe_allow_html=True
            )

            st.markdown("### Problem")
            problem_text = q.get("problem_statement") or q.get("title") or "No question available"
            st.write(problem_text)

            example = q.get("example_input_output")
            if example:
                st.markdown("### Example")
                st.code(example)

            # --- HINT SYSTEM ---
            hint_key = f"hints_{i}"
            if hint_key not in st.session_state:
                st.session_state[hint_key] = []

            hints_used = len(st.session_state[hint_key])
            hint_col, _ = st.columns([1, 3])
            with hint_col:
                if hints_used < 3:
                    if st.button(f"💡 Hint {hints_used + 1}/3", key=f"hint_btn_{i}"):
                        with st.spinner("Generating hint..."):
                            hint = generate_hint(problem_text, hints_used + 1)
                        st.session_state[hint_key].append(hint)
                        st.rerun()
                else:
                    st.caption("All hints used")

            for j, hint_text in enumerate(st.session_state[hint_key], 1):
                st.warning(f"**Hint {j}/3:** {hint_text}")

            # --- CODE INPUT ---
            lang = st.selectbox(
                f"Language Q{i}", ["python", "java", "cpp"], key=f"lang_{i}"
            )
            code = st.text_area(f"Write Code Q{i}", height=300, key=f"code_{i}")
            stdin = st.text_area(f"Custom Input Q{i}", key=f"stdin_{i}")

            run_col, validate_col = st.columns(2)

            # --- RUN CODE ---
            with run_col:
                if st.button(f"▶ Run Code Q{i}", key=f"run_{i}"):
                    res = run_code(code, lang, stdin)
                    st.markdown(f"**Output (via {res.get('source')})**")
                    output = res.get("stdout")
                    if output and output.strip():
                        st.code(output)
                    else:
                        st.info("No output. Use print statements.")
                    if res.get("stderr"):
                        st.error(res["stderr"])
                    if res.get("compile_output"):
                        st.error(res["compile_output"])

            # --- TEST CASE VALIDATION ---
            with validate_col:
                test_cases = q.get("test_cases", [])
                st.caption("💡 Validator auto-strips hardcoded calls and injects stdin.")
                if st.button(f"✅ Validate Test Cases Q{i}", key=f"tc_{i}"):
                    if not code or not code.strip():
                        st.warning("Write your code first.")
                    elif not test_cases:
                        st.info("No test cases available.")
                    else:
                        with st.spinner("Running test cases..."):
                            tc_results = run_test_cases(code, lang, test_cases)
                        if not tc_results:
                            st.info("Could not parse test cases.")
                        else:
                            passed = sum(1 for r in tc_results if r["passed"])
                            st.markdown(f"**{passed}/{len(tc_results)} test cases passed**")
                            for tc_r in tc_results:
                                icon = "✅" if tc_r["passed"] else "❌"
                                with st.expander(f"{icon} Test Case {tc_r['case']}"):
                                    st.write(f"**Input:** `{tc_r['input']}`")
                                    st.write(f"**Expected:** `{tc_r['expected']}`")
                                    st.write(f"**Got:** `{tc_r['actual']}`")
                                    if tc_r["error"]:
                                        st.error(tc_r["error"])

            user_answers.append({"question": problem_text, "answer": code or ""})

    # ----------------------------
    # SYSTEM DESIGN
    # ----------------------------
    st.markdown("## System Design")
    sd = parsed.get("system_design", {})
    sd_title = sd.get("title", "No title")
    st.markdown(f"### {sd_title}")

    st.markdown("**Use Case**")
    st.write(sd.get("use_case", "Not provided"))

    st.markdown("**Functional Requirements**")
    for item in sd.get("functional_requirements", []):
        st.write(f"- {item}")

    st.markdown("**Non-Functional Requirements**")
    for item in sd.get("non_functional_requirements", []):
        st.write(f"- {item}")

    st.markdown("**Key Discussion Points**")
    for item in sd.get("key_discussion_points", []):
        st.write(f"- {item}")

    sd_answer = st.text_area("Your System Design Answer", key="sd_answer")
    user_answers.append({"question": f"Design a {sd_title}", "answer": sd_answer or ""})

    # ----------------------------
    # BEHAVIORAL
    # ----------------------------
    st.markdown("## Behavioral")
    behavioral_q = parsed.get("behavioral", {}).get("question", "")
    st.write(behavioral_q)
    beh_answer = st.text_area("Your Behavioral Answer", key="beh_answer")
    user_answers.append({"question": behavioral_q, "answer": beh_answer or ""})

    st.session_state.user_answers = user_answers

    # ----------------------------
    # EVALUATE
    # ----------------------------
    if st.button("Evaluate My Answers", type="primary"):
        with st.spinner("Evaluating..."):
            results = evaluate_interview(company, role, st.session_state.user_answers)
        st.session_state.eval_results = results
        try:
            save_session(company, role, results)
        except Exception as e:
            st.warning(f"Could not save session history: {e}")

    # ----------------------------
    # SHOW RESULTS
    # ----------------------------
    if st.session_state.eval_results:
        results = st.session_state.eval_results

        st.markdown("## Evaluation Results")

        all_scores = []
        for r in results:
            s = r.get("evaluation", {}).get("scores", {})
            if s:
                all_scores.append(sum(s.values()) / len(s))
        overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

        ov_col, radar_col = st.columns([1, 2])
        with ov_col:
            st.metric("Overall Score", f"{overall}/10")
            optimized_count = sum(
                1 for r in results
                if r.get("evaluation", {}).get("is_optimized", False)
            )
            st.metric("Optimized Answers", f"{optimized_count}/{len(results)}")

        with radar_col:
            fig = render_radar_chart(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        for i, r in enumerate(results, start=1):
            with st.expander(f"Feedback Q{i}"):
                evaluation = r.get("evaluation", {})
                scores = evaluation.get("scores", {})
                is_optimized = evaluation.get("is_optimized", False)

                cols = st.columns(len(scores))
                for col, (k, v) in zip(cols, scores.items()):
                    col.metric(k.replace("_", " ").capitalize(), f"{v}/10")

                st.markdown("---")

                if is_optimized:
                    st.success("✅ Optimized solution — great answer, no suggestions needed.")
                else:
                    strengths = evaluation.get("strengths", [])
                    weaknesses = evaluation.get("weaknesses", [])
                    optimized_approach = evaluation.get("optimized_approach", "")

                    if strengths:
                        st.markdown("### Strengths")
                        for s in strengths:
                            st.write("✅", s)

                    if weaknesses:
                        st.markdown("### Weaknesses")
                        for w in weaknesses:
                            st.write("⚠️", w)

                    if optimized_approach:
                        st.markdown("### Optimized Approach")
                        st.info(optimized_approach)

                    if not strengths and not weaknesses and not optimized_approach:
                        st.info("No detailed feedback available.")

        # ----------------------------
        # LEARNING PLAN
        # ----------------------------
        st.markdown("## Personalized Learning Plan")

        plan = generate_learning_path(
            results, role, company,
            job_description=job_description,
            jd_requirements=jd_requirements
        )

        if isinstance(plan, str):
            plan = plan.strip()
            if plan.startswith('"') and plan.endswith('"'):
                try:
                    plan = json.loads(plan)
                except Exception:
                    pass
            plan = plan.replace("\\n", "\n")

        st.markdown(plan)