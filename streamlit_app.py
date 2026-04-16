import json
import time
import plotly.graph_objects as go
import streamlit as st
from graphrag_pipeline import coerce_interview_json
from agents import InterviewPipeline, AgentState
from evaluation_pipeline import evaluate_interview, generate_learning_path, generate_hint
from code_executor import run_code, run_test_cases, reverify_with_user_code
from history_tracker import save_session, load_recent_sessions
from auth import login_user, register_user, update_profile
from user_profile import load_user_sessions, get_score_trend
import logging

pipeline = InterviewPipeline()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

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


def render_score_trend(user_id: str, role: str, company: str):
    """Show score trend across attempts for this role+company."""
    trend = get_score_trend(user_id, role, company)

    if len(trend) < 2:
        return None

    attempts = [f"Attempt {t['attempt']}" for t in trend]
    scores   = [t["avg_score"] for t in trend]
    dates    = [t["date"] for t in trend]

    fig = go.Figure()

    # Score line
    fig.add_trace(go.Scatter(
        x=attempts, y=scores,
        mode="lines+markers+text",
        text=[f"{s}/10" for s in scores],
        textposition="top center",
        line=dict(color="rgba(99,110,250,0.9)", width=2.5),
        marker=dict(size=10, color="rgba(99,110,250,0.9)"),
        name="Avg Score",
        hovertemplate="<b>%{x}</b><br>Score: %{y}/10<br>Date: " +
                      "<br>".join(dates) + "<extra></extra>"
    ))

    # Target line at 8
    fig.add_hline(
        y=8, line_dash="dash",
        line_color="rgba(0,200,80,0.5)",
        annotation_text="Target (8/10)",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"Score trend — {role} at {company}",
        xaxis_title="Attempt",
        yaxis=dict(range=[0, 10], title="Avg Score"),
        height=300,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    return fig

def render_timer(start_time):
    remaining = max(0, TOTAL_SECONDS - int(time.time() - start_time))
    mins, secs = remaining // 60, remaining % 60

    if remaining <= 0:
        st.error("⏰ Time's up!")
        return

    color = "🔴" if remaining < 300 else "🟠" if remaining < 900 else "🟢"
    st.markdown(
        f"<div style='text-align:center;padding:12px;background:#1e1e2e;"
        f"border-radius:8px;margin-bottom:8px'>"
        f"<span style='font-size:32px;font-weight:bold'>"
        f"{color} {mins:02d}:{secs:02d}</span><br>"
        f"<small style='color:#888'>Time Remaining</small></div>",
        unsafe_allow_html=True
    )


# -------------------------------
# SESSION STATE
# -------------------------------
for key in ["interview_data", "user_answers", "interview_start_time", "eval_results", "agent_state", "user"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# LOGIN / REGISTER PAGE
# -------------------------------
def show_auth_page():
    st.title("JobPrep AI")
    st.markdown("#### Your personalized mock interview platform")
    st.markdown("---")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        st.markdown("### Welcome back")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", key="login_btn"):
            if not email or not password:
                st.warning("Please enter email and password.")
            else:
                result = login_user(email, password)
                if result["success"]:
                    st.session_state.user = result["user"]
                    st.success(f"Welcome back, {result['user']['name']}!")
                    st.rerun()
                else:
                    st.error(result["message"])

    with tab_register:
        st.markdown("### Create your account")
        name          = st.text_input("Full name", key="reg_name")
        email_r       = st.text_input("Email", key="reg_email")
        password_r    = st.text_input("Password", type="password", key="reg_password")
        target_role   = st.text_input("Target role (e.g. Software Engineer)", key="reg_role")
        target_company= st.text_input("Target company (e.g. Google)", key="reg_company")

        if st.button("Register", type="primary", key="register_btn"):
            if not name or not email_r or not password_r:
                st.warning("Name, email and password are required.")
            else:
                result = register_user(name, email_r, password_r,
                                       target_role, target_company)
                if result["success"]:
                    # Auto-login after register
                    login_result = login_user(email_r, password_r)
                    if login_result["success"]:
                        st.session_state.user = login_result["user"]
                        st.success("Account created! Welcome to JobPrep AI.")
                        st.rerun()
                else:
                    st.error(result["message"])


# Gate: show login page if not logged in
if not st.session_state.user:
    show_auth_page()
    st.stop()

# -------------------------------
# SIDEBAR — Timer + History + Profile
# -------------------------------
@st.fragment(run_every=5)
def timer_fragment():
    st.markdown("## ⏱️ Interview Timer")
    if st.session_state.interview_start_time:
        render_timer(st.session_state.interview_start_time)
    else:
        st.info("Timer starts when you generate an interview.")

with st.sidebar:
    # User profile
    user = st.session_state.user
    st.markdown(f"### 👤 {user['name']}")
    st.caption(user["email"])
    if user.get("target_role"):
        st.caption(f"Target: {user['target_role']} @ {user.get('target_company','')}")

    if st.button("Logout", key="logout_btn"):
        for key in ["interview_data", "user_answers", "interview_start_time",
                    "eval_results", "agent_state", "user"]:
            st.session_state[key] = None
        st.rerun()

    st.markdown("---")
    timer_fragment()

    st.markdown("---")
    st.markdown("## 📋 My Sessions")
    try:
        sessions = load_user_sessions(user["user_id"], limit=5)
        if sessions:
            for s in sessions:
                with st.expander(f"#{s['attempt_number']} {s['company']} — {s['role'][:18]}"):
                    st.caption(s["created_at"])
                    col1, col2 = st.columns(2)
                    col1.metric("Avg Score", f"{s['avg_score']}/10")
                    col2.metric("Optimized", f"{s['optimized']}/{s['total']}")
                    if s["weak_areas"]:
                        st.caption("Weak: " + ", ".join(s["weak_areas"][:3]))
        else:
            st.caption("No sessions yet. Generate your first interview!")
    except Exception as e:
        st.caption(f"History unavailable: {e}")


# -------------------------------
# MAIN
# -------------------------------
st.title("JobPrep AI — Mock Interview Generator")

user = st.session_state.user
default_company = user.get("target_company") or "Google"
default_role    = user.get("target_role")    or "Software Engineer"
company = st.text_input("Company Name", default_company)
role    = st.text_input("Role Name",    default_role)
job_description = st.text_area("Job Description", height=220)

# -------------------------------
# GENERATE INTERVIEW
# -------------------------------
if st.button("Generate Mock Interview", type="primary"):
    if not company or not role or not job_description.strip():
        st.warning("Please provide all inputs.")
    else:
        with st.spinner("Generating your interview..."):
            state = pipeline.generate_interview(
                company, role, job_description,
                user_id=st.session_state.user["user_id"]
            )

        # Check if pipeline was stopped by guardrail
        if state.errors:
            for error in state.errors:
                st.error(f"❌ {error}")
            st.stop()

        # Show agent logs in expander
        with st.expander("🤖 Agent Pipeline Logs", expanded=False):
            for log in state.agent_logs:
                st.caption(log)

        # Show reflection result
        if not state.reflection_passed:
            st.warning(
                f"⚠️ Reflection Agent flagged some issues after "
                f"{state.reflection_attempts} attempt(s) — "
                f"showing best available interview."
            )
        else:
            st.success(
                f"✅ Reflection Agent approved the interview "
                f"on attempt {state.reflection_attempts}."
            )

        # Store state
        st.session_state.agent_state = state
        st.session_state.interview_data = (
            state.top_clusters,
            state.selected_questions,
            state.interview_raw,
            state.jd_requirements,
            state.missing_requirements
        )
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
    
    # Use already parsed interview from agent state if available
    agent_state = st.session_state.get("agent_state")
    if agent_state and agent_state.interview_parsed:
        parsed = agent_state.interview_parsed
    else:
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

            starter_code = q.get("starter_code", {})
            starter = starter_code.get(lang, "") if isinstance(starter_code, dict) else ""

            code_key = f"code_{i}"
            lang_key = f"lang_prev_{i}"

            # Pre-fill when first loaded OR when language is switched
            if code_key not in st.session_state or not st.session_state[code_key]:
                # First load — pre-fill with starter
                if starter:
                    st.session_state[code_key] = starter
            elif st.session_state.get(lang_key) != lang:
                # Language switched — load new starter code
                if starter:
                    st.session_state[code_key] = starter

            # Track current language to detect switches
            st.session_state[lang_key] = lang

            code = st.text_area(f"Write Code Q{i}", height=300, key=code_key)
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
                            corrected = reverify_with_user_code(code, lang, test_cases)
                            tc_results = run_test_cases(code, lang, corrected)
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
                                    if not tc_r["passed"] and tc_r["actual"]:
                                        st.warning(
                                            "⚠️ If your logic is correct, the expected output "
                                            "may have been incorrectly generated by the LLM. "
                                            "Verify manually before assuming your code is wrong."
                                        )

            user_answers.append({
                "question": problem_text,
                "answer": code or "",
                "lang": lang,
                "test_cases": test_cases
            })

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
        # Validate answers before evaluation
        from guardrails import guardrails
        invalid_answers = []
        for idx, qa in enumerate(st.session_state.user_answers or []):
            answer = qa.get("answer", "")
            q_type = qa.get("lang")
            if q_type in ["python", "java", "cpp"]:
                result = guardrails.answer.validate_coding_answer(answer)
            else:
                result = guardrails.answer.validate_text_answer(answer)
            if not result.passed:
                invalid_answers.append(f"Q{idx+1}: {result.reason}")

        if invalid_answers:
            st.warning(
                "⚠️ Some answers need attention before evaluating:\n" +
                "\n".join(f"- {a}" for a in invalid_answers)
            )
        else:
            with st.spinner("Evaluating answers using multi-agent pipeline..."):
                agent_state = st.session_state.get("agent_state")
                if agent_state:
                    agent_state = pipeline.evaluate(agent_state, st.session_state.user_answers)
                    st.session_state.agent_state = agent_state
                    results = agent_state.eval_results
                else:
                    from evaluation_pipeline import evaluate_interview
                    results = evaluate_interview(company, role, st.session_state.user_answers)

            st.session_state.eval_results = results

            # Show evaluation agent logs
            if agent_state:
                with st.expander("🤖 Evaluation Agent Logs", expanded=False):
                    for log in agent_state.agent_logs:
                        st.caption(log)

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
                st.plotly_chart(fig, width='stretch')

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

        agent_state = st.session_state.get("agent_state")
        plan = agent_state.learning_plan if agent_state and agent_state.learning_plan else ""

        if not plan or not isinstance(plan, str) or not plan.strip():
            st.warning("Learning plan could not be generated. Please try evaluating again.")
        else:
            plan = plan.strip()
            if plan.startswith('"') and plan.endswith('"'):
                try:
                    plan = json.loads(plan)
                except Exception:
                    pass
            if isinstance(plan, str):
                plan = plan.replace("\\n", "\n")
                st.markdown(plan)
            else:
                st.warning("Learning plan returned unexpected format.")
        
        # ----------------------------
        # SESSION COMPARISON
        # ----------------------------
        user = st.session_state.user
        trend = get_score_trend(user["user_id"], role, company)

        if len(trend) >= 2:
            st.markdown("## Your Progress")
            trend_fig = render_score_trend(user["user_id"], role, company)
            if trend_fig:
                st.plotly_chart(trend_fig, width='stretch')

            # Delta from last attempt
            latest  = trend[-1]
            previous = trend[-2]
            delta   = round(latest["avg_score"] - previous["avg_score"], 1)
            delta_str = f"+{delta}" if delta >= 0 else str(delta)
            delta_color = "green" if delta >= 0 else "red"

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "This attempt",
                f"{latest['avg_score']}/10",
                f"{delta_str} vs last"
            )
            col2.metric(
                "Best score",
                f"{max(t['avg_score'] for t in trend)}/10"
            )
            col3.metric(
                "Total attempts",
                f"{len(trend)}"
            )

            # Show weak areas that improved
            prev_weak = set(previous.get("weak_areas", []))
            curr_weak = set(latest.get("weak_areas",  []))
            improved  = prev_weak - curr_weak
            still_weak = prev_weak & curr_weak

            if improved:
                st.success(
                    "✅ Improved since last attempt: " +
                    ", ".join(improved)
                )
            if still_weak:
                st.warning(
                    "⚠️ Still needs work: " +
                    ", ".join(still_weak)
                )
        elif len(trend) == 1:
            st.info(
                "Complete one more interview for this role to see your progress trend."
            )