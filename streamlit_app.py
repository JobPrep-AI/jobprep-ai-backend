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
import os

# LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"]     = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = st.secrets.get("LANGCHAIN_PROJECT", "jobprep-ai")
os.environ["LANGCHAIN_ENDPOINT"]    = st.secrets.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

pipeline = InterviewPipeline()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

st.set_page_config(page_title="JobPrep AI", layout="wide")

DIFFICULTY_COLORS = {"Easy": "#00C851", "Medium": "#FF8800", "Hard": "#FF4444"}
TOTAL_SECONDS = 120 * 60


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
    # Stop timer if evaluation is done
    if st.session_state.get("evaluation_done"):
        st.markdown(
            f"<div style='text-align:center;padding:12px;background:#1e1e2e;"
            f"border-radius:8px;margin-bottom:8px'>"
            f"<span style='font-size:32px;font-weight:bold'>"
            f"✅ Done</span><br>"
            f"<small style='color:#888'>Interview Evaluated</small></div>",
            unsafe_allow_html=True
        )
        return

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
for key in ["interview_data", "user_answers", "interview_start_time",
            "eval_results", "agent_state", "user",
            "mode", "quick_question", "quick_answer", "quick_result",
            "learning_plan_visible", "evaluation_done", "current_page"]:
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


# -------------------------------
# QUICK PRACTICE PAGE
# -------------------------------
CODING_TOPICS = [
    "Arrays & Strings",
    "Trees & Graphs",
    "Dynamic Programming",
    "Stack & Queue",
    "Linked Lists",
    "Sorting & Searching",
    "Math & Numbers",
    "Hashing & Maps",
    "Matrix & Grid",
    "Recursion & Backtracking",
]

def show_quick_practice():
    st.title("Quick Practice")
    st.caption("Get comfortable with one question at a time. No timer, no pressure.")
    st.markdown("---")

    # Question type selector
    q_type = st.radio(
        "What do you want to practice?",
        ["Coding", "System Design", "Behavioral"],
        horizontal=True,
        key="quick_q_type"
    )

    topic = None
    if q_type == "Coding":
        topic = st.selectbox(
            "Select a topic",
            CODING_TOPICS,
            key="quick_topic"
        )

    if st.button("Generate Question", type="primary", key="quick_gen_btn"):
        with st.spinner("Generating question..."):
            from graphrag_pipeline import generate_single_question
            user = st.session_state.user
            q = generate_single_question(
                topic=topic or q_type,
                q_type=q_type.lower().replace(" ", "_"),
                company=user.get("target_company", ""),
                role=user.get("target_role", "")
            )
        if q:
            st.session_state.quick_question = q
            st.session_state.quick_answer = None
            st.session_state.quick_result = None
            st.rerun()
        else:
            st.error("Could not generate question. Please try again.")

    # Display question
    if st.session_state.quick_question:
        q = st.session_state.quick_question
        qt = q.get("q_type", "coding")

        st.markdown("---")

        if qt == "coding":
            diff = q.get("difficulty", "Medium")
            diff_color = DIFFICULTY_COLORS.get(diff, "#888")
            st.markdown(
                f'<span style="background:{diff_color};color:white;padding:2px 10px;'
                f'border-radius:12px;font-size:13px;font-weight:600">{diff}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"### {q.get('title', '')}")
            st.write(q.get("problem_statement", ""))
            if q.get("example_input_output"):
                st.markdown("**Example**")
                st.code(q.get("example_input_output"))
            if q.get("constraints"):
                st.caption(f"Constraints: {q.get('constraints')}")

            lang = st.selectbox("Language", ["python", "java", "cpp"], key="quick_lang")
            starter = q.get("starter_code", {}).get(lang, "")
            if "quick_code" not in st.session_state or not st.session_state.quick_code:
                st.session_state.quick_code = starter
            code = st.text_area("Write your solution", height=300, key="quick_code")

            run_col, submit_col = st.columns(2)
            with run_col:
                if st.button("▶ Run Code", key="quick_run"):
                    res = run_code(code, lang, "")
                    output = res.get("stdout")
                    if output and output.strip():
                        st.code(output)
                    else:
                        st.info("No output.")
                    if res.get("stderr"):
                        st.error(res["stderr"])

            with submit_col:
                if st.button("Submit for Review", type="primary", key="quick_submit"):
                    if not code or not code.strip():
                        st.warning("Write your solution first.")
                    else:
                        with st.spinner("Evaluating..."):
                            from evaluation_pipeline import evaluate_single_answer
                            result = evaluate_single_answer(
                                question=q.get("problem_statement", ""),
                                answer=code,
                                q_type="coding"
                            )
                        st.session_state.quick_result = result
                        st.rerun()

        elif qt == "system_design":
            st.markdown(f"### {q.get('title', 'System Design')}")
            st.markdown("**Use Case**")
            st.write(q.get("use_case", ""))
            st.markdown("**Functional Requirements**")
            for item in q.get("functional_requirements", []):
                st.write(f"- {item}")
            st.markdown("**Non-Functional Requirements**")
            for item in q.get("non_functional_requirements", []):
                st.write(f"- {item}")
            st.markdown("**Key Discussion Points**")
            for item in q.get("key_discussion_points", []):
                st.write(f"- {item}")

            answer = st.text_area("Your answer", height=200, key="quick_sd_answer")
            if st.button("Submit for Review", type="primary", key="quick_sd_submit"):
                if not answer.strip():
                    st.warning("Write your answer first.")
                else:
                    with st.spinner("Evaluating..."):
                        from evaluation_pipeline import evaluate_single_answer
                        result = evaluate_single_answer(
                            question=f"Design {q.get('title', '')}",
                            answer=answer,
                            q_type="system_design"
                        )
                    st.session_state.quick_result = result
                    st.rerun()

        elif qt == "behavioral":
            st.markdown("### Behavioral Question")
            st.write(q.get("question", ""))
            answer = st.text_area("Your answer", height=200, key="quick_beh_answer")
            if st.button("Submit for Review", type="primary", key="quick_beh_submit"):
                if not answer.strip():
                    st.warning("Write your answer first.")
                else:
                    with st.spinner("Evaluating..."):
                        from evaluation_pipeline import evaluate_single_answer
                        result = evaluate_single_answer(
                            question=q.get("question", ""),
                            answer=answer,
                            q_type="behavioral"
                        )
                    st.session_state.quick_result = result
                    st.rerun()

        # Show result
        if st.session_state.quick_result:
            result = st.session_state.quick_result
            st.markdown("---")
            st.markdown("## Feedback")

            scores = result.get("scores", {})
            is_optimized = result.get("is_optimized", False)

            cols = st.columns(len(scores))
            for col, (k, v) in zip(cols, scores.items()):
                col.metric(k.replace("_", " ").capitalize(), f"{v}/10")

            st.markdown("---")

            if is_optimized:
                st.success("✅ Great answer!")
            else:
                strengths = result.get("strengths", [])
                weaknesses = result.get("weaknesses", [])
                approach = result.get("optimized_approach", "")

                if strengths:
                    st.markdown("### Strengths")
                    for s in strengths:
                        st.write("✅", s)
                if weaknesses:
                    st.markdown("### Weaknesses")
                    for w in weaknesses:
                        st.write("⚠️", w)
                if approach:
                    st.markdown("### Better Approach")
                    st.info(approach)

            if st.button("Try Another Question", key="quick_retry"):
                st.session_state.quick_question = None
                st.session_state.quick_result = None
                st.session_state.quick_code = None
                st.rerun()


# -------------------------------
# LEARNING PLAN PAGE
# -------------------------------
def show_learning_plan_page():
    agent_state = st.session_state.get("agent_state")
    plan = agent_state.learning_plan if agent_state else None

    # Back button
    if st.button("← Back to Results", key="back_to_results"):
        st.session_state.current_page = "main"
        st.rerun()

    if not plan or not isinstance(plan, dict):
        st.warning("Learning plan not available. Please complete an interview first.")
        return

    role    = plan.get("role", "")
    company = plan.get("company", "")
    scored_gaps  = plan.get("scored_gaps", [])
    not_assessed = plan.get("not_assessed", [])
    days         = plan.get("days", [])

    # ── PAGE HEADER ──────────────────────────────────────────────
    st.markdown(f"## 📚 Personalized Learning Plan")
    st.markdown(f"**{role}** at **{company}** · 14-day preparation schedule")
    st.markdown("---")

    # ── GAP ANALYSIS SUMMARY ─────────────────────────────────────
    st.markdown("### 🔍 Gap Analysis")

    col1, col2, col3 = st.columns(3)
    critical_count = sum(1 for g in scored_gaps if g["level"] == "critical")
    medium_count   = sum(1 for g in scored_gaps if g["level"] == "medium")
    light_count    = sum(1 for g in scored_gaps if g["level"] == "light")

    col1.metric("🔴 Critical Gaps", critical_count)
    col2.metric("🟠 Medium Gaps",   medium_count)
    col3.metric("🟡 Light Gaps",    light_count)

    st.markdown("---")

    # ── SCORED GAPS TABLE ─────────────────────────────────────────
    if scored_gaps:
        st.markdown("#### Your Performance Gaps")
        st.caption("Ranked by priority = severity × role importance")

        for gap in scored_gaps:
            color_map = {"critical": "#FF4444", "medium": "#FF8800", "light": "#FFC107"}
            bg_color  = color_map.get(gap["level"], "#888")

            with st.container():
                g_col1, g_col2, g_col3, g_col4 = st.columns([3, 1, 1, 1])
                g_col1.markdown(
                    f"{gap['color']} **{gap['topic'].title()}**"
                )
                g_col2.markdown(
                    f"<span style='color:{bg_color};font-weight:600'>"
                    f"{gap['level'].upper()}</span>",
                    unsafe_allow_html=True
                )
                g_col3.markdown(f"Score: **{gap['avg_score']}/10**")
                g_col4.markdown(
                    f"Priority: **{gap['priority_score']}**"
                )

    # ── NOT ASSESSED ─────────────────────────────────────────────
    if not_assessed:
        st.markdown("---")
        st.markdown("#### ⚪ Not Assessed in This Interview")
        st.caption(
            "These JD skills were not covered in your interview. "
            "Not penalized — included in your plan for completeness."
        )
        cols = st.columns(min(len(not_assessed), 4))
        for i, skill in enumerate(not_assessed):
            cols[i % 4].markdown(
                f"<div style='background:#2d2d2d;border-radius:8px;"
                f"padding:6px 12px;margin:4px;text-align:center;"
                f"font-size:13px;color:#ccc'>{skill}</div>",
                unsafe_allow_html=True
            )

    # ── SPACED REPETITION EXPLANATION ────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 14-Day Schedule")
    st.caption(
        "Critical gaps appear on Days 1, 4, 8, 13 — spaced repetition "
        "for maximum retention. Medium gaps on Days 2, 6, 11. "
        "Not-assessed skills fill remaining days."
    )

    # ── DAY CARDS ────────────────────────────────────────────────
    level_colors = {
        "critical":     "#FF4444",
        "medium":       "#FF8800",
        "light":        "#FFC107",
        "not_assessed": "#888888",
        "general review": "#888888"
    }
    level_bg = {
        "critical":     "#2d1a1a",
        "medium":       "#2d1f0e",
        "light":        "#2d2a0e",
        "not_assessed": "#1e1e2e",
        "general review": "#1e1e2e"
    }

    # Show days in groups of 7
    week1 = [d for d in days if d["day"] <= 7]
    week2 = [d for d in days if d["day"] > 7]

    for week_label, week_days in [("Week 1", week1), ("Week 2", week2)]:
        st.markdown(f"#### {week_label}")
        for d in week_days:
            level      = d.get("level", "light")
            bar_color  = level_colors.get(level, "#888")
            revisit_tag = " 🔄 Revisit" if d.get("is_revisit") else ""
            score_tag   = f" · scored {d['avg_score']}/10" if d.get("avg_score") else ""

            with st.expander(
                f"Day {d['day']} — {d['topic'].title()}{revisit_tag}{score_tag}",
                expanded=False
            ):
                # Colored top bar
                st.markdown(
                    f"<div style='height:4px;background:{bar_color};"
                    f"border-radius:2px;margin-bottom:12px'></div>",
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**🎯 Why It Matters**")
                    st.caption(
                        d.get("why_it_matters") or
                        f"Important for {role} at {company}."
                    )

                with c2:
                    st.markdown("**📖 Focus**")
                    st.caption(
                        d.get("focus") or
                        f"Study {d['topic']} concepts."
                    )

                with c3:
                    st.markdown("**✍️ Practice**")
                    st.caption(
                        d.get("practice") or
                        f"Practice {d['topic']} problems."
                    )

                # Priority badge for non-not-assessed
                if d.get("priority_score"):
                    st.markdown(
                        f"<div style='text-align:right'>"
                        f"<span style='background:{bar_color}20;"
                        f"color:{bar_color};border:1px solid {bar_color};"
                        f"border-radius:12px;padding:2px 10px;"
                        f"font-size:11px;font-weight:600'>"
                        f"{level.upper()} · priority {d['priority_score']}"
                        f"</span></div>",
                        unsafe_allow_html=True
                    )

        st.markdown("---")

    # ── FOOTER ───────────────────────────────────────────────────
    st.success(
        f"Complete this 14-day plan to significantly improve your "
        f"{role} interview performance at {company}. "
        f"Focus on critical gaps first — they have the highest impact."
    )


# Gate: show login page if not logged in
if not st.session_state.user:
    show_auth_page()
    st.stop()

# Page routing
if st.session_state.current_page == "learning_plan":
    show_learning_plan_page()
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

# Mode selector
mode = st.radio(
    "Select Mode",
    ["Quick Practice", "Deep Interview"],
    horizontal=True,
    key="selected_mode"
)
st.session_state.mode = mode
st.markdown("---")

# Quick Practice — separate flow
if mode == "Quick Practice":
    show_quick_practice()
    st.stop()

# Deep Interview — full flow below
st.caption("4 coding questions + 2 system design + 1 behavioral · 120 minutes · Personalized")

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
                user_id=st.session_state.user["user_id"],
                mode="deep"
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
    # SYSTEM DESIGN (2 questions)
    # ----------------------------
    st.markdown("## System Design")
    sd_questions = parsed.get("system_design_questions", [])

    for sd_i, sd in enumerate(sd_questions, start=1):
        sd_title = sd.get("title", f"System Design {sd_i}")
        st.markdown(f"### SD{sd_i}: {sd_title}")

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

        sd_answer = st.text_area(
            f"Your Answer for SD{sd_i}",
            key=f"sd_answer_{sd_i}"
        )
        user_answers.append({
            "question": f"Design a {sd_title}",
            "answer": sd_answer or ""
        })
        st.markdown("---")

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
    col_eval, col_new = st.columns([2, 1])

    with col_eval:
        if not st.session_state.evaluation_done:
            if st.button("Evaluate My Answers", type="primary"):
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
                            agent_state.user_answers = st.session_state.user_answers or []
                            agent_state = pipeline.evaluate(agent_state, st.session_state.user_answers)
                            st.session_state.agent_state = agent_state
                            results = agent_state.eval_results
                        else:
                            from evaluation_pipeline import evaluate_interview
                            results = evaluate_interview(company, role, st.session_state.user_answers)

                    st.session_state.eval_results = results
                    st.session_state.evaluation_done = True
                    st.session_state.learning_plan_visible = False

                    if agent_state:
                        with st.expander("🤖 Evaluation Agent Logs", expanded=False):
                            for log in agent_state.agent_logs:
                                st.caption(log)

                    try:
                        save_session(company, role, results)
                    except Exception as e:
                        st.warning(f"Could not save session history: {e}")

                    st.rerun()

    with col_new:
        if st.button("🔄 New Attempt", key="new_attempt_btn"):
            for key in ["interview_data", "user_answers", "interview_start_time",
                        "eval_results", "agent_state", "evaluation_done",
                        "learning_plan_visible", "current_page"]:
                st.session_state[key] = None
            for k in list(st.session_state.keys()):
                if k.startswith("hints_") or k.startswith("code_") or \
                   k.startswith("lang_") or k.startswith("stdin_") or \
                   k.startswith("sd_answer") or k in ["beh_answer"]:
                    del st.session_state[k]
            st.rerun()

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
        st.markdown("---")

        lp_col, _ = st.columns([2, 2])
        with lp_col:
            if st.button("📚 View Personalized Learning Plan",
                         type="secondary", key="show_lp_btn"):
                st.session_state.current_page = "learning_plan"
                st.rerun()
        
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