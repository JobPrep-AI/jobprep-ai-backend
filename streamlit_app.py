import json
import streamlit as st
from graphrag_pipeline import coerce_interview_json, run_graphrag_interview
from evaluation_pipeline import evaluate_interview, generate_learning_path
from code_executor import run_code

st.set_page_config(page_title="JobPrep AI", layout="wide")
st.title("JobPrep AI Mock Interview Generator")

company = st.text_input("Company Name", "Google")
role = st.text_input("Role Name", "Software Engineer")
job_description = st.text_area("Job Description", height=220)

# -------------------------------
# SESSION STATE
# -------------------------------
if "interview_data" not in st.session_state:
    st.session_state.interview_data = None

if "user_answers" not in st.session_state:
    st.session_state.user_answers = []

# -------------------------------
# HELPER: FILTER DSA QUESTIONS
# -------------------------------
def is_valid_dsa(q):
    text = q.lower()
    bad = ["sql query", "database query", "table", "join"]
    return not any(b in text for b in bad)

# -------------------------------
# GENERATE INTERVIEW
# -------------------------------
if st.button("Generate Mock Interview"):
    if not company or not role or not job_description.strip():
        st.warning("Please provide all inputs.")
    else:
        with st.spinner("Generating..."):
            result = run_graphrag_interview(company, role, job_description)
        st.session_state.interview_data = result
        st.session_state.user_answers = []

# -------------------------------
# DISPLAY INTERVIEW
# -------------------------------
if st.session_state.interview_data:

    _, _, interview, jd_requirements, _ = st.session_state.interview_data

    parsed = coerce_interview_json(interview)

    if not isinstance(parsed, dict):
        st.error("Invalid interview format. Please regenerate.")
        st.stop()

    # -------------------------------
    # CODING QUESTIONS
    # -------------------------------
    st.markdown("## Coding Questions")

    user_answers = []

    raw_questions = parsed.get("coding_questions", [])
    filtered_questions = [
        q for q in raw_questions
        if is_valid_dsa(q.get("problem_statement", ""))
    ]
    coding_questions = filtered_questions if filtered_questions else raw_questions

    for i, q in enumerate(coding_questions, start=1):
        with st.expander(f"Q{i}: {q.get('title', 'Untitled')}"):

            st.markdown("### Problem")
            problem_text = q.get("problem_statement") or q.get("title") or "No question text available"
            st.write(problem_text)

            st.markdown("### Example")
            example = q.get("example_input_output")
            if example:
                st.code(example)
            else:
                st.info("No example provided")

            lang = st.selectbox(
                f"Language Q{i}",
                ["python", "java", "cpp"],
                key=f"lang_{i}"
            )

            code = st.text_area(
                f"Write Code Q{i}",
                height=300,
                key=f"code_{i}"
            )

            stdin = st.text_area(
                f"Custom Input Q{i}",
                key=f"stdin_{i}"
            )

            if st.button(f"Run Code Q{i}"):
                result = run_code(code, lang, stdin)

                st.markdown(f"### Output (via **{result.get('source')}**)")
                output = result.get("stdout")

                if output and output.strip():
                    st.code(output)
                else:
                    st.info("No output produced. Use print statements.")

                if result.get("stderr"):
                    st.error(result["stderr"])

                if result.get("compile_output"):
                    st.error(result["compile_output"])

            user_answers.append({
                "question": problem_text,
                "answer": code or ""
            })

    # -------------------------------
    # SYSTEM DESIGN
    # -------------------------------
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

    user_answers.append({
        "question": f"Design a {sd_title}",
        "answer": sd_answer or ""
    })

    # -------------------------------
    # BEHAVIORAL
    # -------------------------------
    st.markdown("## Behavioral")

    behavioral_q = parsed.get("behavioral", {}).get("question", "")
    st.write(behavioral_q)

    beh_answer = st.text_area("Your Behavioral Answer", key="beh_answer")

    user_answers.append({
        "question": behavioral_q,
        "answer": beh_answer or ""
    })

    # SAVE TO SESSION
    st.session_state.user_answers = user_answers

    # -------------------------------
    # EVALUATE
    # -------------------------------
    if st.button("Evaluate My Answers"):

        with st.spinner("Evaluating..."):
            results = evaluate_interview(
                company,
                role,
                st.session_state.user_answers
            )

        st.markdown("## Evaluation Results")

        for i, r in enumerate(results, start=1):
            with st.expander(f"Feedback Q{i}"):

                evaluation = r.get("evaluation", {})
                scores = evaluation.get("scores", {})
                is_optimized = evaluation.get("is_optimized", False)
                q_type = evaluation.get("question_type", "coding")

                # Score metrics — dynamic column count based on number of metrics
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

        # -------------------------------
        # LEARNING PLAN
        # -------------------------------
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