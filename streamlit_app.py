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

    # only STRICT unwanted types
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

    _, _, interview, _, _ = st.session_state.interview_data

    parsed = coerce_interview_json(interview)

    if not isinstance(parsed, dict):
        st.error("Invalid interview format. Please regenerate.")
        st.stop()

    # 🔥 DEBUG START (ADD HERE)
    raw_questions = parsed.get("coding_questions", [])

    st.write("🔍 Total coding questions:", len(raw_questions))

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

    st.write("🔍 After filtering:", len(filtered_questions))

    # 🔥 fallback if empty
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

            # LANGUAGE
            lang = st.selectbox(
                f"Language Q{i}",
                ["python", "java", "cpp"],
                key=f"lang_{i}"
            )

            # CODE INPUT
            code = st.text_area(
                f"Write Code Q{i}",
                height=300,
                key=f"code_{i}"
            )

            stdin = st.text_area(
                f"Custom Input Q{i}",
                key=f"stdin_{i}"
            )

            # RUN CODE
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

    st.markdown(f"### {sd.get('title', 'No title')}")

    # ✅ USE CASE
    st.markdown("**Use Case**")
    st.write(sd.get("use_case", "Not provided"))

    # ✅ FUNCTIONAL REQUIREMENTS
    st.markdown("**Functional Requirements**")
    for item in sd.get("functional_requirements", []):
        st.write(f"- {item}")

    # ✅ NON-FUNCTIONAL REQUIREMENTS
    st.markdown("**Non-Functional Requirements**")
    for item in sd.get("non_functional_requirements", []):
        st.write(f"- {item}")

    # ✅ DISCUSSION POINTS
    st.markdown("**Key Discussion Points**")
    for item in sd.get("key_discussion_points", []):
        st.write(f"- {item}")

    # ANSWER BOX
    sd_answer = st.text_area("Your System Design Answer")

    # -------------------------------
    # BEHAVIORAL
    # -------------------------------
    st.markdown("## Behavioral")

    behavioral_q = parsed.get("behavioral", {}).get("question", "")
    st.write(behavioral_q)

    beh_answer = st.text_area("Your Behavioral Answer")

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

                scores = r.get("evaluation", {}).get("scores", {})
                st.json(scores)

                st.markdown("### Strengths")
                for s in r.get("evaluation", {}).get("strengths", []):
                    st.write("✅", s)

                st.markdown("### Weaknesses")
                for w in r.get("evaluation", {}).get("weaknesses", []):
                    st.write("⚠️", w)

                st.markdown("### Missing Concepts")
                for m in r.get("evaluation", {}).get("missing_concepts", []):
                    st.write("❌", m)

        # -------------------------------
        # LEARNING PLAN
        # -------------------------------
        st.markdown("## Personalized Learning Plan")

        plan = generate_learning_path(
            results,
            role,
            company
        )

        st.write(plan)