import streamlit as st
from vector_rag_pipeline import parse_interview_json, run_vector_rag_interview

st.set_page_config(page_title="JobPrep AI - Vector RAG", layout="wide")
st.title("JobPrep AI Mock Interview Generator - Vector RAG")

company = st.text_input("Company Name", "Google")
role = st.text_input("Role Name", "Software Engineer")
job_description = st.text_area(
    "Job Description",
    height=220,
    placeholder="Paste the job description here..."
)

if st.button("Generate Mock Interview", type="primary"):
    if not company or not role or not job_description.strip():
        st.warning("Please provide company, role, and job description.")
    else:
        with st.spinner("Generating mock interview with Vector RAG..."):
            retrieved_df, interview = run_vector_rag_interview(
                company=company,
                role=role,
                job_description=job_description
            )

        parsed = parse_interview_json(interview)

        st.subheader("Retrieved Questions")
        for q in retrieved_df["interview_question"].tolist():
            st.write(f"- {q}")

        st.subheader("Generated Mock Interview")

        if parsed is None:
            st.markdown(interview)
        else:
            st.markdown("## Coding Questions")
            for i, q in enumerate(parsed["coding_questions"], start=1):
                with st.expander(f"Coding Question {i}: {q.get('title', 'Untitled')}", expanded=True):
                    st.markdown("**Problem Statement**")
                    st.write(q.get("problem_statement", ""))

                    st.markdown("**Example Input / Output**")
                    st.code(q.get("example_input_output", ""))

                    st.markdown("**Constraints**")
                    st.write(q.get("constraints", ""))

                    st.markdown("**Test Cases**")
                    for tc in q.get("test_cases", []):
                        st.code(tc)

            sd = parsed["system_design"]
            st.markdown("## System Design")
            with st.expander(f"System Design: {sd.get('title', 'Untitled')}", expanded=True):
                st.markdown("**Use Case**")
                st.write(sd.get("use_case", ""))

                st.markdown("**Functional Requirements**")
                for item in sd.get("functional_requirements", []):
                    st.write(f"- {item}")

                st.markdown("**Non-Functional Requirements**")
                for item in sd.get("non_functional_requirements", []):
                    st.write(f"- {item}")

                st.markdown("**Key Discussion Points**")
                for item in sd.get("key_discussion_points", []):
                    st.write(f"- {item}")

            st.markdown("## Behavioral Question")
            st.write(parsed["behavioral"].get("question", ""))
