import streamlit as st
from src.rag_pipeline import ask_rag


def chat_section():

    st.header("💬 Chat With Your Research Papers")

    # Check if papers exist
    if not st.session_state.get("papers"):

        st.info(
            "📤 Upload one or more research papers first."
        )
        return

    st.write(
        """
Ask questions about your uploaded papers.

Examples:
- Summarize this paper
- What is the main contribution?
- What methodology was used?
- What are the limitations?
- What future work is suggested?
- What biomarkers were identified?
- Compare the uploaded papers
"""
    )

    question = st.text_area(
        "Enter your question",
        height=120,
        placeholder="Example: What are the main findings of this paper?"
    )

    col1, col2 = st.columns([1, 5])

    with col1:

        ask = st.button(
            "🔍 Ask",
            use_container_width=True
        )

    if ask:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )
            return

        with st.spinner(
            "Searching relevant sections..."
        ):

            try:

                answer = ask_rag(question)

                st.success("Answer Generated")

                st.markdown("## 🧠 Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )