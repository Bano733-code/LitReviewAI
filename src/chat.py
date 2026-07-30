import streamlit as st

from src.rag_pipeline import ask_rag


def chat_section():

    st.header("💬 Chat With Papers")


    question = st.text_area(
        "Ask your question about uploaded papers"
    )


    if st.button("Ask"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            return


        with st.spinner(
            "Searching papers..."
        ):

            answer = ask_rag(
                question
            )


        st.subheader(
            "🧠 Answer"
        )

        st.write(answer)