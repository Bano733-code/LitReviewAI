import streamlit as st
from src.config import client
def chat_section():

    st.header("💬 Chat With Papers")

    context = "\n".join(
        [
            p["abstract"]
            for p in st.session_state.papers
        ]
    )


    question = st.text_area(
        "Ask your question"
    )


    if st.button("Ask"):

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role":"user",
                    "content":
                    f"""
                    Context:
                    {context}

                    Question:
                    {question}
                    """
                }
            ]
        )


        st.write(
            response.choices[0].message.content
        )