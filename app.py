import streamlit as st

from components.upload import upload_section
from components.summaries import summary_section
from components.insights import insights_section
from src.embeddings import extract_keywords
from src.topic_modeling import lda_topic_modeling
from src.visualizations import generate_wordcloud


st.set_page_config(
    page_title="LitReviewAI",
    page_icon="📚",
    layout="wide"
)


st.title("📚 LitReviewAI: Automated Research Paper Reviewer")


if "papers" not in st.session_state:
    st.session_state.papers = []


tabs = st.tabs([
    "About",
    "Upload",
    "Summaries",
    "Topics",
    "Insights",
    "Chat"
])


with tabs[0]:
    st.header("About LitReviewAI")

    st.write("""
    AI-powered research assistant for:
    
    - Paper summarization
    - Research gap detection
    - Literature organization
    - Topic discovery
    - Citation export
    - Paper Q&A
    """)


with tabs[1]:
    upload_section()


with tabs[2]:
    summary_section()


with tabs[3]:

    if st.session_state.papers:

        df = lda_topic_modeling(
            st.session_state.papers
        )

        st.dataframe(df)

        generate_wordcloud(
            st.session_state.papers
        )


with tabs[4]:
    insights_section()


with tabs[5]:
    from src.chat import chat_section

    chat_section()