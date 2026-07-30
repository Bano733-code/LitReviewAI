import streamlit as st

from components.upload import upload_section
from components.summaries import summary_section
from components.insights import insights_section

from src.chat import chat_section
from src.topic_modeling import lda_topic_modeling
from src.visualizations import generate_wordcloud


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="LitReviewAI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 LitReviewAI: AI-Powered Research Literature Assistant")


# =====================================================
# SESSION STATE
# =====================================================

if "papers" not in st.session_state:
    st.session_state.papers = []

if "collections" not in st.session_state:
    st.session_state.collections = {}


# =====================================================
# TABS
# =====================================================

tabs = st.tabs(
    [
        "ℹ️ About",
        "📤 Upload",
        "💬 Chat",
        "📑 Summaries",
        "📊 Topic Modeling",
        "⚡ Insights"
    ]
)


# =====================================================
# ABOUT
# =====================================================

with tabs[0]:

    st.header("About LitReviewAI")

    st.markdown(
        """
### LitReviewAI

LitReviewAI is an AI-powered research assistant designed to help researchers analyze, summarize, and explore scientific literature.

### Features

- 📄 Automatic PDF parsing
- 🧠 AI-powered paper analysis
- 🔍 Research gap identification
- ⚠️ Limitation extraction
- 🔑 Keyword extraction
- 📊 Topic modeling (LDA)
- 🌐 Co-author collaboration network
- 💬 Chat with uploaded research papers (RAG)
- 📚 BibTeX export
"""
    )


# =====================================================
# UPLOAD
# =====================================================

with tabs[1]:
    upload_section()


# =====================================================
# CHAT
# =====================================================

with tabs[2]:
    chat_section()


# =====================================================
# SUMMARIES
# =====================================================

with tabs[3]:
    summary_section()


# =====================================================
# TOPIC MODELING
# =====================================================

with tabs[4]:

    st.header("📊 Topic Modeling")

    if st.session_state.papers:

        topics = lda_topic_modeling(
            st.session_state.papers
        )

        if not topics.empty:
            st.dataframe(
                topics,
                use_container_width=True
            )

        st.subheader("☁️ Word Cloud")

        generate_wordcloud(
            st.session_state.papers
        )

    else:
        st.info("Upload papers first.")


# =====================================================
# INSIGHTS
# =====================================================

with tabs[5]:
    insights_section()