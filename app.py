import streamlit as st

from components.upload import upload_section
from components.summaries import summary_section
from components.insights import insights_section

from src.topic_modeling import lda_topic_modeling
from src.visualizations import generate_wordcloud
from src.chat import chat_section


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="LitReviewAI",
    page_icon="📚",
    layout="wide"
)


st.title("📚 LitReviewAI: Automated Research Paper Reviewer")


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
        "📤 Upload Papers",
        "📑 Summaries",
        "📊 Topic Modeling",
        "⚡ Insights",
        "💬 Chat"
    ]
)


# =====================================================
# ABOUT
# =====================================================

with tabs[0]:

    st.header("About LitReviewAI")

    st.write(
        """
        **LitReviewAI** is an AI-powered research assistant that helps
        researchers analyze scientific literature.

        Features:

        - 📄 Automatic PDF analysis
        - 🧠 AI-powered paper summarization
        - 🔍 Research gap detection
        - ⚠️ Limitation extraction
        - 🔑 Keyword extraction using KeyBERT
        - 📊 Topic modeling using LDA
        - 🌐 Collaboration network analysis
        - 💬 Chat with research papers
        - 📚 BibTeX citation export
        """
    )


# =====================================================
# UPLOAD
# =====================================================

with tabs[1]:

    upload_section()



# =====================================================
# SUMMARIES
# =====================================================

with tabs[2]:

    summary_section()



# =====================================================
# TOPIC MODELING
# =====================================================

with tabs[3]:

    st.header("📊 Topic Modeling")

    if st.session_state.papers:

        topics_df = lda_topic_modeling(
            st.session_state.papers
        )

        if not topics_df.empty:
            st.dataframe(
                topics_df,
                use_container_width=True
            )

        else:
            st.info(
                "Not enough text for topic modeling."
            )


        st.subheader("Word Cloud")

        generate_wordcloud(
            st.session_state.papers
        )


    else:

        st.info(
            "Upload papers first."
        )



# =====================================================
# INSIGHTS
# =====================================================

with tabs[4]:

    insights_section()



# =====================================================
# CHAT
# =====================================================

with tabs[5]:

    chat_section()