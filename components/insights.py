import streamlit as st

from src.visualizations import build_coauthor_graph


def insights_section():

    st.header("⚡ Research Insights")


    if not st.session_state.papers:

        st.info("Upload papers first.")
        return


    st.subheader(
        "Co-author Network"
    )

    build_coauthor_graph(
        st.session_state.papers
    )


    st.write(
        "Total Papers:",
        len(st.session_state.papers)
    )