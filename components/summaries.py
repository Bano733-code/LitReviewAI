import streamlit as st
import pandas as pd
from src.bibtex import export_bibtex

def summary_section():

    st.header("📑 Paper Summaries")


    if not st.session_state.papers:

        st.info("Upload papers first.")
        return


    for i, paper in enumerate(
        st.session_state.papers,
        start=1
    ):

        with st.expander(
            f"{i}. {paper['title']}"
        ):

            st.write(
                "Authors:",
                ", ".join(paper["authors"])
            )

            st.write(
                "Summary:",
                paper["summary"]
            )

            st.write(
                "Limitations:",
                paper["limitations"]
            )

            st.write(
                "Research gaps:",
                paper["research_gaps"]
            )

            st.write(
                "Keywords:",
                paper["keywords"]
            )