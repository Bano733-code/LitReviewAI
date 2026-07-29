import streamlit as st

from src.metadata_extractor import extract_metadata
from src.pdf_parser import extract_text_from_pdf
from src.ai_functions import (
    get_summary,
    get_limitations,
    get_research_gaps,
    get_section_summaries
)

def upload_section():

    st.header("📤 Upload Research Papers")


    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type="pdf",
        accept_multiple_files=True
    )


    if uploaded_files:

        for file in uploaded_files:

            meta = extract_metadata(file)


            text = extract_text_from_pdf(file)


            meta["section_summary"] = get_section_summaries(text)

            meta["summary"] = get_summary(
                meta["abstract"]
            )

            meta["limitations"] = get_limitations(
                meta["abstract"]
            )

            meta["research_gaps"] = get_research_gaps(
                meta["abstract"]
            )


            meta["keywords"] = ", ".join(
                extract_keywords(
                    meta["abstract"]
                )
            )


            st.session_state.papers.append(meta)


        st.success(
            "Papers processed successfully!"
        )