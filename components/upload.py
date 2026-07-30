import streamlit as st

from src.metadata_extractor import extract_metadata
from src.pdf_parser import extract_text_from_pdf
from src.ai_functions import get_paper_analysis

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
            analysis = get_paper_analysis(
                meta["abstract"]
            )
            meta["summary"] = analysis["summary"]
            meta["limitations"] = analysis["limitations"]

            meta["research_gaps"] = analysis["research_gaps"]


            st.session_state.papers.append(meta)


        st.success(
            "Papers processed successfully!"
        )