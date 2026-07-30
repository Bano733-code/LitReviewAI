import streamlit as st

from src.metadata_extractor import extract_metadata
from src.pdf_parser import extract_text_from_pdf

from src.rag_pipeline import (
    create_rag_database,
    generate_paper_summary,
    generate_limitations,
    generate_research_gaps
)


def upload_section():

    st.header("📤 Upload Research Papers")

    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        new_papers = []

        progress = st.progress(0)
        status = st.empty()

        total_files = len(uploaded_files)

        for idx, file in enumerate(uploaded_files):

            status.write(f"Processing **{file.name}**...")

            # ---------------------------------------
            # Extract metadata
            # ---------------------------------------
            meta = extract_metadata(file)

            # ---------------------------------------
            # Extract complete paper text
            # ---------------------------------------
            text = extract_text_from_pdf(file)

            meta["text"] = text

            # ---------------------------------------
            # Generate AI Summary
            # ---------------------------------------
            try:
                meta["summary"] = generate_paper_summary(text)
            except Exception:
                meta["summary"] = "Summary could not be generated."

            # ---------------------------------------
            # Generate Limitations
            # ---------------------------------------
            try:
                meta["limitations"] = generate_limitations(text)
            except Exception:
                meta["limitations"] = "Limitations could not be extracted."

            # ---------------------------------------
            # Generate Research Gaps
            # ---------------------------------------
            try:
                meta["research_gaps"] = generate_research_gaps(text)
            except Exception:
                meta["research_gaps"] = "Research gaps could not be generated."

            new_papers.append(meta)

            progress.progress((idx + 1) / total_files)

        # ---------------------------------------
        # Save papers
        # ---------------------------------------
        st.session_state.papers.extend(new_papers)

        # ---------------------------------------
        # Build Vector Database
        # ---------------------------------------
        create_rag_database(st.session_state.papers)

        status.empty()
        progress.empty()

        st.success(
            f"Successfully processed {len(new_papers)} paper(s)."
        )