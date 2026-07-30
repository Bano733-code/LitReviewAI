import streamlit as st

from src.metadata_extractor import extract_metadata
from src.pdf_parser import extract_text_from_pdf
from src.rag_pipeline import create_rag_database



def upload_section():

    st.header("📤 Upload Research Papers")


    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type="pdf",
        accept_multiple_files=True
    )


    if uploaded_files:


        new_papers = []


        for file in uploaded_files:


            # Metadata extraction
            meta = extract_metadata(file)


            # Extract complete paper text
            text = extract_text_from_pdf(file)


            # Store full text for RAG
            meta["text"] = text


            # Add empty placeholders
            meta["summary"] = (
                "Generated using RAG pipeline"
            )

            meta["limitations"] = (
                "Generated when queried"
            )

            meta["research_gaps"] = (
                "Generated when queried"
            )


            new_papers.append(meta)



        # Add papers to session
        st.session_state.papers.extend(
            new_papers
        )


        # Build FAISS vector database
        create_rag_database(
            st.session_state.papers
        )


        st.success(
            "Papers processed successfully!"
        )