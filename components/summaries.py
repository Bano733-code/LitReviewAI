import streamlit as st
import pandas as pd
from src.bibtex import export_bibtex


def summary_section():

    st.header("📑 Paper Summaries")

    if not st.session_state.papers:
        st.info("📂 Upload research papers first.")
        return

    st.success(f"Total Papers: {len(st.session_state.papers)}")

    for i, paper in enumerate(st.session_state.papers, start=1):

        with st.expander(f"📄 {i}. {paper.get('title', 'Untitled')}"):

            # -------------------------
            # Basic Information
            # -------------------------

            st.subheader("📌 Paper Information")

            authors = paper.get("authors", ["Unknown"])

            if isinstance(authors, list):
                authors_text = ", ".join(authors)
            else:
                authors_text = str(authors)

            st.write("**👤 Authors:**", authors_text)

            st.write(
                "**📄 Title:**",
                paper.get("title", "Unknown")
            )

            # -------------------------
            # Paper Statistics
            # -------------------------

            text = paper.get("text", "")

            word_count = len(text.split())

            reading_time = max(1, word_count // 220)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Words",
                f"{word_count:,}"
            )

            col2.metric(
                "Authors",
                len(authors) if isinstance(authors, list) else 1
            )

            col3.metric(
                "Reading Time",
                f"{reading_time} min"
            )

            st.divider()

            # -------------------------
            # Abstract
            # -------------------------

            with st.expander("📖 Abstract"):

                abstract = paper.get("abstract", "")

                if abstract:
                    st.write(abstract)
                else:
                    st.info("Abstract not available.")

            # -------------------------
            # AI Summary
            # -------------------------

            st.subheader("🧠 AI Summary")

            summary = paper.get("summary", "")

            if summary:
                st.write(summary)
            else:
                st.info("Summary has not been generated yet.")

            # -------------------------
            # Research Gaps
            # -------------------------

            st.subheader("🔬 Research Gaps")

            gaps = paper.get("research_gaps", "")

            if gaps:
                st.write(gaps)
            else:
                st.info("Research gaps not available.")

            # -------------------------
            # Limitations
            # -------------------------

            st.subheader("⚠️ Limitations")

            limitations = paper.get("limitations", "")

            if limitations:
                st.write(limitations)
            else:
                st.info("Limitations not available.")

            # -------------------------
            # Keywords
            # -------------------------

            if "keywords" in paper and paper["keywords"]:

                st.subheader("🏷 Keywords")

                if isinstance(paper["keywords"], list):
                    st.write(", ".join(paper["keywords"]))
                else:
                    st.write(paper["keywords"])

            st.divider()

    # ---------------------------------------
    # Export Section
    # ---------------------------------------

    st.subheader("📥 Export")

    df = pd.DataFrame(st.session_state.papers)

    st.download_button(
        "⬇ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="litreviewai_results.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇ Download JSON",
        df.to_json(indent=2).encode("utf-8"),
        file_name="litreviewai_results.json",
        mime="application/json",
    )

    st.download_button(
        "⬇ Export BibTeX",
        export_bibtex(st.session_state.papers),
        file_name="papers.bib",
        mime="text/plain",
    )