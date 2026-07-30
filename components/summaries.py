import streamlit as st
import pandas as pd

from src.bibtex import export_bibtex


def display_section(title, content, icon):

    st.markdown(
        f"### {icon} {title}"
    )

    if content:
        st.write(content)
    else:
        st.info(
            f"{title} not generated yet."
        )


def summary_section():

    st.header("📑 Literature Analysis Dashboard")


    if not st.session_state.papers:

        st.info(
            "📂 Upload research papers first."
        )

        return



    total = len(
        st.session_state.papers
    )


    st.success(
        f"📚 Total Papers Analyzed: {total}"
    )



    # ===============================
    # PAPER CARDS
    # ===============================

    for index, paper in enumerate(
        st.session_state.papers,
        start=1
    ):


        title = paper.get(
            "title",
            "Untitled Paper"
        )


        with st.expander(
            f"📄 {index}. {title}",
            expanded=False
        ):


            # -------------------------
            # Metadata
            # -------------------------

            st.markdown(
                "## 📌 Paper Information"
            )


            authors = paper.get(
                "authors",
                ["Unknown"]
            )


            if isinstance(authors, list):

                authors_text = ", ".join(
                    authors
                )

                author_count = len(authors)

            else:

                authors_text = str(authors)
                author_count = 1



            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    "👤 **Authors**"
                )

                st.write(
                    authors_text
                )


            with col2:

                st.write(
                    "📄 **Title**"
                )

                st.write(
                    title
                )



            st.divider()



            # -------------------------
            # Statistics
            # -------------------------

            text = paper.get(
                "text",
                ""
            )


            words = len(
                text.split()
            )


            reading_time = max(
                1,
                words // 220
            )



            c1, c2, c3 = st.columns(3)


            c1.metric(
                "📝 Words",
                f"{words:,}"
            )


            c2.metric(
                "👥 Authors",
                author_count
            )


            c3.metric(
                "⏱ Reading Time",
                f"{reading_time} min"
            )



            st.divider()

            # -------------------------
            # AI Insights
            # -------------------------

            st.divider()

            display_section(
                "Abstract Summary",
                paper.get(
                    "abstract_summary",
                    ""
                ),
                "📖"
            )


            display_section(
                "AI Summary",
                paper.get(
                    "summary",
                    ""
                ),
                "🧠"
            )


            display_section(
                "Research Gaps",
                paper.get(
                    "research_gaps",
                    ""
                ),
                "🔬"
            )


            display_section(
                "Limitations",
                paper.get(
                    "limitations",
                    ""
                ),
                "⚠️"
            )



            # -------------------------
            # Keywords
            # -------------------------

            keywords = paper.get(
                "keywords",
                []
            )


            if keywords:

                st.divider()

                st.markdown(
                    "### 🏷 Keywords"
                )


                if isinstance(
                    keywords,
                    list
                ):

                    st.write(
                        " • ".join(keywords)
                    )

                else:

                    st.write(
                        keywords
                    )



    # ===============================
    # EXPORT
    # ===============================

    st.divider()


    st.subheader(
        "📥 Export Results"
    )


    df = pd.DataFrame(
        st.session_state.papers
    )


    col1, col2, col3 = st.columns(3)



    with col1:

        st.download_button(
            "⬇ CSV",
            df.to_csv(
                index=False
            ).encode("utf-8"),
            "litreviewai_results.csv",
            "text/csv"
        )



    with col2:

        st.download_button(
            "⬇ JSON",
            df.to_json(
                indent=2
            ).encode("utf-8"),
            "litreviewai_results.json",
            "application/json"
        )



    with col3:

        st.download_button(
            "⬇ BibTeX",
            export_bibtex(
                st.session_state.papers
            ),
            "papers.bib",
            "text/plain"
        )