# app.py
import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF for PDF parsing
from gensim import corpora, models
from gensim.utils import simple_preprocess
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from keybert import KeyBERT
from deep_translator import GoogleTranslator
import networkx as nx
import plotly.graph_objects as go
import io
import bibtexparser
from nltk.corpus import stopwords
from groq import Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
import re
import nltk

# ================== CONFIG ==================
st.set_page_config(page_title="LitReviewAI", page_icon="📚", layout="wide")
st.title("📚 LitReviewAI: Automated Research Paper Reviewer")

# ================== GLOBALS ==================
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=embedding_model)

if "papers" not in st.session_state:
    st.session_state.papers = []  # store metadata & results
if "collections" not in st.session_state:
    st.session_state.collections = {}

# ensure nltk stopwords present
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# ================== HELPERS ==================
def extract_text_from_pdf(pdf_file):
    """Extract all text from PDF. Resets pointer outside if needed."""
    data = pdf_file.read()
    pdf_file.seek(0)
    with fitz.open(stream=data, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")
    return text


def extract_title(text):
    """Extracts paper title (before Abstract), avoiding citations and irrelevant headers."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Find index of "Abstract" (case-insensitive) or common translations
    abs_index = None
    for i, line in enumerate(lines):
        low = line.lower()
        if low.startswith("abstract") or any(k in low for k in ["résumé", "resumen", "summary", "overview", "abstract—"]):
            abs_index = i
            break

    # If Abstract found, check lines before abstract; else check top area
    if abs_index is not None:
        candidate_lines = lines[:abs_index]
    else:
        candidate_lines = lines[:100]

    skip_words = ["journal", "doi", "copyright", "arxiv", "volume",
                  "methods", "open access", "citation", "editor", "published"]

    candidates = []
    for line in candidate_lines:
        low = line.lower()
        # Skip unwanted lines
        if any(w in low for w in skip_words):
            continue
        # Skip author-like lists (many commas) or emails
        if "@" in line or re.search(r"\bjournal\b", low):
            continue
        # Skip short/very long lines
        if 5 <= len(line.split()) <= 30:
            # avoid lines that look like "Editor: Name" etc
            if re.match(r"^(editor|edited by|edited).+", low):
                continue
            candidates.append(line)

    # Prefer line with colon (typical title: subtitle)
    for line in candidates:
        if ":" in line:
            return line

    # fallback: longest candidate
    if candidates:
        return max(candidates, key=len)
    return "Untitled"


def extract_authors(text, max_scan_lines=80):
    """
    Flexible author extraction:
    - Finds a likely title, then scans the following lines looking for names.
    - Handles comma-separated authors on one line or names on separate lines.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    authors = []

    # 1) find a likely title index (first reasonably long line)
    title_idx = None
    for i, line in enumerate(lines[:40]):
        if len(line.split()) >= 4 and not any(k in line.lower() for k in ["journal", "doi", "copyright", "arxiv"]):
            title_idx = i
            break
    start = title_idx + 1 if title_idx is not None else 0

    # 2) scan lines after title until we hit abstract/keywords or a limit
    abstract_markers = ["abstract", "résumé", "resumen", "summary", "overview", "introduction", "keywords"]
    for line in lines[start:start + max_scan_lines]:
        low = line.lower()
        if any(m in low for m in abstract_markers):
            break

        # If line contains commas or " and ", likely multiple authors
        if "," in line or " and " in low or ";" in line:
            parts = re.split(r",|;|\sand\s", line)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # match typical name patterns: "First Last", "First M. Last", "Last, First"
                # handle "Last, First" by swapping
                if re.match(r"^[A-Z][a-z]+,\s*[A-Z][a-z]+", part):
                    # swap "Last, First" -> "First Last"
                    t = re.split(r",\s*", part)
                    name = t[1] + " " + t[0]
                else:
                    name = part

                # require at least two capitalized tokens to be considered a name
                cap_tokens = [t for t in name.split() if re.match(r"^[A-Z][a-z]+\.?$", t) or re.match(r"^[A-Z]\.$", t)]
                if len(cap_tokens) >= 2 and not looks_like_affiliation(name):
                    authors.append(name)
            if authors:
                continue
        else:
            # attempt single-line name extraction
            matches = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z]\.?\s?[A-Z][a-z]+){0,2}\b', line)
            for m in matches:
                if len(m.split()) >= 2 and not looks_like_affiliation(m):
                    authors.append(m)

    # 3) fallback: check metadata-like "By X Y" lines
    if not authors:
        for line in lines[:40]:
            m = re.match(r"^(By|BY|by)\s+(.+)$", line)
            if m:
                parts = re.split(r",|;|\sand\s", m.group(2))
                for p in parts:
                    p = p.strip()
                    if p and not looks_like_affiliation(p):
                        authors.append(p)
                if authors:
                    break

    return authors if authors else ["Unknown"]


AFFIL_KEYWORDS = [
    "university", "institute", "department", "school", "hospital",
    "clinic", "center", "centre", "laboratory", "lab", "college",
    "medicine", "research", "faculty", "division", "program", "department of"
]

def looks_like_affiliation(line):
    low = line.lower()
    if any(k in low for k in AFFIL_KEYWORDS):
        return True
    if "@" in line or "http" in low or "www." in low:
        return True
    # if line has few capitalized tokens relative to total tokens, it's likely not a name
    toks = [t for t in line.split() if t.strip()]
    if len(toks) == 0:
        return True
    cap = sum(1 for t in toks if re.match(r"^[A-Z][a-z]+$", t))
    if cap / len(toks) < 0.4:
        return True
    return False


def extract_abstract(text):
    """Extract abstract block (multiple formats supported). Gathers until next main heading."""
    lines = text.split("\n")
    abstract_lines = []
    capture = False
    abstract_markers = ["abstract", "résumé", "resumen", "summary", "overview"]
    stop_markers = ["introduction", "keywords", "1.", "methods", "materials", "results", "conclusion", "references"]

    for line in lines:
        l = line.lower().strip()
        if any(m in l for m in abstract_markers):
            capture = True
            continue
        if capture:
            # stop when a common section heading appears
            if any(re.match(rf"^{sm}\b", l) for sm in stop_markers):
                break
            abstract_lines.append(line.strip())
    abstract = " ".join(abstract_lines).strip()
    return abstract


def extract_metadata(pdf_file):
    """Extract title, authors, abstract from PDF file-like object."""
    text = extract_text_from_pdf(pdf_file)
    title = extract_title(text)
    authors = extract_authors(text)
    abstract = extract_abstract(text)
    return {"title": title or "Untitled", "authors": authors, "abstract": abstract or ""}


# ---- LLM helpers: summarization, sections, gaps, limitations ----
def get_summary(text):
    """Simple summary of abstract (concise)."""
    if not text:
        return "No abstract available."
    prompt = f"""
You are an academic assistant. Summarize the abstract concisely in 3-5 sentences, focusing on contributions, methods and main findings. Use simple language suitable for a researcher.
Abstract:
{text}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def chunk_text(text, max_chars=4000):
    """Split text into smaller parts safely for Groq API."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def get_section_summaries(text):
    summaries = []
    chunks = chunk_text(text)
    for chunk in chunks:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Summarize this section:\n{chunk}"}],
            temperature=0.3,
        )
        summaries.append(response.choices[0].message.content)
    return "\n\n".join(summaries)

def get_limitations(text):
    if not text:
        return "No abstract available."
    prompt = f"Extract only the limitations or challenges discussed in this abstract (if any). If none, write 'No explicit limitations mentioned'.\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_research_gaps(text):
    """Return 2-4 potential research gaps based on abstract."""
    if not text:
        return "No abstract available."
    prompt = f"""
You are a research assistant. Based on this abstract, list 2–4 realistic research gaps or unexplored questions that follow logically from the study. Provide concise bullet points.
Abstract:
{text}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ========== LDA TOPIC MODELING ==========
def lda_topic_modeling(papers, num_topics=3):
    # Preprocess abstracts with stopword removal
    texts = [
        [word for word in simple_preprocess(p["abstract"]) if word not in stop_words]
        for p in papers if p.get("abstract")
    ]
    # Guard: if no texts or empty docs
    if not texts or all(len(t) == 0 for t in texts):
        return pd.DataFrame([])

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    topics = lda_model.print_topics(num_words=6)
    topic_data = []
    for topic_id, words in topics:
        topic_data.append({"Topic ID": topic_id, "Keywords": words})
    df = pd.DataFrame(topic_data)
    return df


# ========== WORDCLOUD ==========
def generate_wordcloud(papers):
    text = " ".join([p.get("abstract", "") for p in papers if p.get("abstract")])
    if not text:
        st.info("No abstracts to build a word cloud.")
        return

    custom_stopwords = set(STOPWORDS).union(stop_words)
    wc = WordCloud(
        width=800,
        height=400,
        stopwords=custom_stopwords,
        background_color="white",
        colormap="viridis"
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# ========== CO-AUTHOR GRAPH ==========
def build_coauthor_graph(papers):
    G = nx.Graph()

    for p in papers:
        authors = p.get("authors", ["Unknown"])
        # normalize
        authors = [a.strip() for a in authors if a and isinstance(a, str)]
        # filter obvious affiliation tokens
        authors = [a for a in authors if not looks_like_affiliation(a) and a.lower() != "unknown"]

        # add nodes
        for a in authors:
            G.add_node(a)

        # add edges (with weight)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                u, v = authors[i], authors[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    if G.number_of_nodes() == 0:
        st.info("No author data available to build co-author graph.")
        return

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_text = [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y); node_text.append(n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="top center",
        marker=dict(size=12, color='skyblue')
    ))
    st.plotly_chart(fig, use_container_width=True)


# ========== BIBTEX EXPORT ==========
def export_bibtex(papers):
    db = {"entries": []}
    for i, p in enumerate(papers):
        title = p.get("title") or f"paper_{i}"
        authors = " and ".join(p.get("authors", ["Unknown"]))
        entry = {
            "ENTRYTYPE": "article",
            "ID": re.sub(r"\s+", "_", title)[:80],
            "title": title,
            "author": authors,
            "year": p.get("year", "2025"),
            "abstract": p.get("abstract", "")
        }
        db["entries"].append(entry)
    return bibtexparser.dumps(db)


# ================== UI / TABS ==================
tabs = st.tabs(["ℹ️ About", "📤 Upload Papers", "📑 Paper Summaries",
                "📊 Topic Modeling", "📂 Collections", "⚡ Trends & Insights", "💬 Chat with Papers"])

# --- About ---
with tabs[0]:
    st.header("About LitReviewAI")
    st.write("""
**LitReviewAI** helps researchers save time by:
- Uploading and analyzing research papers
- Extracting abstracts, keywords, and authors
- Generating AI-powered summaries, limitations, and research gaps
- Providing section-wise highlights (Intro / Methods / Results / Conclusions)
- Chatting interactively with your papers
- Building collaboration networks and topic clusters
- Exporting citations in BibTeX format
""")

# --- Upload ---
with tabs[1]:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Upload research papers (PDF)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file.seek(0)
            meta = extract_metadata(file)
            # section-wise summary (model may be slower; optional)
            text_for_sections = extract_text_from_pdf(file)
            meta["section_summary"] = get_section_summaries(text_for_sections)
            meta["summary"] = get_summary(meta.get("abstract", ""))
            meta["limitations"] = get_limitations(meta.get("abstract", ""))
            meta["research_gaps"] = get_research_gaps(meta.get("abstract", ""))
            # Extract keywords
            try:
                keywords = kw_model.extract_keywords(meta.get("abstract", ""), top_n=5)
                meta["keywords"] = ", ".join([kw[0] for kw in keywords]) if keywords else ""
            except Exception:
                meta["keywords"] = ""
            st.session_state.papers.append(meta)
        st.success("Papers processed and added!")

# --- Summaries ---
with tabs[2]:
    st.header("Paper Summaries")
    if st.session_state.papers:
        for i, paper in enumerate(st.session_state.papers, start=1):
            with st.expander(f"📄 {i}. {paper.get('title', 'Untitled')}"):
                st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}")
                st.markdown(f"**Summary:** {paper.get('summary', 'N/A')}")
                st.markdown(f"**Limitations:** {paper.get('limitations', 'N/A')}")
                st.markdown(f"**Research Gaps:** {paper.get('research_gaps', 'N/A')}")
                st.markdown("**Section-wise highlights (raw):**")
                st.code(paper.get('section_summary', 'N/A'))
                st.markdown(f"**Keywords:** {paper.get('keywords', 'N/A')}")
    else:
        st.info("Upload papers first.")

    if st.session_state.papers:
        df = pd.DataFrame(st.session_state.papers).astype(str)
        st.dataframe(df[["title", "summary", "research_gaps", "keywords"]])
        st.download_button("Download CSV", df.to_csv().encode("utf-8"), "summaries.csv")
        st.download_button("Download JSON", df.to_json(indent=2).encode("utf-8"), "summaries.json")
        st.download_button("Export BibTeX", export_bibtex(st.session_state.papers), "papers.bib")

# --- Topic Modeling ---
with tabs[3]:
    st.header("Topic Modeling & Word Cloud")
    if st.session_state.papers:
        topics_df = lda_topic_modeling(st.session_state.papers)
        if not topics_df.empty:
            st.dataframe(topics_df)
        else:
            st.info("Not enough text to build topics.")
        generate_wordcloud(st.session_state.papers)
    else:
        st.info("Upload papers first.")

# --- Collections ---
with tabs[4]:
    st.header("Collections")
    if st.session_state.papers:
        collection_name = st.text_input("Collection name")
        if st.button("Save to Collection") and collection_name:
            st.session_state.collections[collection_name] = st.session_state.papers.copy()
            st.success(f"Saved {len(st.session_state.papers)} papers to {collection_name}")
        if st.session_state.collections:
            st.write(list(st.session_state.collections.keys()))
    else:
        st.info("Upload papers first.")

# --- Trends & Insights ---
with tabs[5]:
    st.header("Trends & Insights")
    if st.session_state.papers:
        st.markdown("### Co-author Collaboration Network")
        build_coauthor_graph(st.session_state.papers)

        st.markdown("### Quick stats")
        # simple stats: number of papers, unique authors
        num_papers = len(st.session_state.papers)
        all_authors = []
        for p in st.session_state.papers:
            all_authors.extend([a for a in p.get("authors", []) if a and a.lower() != "unknown"])
        unique_authors = sorted(set(all_authors))
        st.write(f"Papers: **{num_papers}** — Unique authors: **{len(unique_authors)}**")
        if unique_authors:
            st.write(", ".join(unique_authors[:20]) + ("" if len(unique_authors) <= 20 else ", ..."))
    else:
        st.info("Upload papers first.")

# --- Chat with Papers ---
with tabs[6]:
    st.header("💬 Chat with Your Papers")
    if st.session_state.papers:
        combined_abstracts = "\n\n".join([p.get("abstract", "") for p in st.session_state.papers if p.get("abstract")])
        st.write("You can ask questions about the collection of uploaded abstracts.")
        user_q = st.text_area("Ask a question", height=120)
        if st.button("Ask"):
            if not user_q.strip():
                st.warning("Please type a question.")
            else:
                prompt = f"""You are an academic assistant. Use the following abstracts as context and answer the question concisely and precisely. If the required info is not available in the abstracts, say you don't have enough info.
Context (abstracts):
{combined_abstracts}

Question:
{user_q}
"""
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.markdown("### 🧠 Answer:")
                st.write(response.choices[0].message.content.strip())
    else:
        st.info("Upload papers first.")
