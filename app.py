# app.py
import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF for PDF parsing
from gensim import corpora, models
from gensim.utils import simple_preprocess
from wordcloud import WordCloud,STOPWORDS
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


# ================== CONFIG ==================
st.set_page_config(page_title="LitReviewAI", page_icon="📚", layout="wide")
st.title("📚 LitReviewAI: Automated Research Paper Reviewer")

# ================== GLOBALS ==================
kw_model = KeyBERT()

if "papers" not in st.session_state:
    st.session_state.papers = []  # store metadata & results
if "collections" not in st.session_state:
    st.session_state.collections = {}

# ================== HELPERS ==================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
def extract_authors_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    first_page_text = doc[0].get_text("text")

    # simple heuristic: authors are usually after title but before abstract
    lines = first_page_text.split("\n")
    authors = []

    for line in lines[1:15]:  # scan first 15 lines after title
        if "abstract" in line.lower():
            break
        # avoid very long lines (likely affiliations)
        if 2 <= len(line.split()) <= 8:
            authors.append(line.strip())

    # fallback if nothing found
    if not authors:
        authors = ["Unknown"]

    return authors

def extract_metadata(text):
    # Very naive metadata extractor (replace with actual if needed)
    lines = text.split("\n")
    title = lines[0] if lines else "Untitled"
    abstract = ""
    for i, l in enumerate(lines):
        if "abstract" in l.lower():
            abstract = " ".join(lines[i+1:i+10])
            break
    return {"title": title, "abstract": abstract}

# ---- SUMMARIZATION ----
def get_summary(text):
    prompt = f"Summarize the key contributions and findings of this abstract in 2–3 sentences:\n\n{text}"
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ---- LIMITATIONS ----
def get_limitations(text):
    prompt = f"Extract only the limitations or challenges discussed in this abstract (if any). If none, write 'No explicit limitations mentioned'.\n\n{text}"
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# Download NLTK stopwords if not already
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# ========== LDA TOPIC MODELING ==========
def lda_topic_modeling(papers):
    # Preprocess abstracts with stopword removal
    texts = [
        [word for word in simple_preprocess(p["abstract"]) if word not in stop_words]
        for p in papers if p["abstract"]
    ]
    
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

    topics = lda_model.print_topics(num_words=6)

    # Convert topics into a DataFrame
    topic_data = []
    for topic_id, words in topics:
        topic_data.append({"Topic ID": topic_id, "Keywords": words})
    
    df = pd.DataFrame(topic_data)
    return df

# ========== WORDCLOUD ==========
def generate_wordcloud(papers):
    text = " ".join([p["abstract"] for p in papers if p["abstract"]])
    
    # Merge NLTK + WordCloud stopwords
    stop_words = set(stopwords.words("english"))
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

def build_coauthor_graph(papers):
    G = nx.Graph()
    for p in papers:
        authors = p.get("authors", ["Unknown"])
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                G.add_edge(authors[i], authors[j])
    pos = nx.spring_layout(G)
    edge_x, edge_y, node_x, node_y, text = [], [], [], [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        text.append(node)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5,color='#888'),
                             hoverinfo='none', mode='lines'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                             text=text, textposition="top center",
                             marker=dict(size=10, color='skyblue')))
    st.plotly_chart(fig)

def export_bibtex(papers):
    db = {"entries": []}
    for p in papers:
        db["entries"].append({
            "ENTRYTYPE": "article",
            "ID": p["title"].replace(" ", "_"),
            "title": p["title"],
            "author": " and ".join(p.get("authors", [])),
            "year": "2025",
            "abstract": p["abstract"]
        })
    return bibtexparser.dumps(db)

# ================== TABS ==================
tabs = st.tabs(["ℹ️ About","📤 Upload Papers", "📑 Paper Summaries", "📊 Topic Modeling",
                "📂 Collections", "⚡ Trends & Insights"])

# --- About ---
with tabs[0]:
    st.header("About LitReviewAI")
    st.write("""
    **LitReviewAI** helps researchers save time by:
    - Uploading papers
    - Extracting abstracts and keywords
    - Generating AI-powered summaries, limitations, and future directions
    - Clustering papers into research themes
    - Building collections for projects
    - Exploring collaboration networks
    """)


# --- Upload ---
with tabs[1]:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Upload research papers (PDF)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            meta = extract_metadata(text)
            file.seek(0)  # reset pointer because we already read it
            meta["authors"] = extract_authors_from_pdf(file)
            meta["summary"] = get_summary(meta["abstract"])
            meta["limitations"] = get_limitations(meta["abstract"])
            # Extract only keywords, not scores, and join into string
            keywords = kw_model.extract_keywords(meta["abstract"], top_n=5)
            meta["keywords"] = ", ".join([kw[0] for kw in keywords]) if keywords else ""
            st.session_state.papers.append(meta)
        st.success("Papers processed and added!")

# --- Summaries ---
with tabs[2]:
    st.header("Paper Summaries")
    if st.session_state.papers:
        for i, paper in enumerate(st.session_state.papers, start=1):
            with st.expander(f"📄 {i}. {paper.get('title', 'Untitled')}"):
                st.markdown(f"**Summary:** {paper.get('summary', 'N/A')}")
                st.markdown(f"**Limitations:** {paper.get('limitations', 'N/A')}")
                st.markdown(f"**Keywords:** {', '.join(paper.get('keywords', [])) if isinstance(paper.get('keywords'), list) else paper.get('keywords', 'N/A')}")
    
    if st.session_state.papers:
        df = pd.DataFrame(st.session_state.papers).astype(str)
        st.dataframe(df[["title", "summary", "limitations", "keywords"]])
        st.download_button("Download CSV", df.to_csv().encode("utf-8"), "summaries.csv")
        st.download_button("Download JSON", df.to_json(indent=2).encode("utf-8"), "summaries.json")
        st.download_button("Export BibTeX", export_bibtex(st.session_state.papers), "papers.bib")
    else:
        st.info("Upload papers first.")

# --- Topic Modeling ---
with tabs[3]:
    st.header("Topic Modeling & Word Cloud")
    if st.session_state.papers:
        topics_df = lda_topic_modeling(st.session_state.papers)
        st.dataframe(topics_df)   # Display as DataFrame instead of CSV
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
            st.write(st.session_state.collections.keys())
    else:
        st.info("Upload papers first.")

# --- Trends & Insights ---
with tabs[5]:
    st.header("Trends & Insights")
    if st.session_state.papers:
        build_coauthor_graph(st.session_state.papers)
    else:
        st.info("Upload papers first.")
