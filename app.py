# app.py
import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF
from gensim import corpora, models
from gensim.utils import simple_preprocess
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from keybert import KeyBERT
import networkx as nx
import plotly.graph_objects as go
import re
import nltk
import time
from nltk.corpus import stopwords
from groq import Groq

# ================== CONFIG ==================
st.set_page_config(page_title="LitReviewAI", page_icon="📚", layout="wide")
st.title("📚 LitReviewAI: Automated Research Paper Reviewer")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ================== MODELS ==================
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=embedding_model)

# ================== STATE ==================
if "papers" not in st.session_state:
    st.session_state.papers = []

if "collections" not in st.session_state:
    st.session_state.collections = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================== NLTK ==================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ================== RATE LIMIT PROTECTION ==================
_last_call = 0
def throttle():
    global _last_call
    now = time.time()
    if now - _last_call < 1.2:
        time.sleep(1.2)
    _last_call = time.time()

# ================== PDF EXTRACTION ==================
def extract_text_from_pdf(pdf_file):
    data = pdf_file.read()
    pdf_file.seek(0)
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_title(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else "Untitled"

def extract_authors(text):
    return ["Unknown"]

def extract_abstract(text):
    return text[:1500]

def extract_metadata(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    return {
        "title": extract_title(text),
        "authors": extract_authors(text),
        "abstract": extract_abstract(text),
    }

# ================== LLM HELPERS ==================
def get_summary(text):
    if not text:
        return "No abstract available."

    throttle()

    prompt = f"Summarize this abstract in 3-5 sentences:\n\n{text}"

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return res.choices[0].message.content.strip()


def get_limitations(text):
    if not text:
        return "No abstract available."

    throttle()

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"Extract limitations:\n{text}"}]
    )

    return res.choices[0].message.content.strip()


def get_research_gaps(text):
    if not text:
        return "No abstract available."

    throttle()

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"List research gaps:\n{text}"}]
    )

    return res.choices[0].message.content.strip()


def get_section_summaries(text):
    throttle()

    chunks = text[:6000]

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Summarize sections:\n{chunks}"
        }],
        temperature=0.3,
    )

    return res.choices[0].message.content.strip()

# ================== TOPIC MODEL ==================
def lda_topic_modeling(papers, num_topics=3):
    texts = [
        [w for w in simple_preprocess(p["abstract"]) if w not in stop_words]
        for p in papers if p.get("abstract")
    ]

    if not texts:
        return pd.DataFrame()

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]

    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    return pd.DataFrame([
        {"Topic": i, "Keywords": words}
        for i, words in lda.print_topics()
    ])

# ================== WORDCLOUD ==================
def generate_wordcloud(papers):
    text = " ".join([p.get("abstract", "") for p in papers])

    if not text.strip():
        st.info("No text for wordcloud")
        return

    wc = WordCloud(
        width=800,
        height=400,
        stopwords=STOPWORDS.union(stop_words),
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

# ================== GRAPH ==================
def build_coauthor_graph(papers):
    G = nx.Graph()

    for p in papers:
        for a in p.get("authors", ["Unknown"]):
            G.add_node(a)

    pos = nx.spring_layout(G)

    fig = go.Figure()
    st.plotly_chart(fig)

# ================== UI ==================
tabs = st.tabs([
    "Upload",
    "Summaries",
    "Topics",
    "Collections",
    "Trends",
    "Chat AI"
])

# ================== UPLOAD ==================
with tabs[0]:
    st.header("Upload Papers")

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if files:
        for f in files:
            meta = extract_metadata(f)
            text = extract_text_from_pdf(f)

            meta["section_summary"] = "Click to generate"
            meta["summary"] = "Click to generate"
            meta["limitations"] = "Click to generate"
            meta["research_gaps"] = "Click to generate"

            st.session_state.papers.append(meta)

        st.success("Uploaded successfully")

# ================== SUMMARIES ==================
with tabs[1]:
    st.header("Paper Insights")

    for i, p in enumerate(st.session_state.papers):

        with st.expander(p["title"]):

            if st.button(f"Generate Summary {i}"):
                p["summary"] = get_summary(p["abstract"])
                st.rerun()

            if st.button(f"Generate Section Summary {i}"):
                p["section_summary"] = get_section_summaries(p["abstract"])
                st.rerun()

            if st.button(f"Generate Limitations {i}"):
                p["limitations"] = get_limitations(p["abstract"])
                st.rerun()

            if st.button(f"Generate Gaps {i}"):
                p["research_gaps"] = get_research_gaps(p["abstract"])
                st.rerun()

            st.write("Summary:", p.get("summary"))
            st.write("Limitations:", p.get("limitations"))
            st.write("Gaps:", p.get("research_gaps"))
            st.write("Section:", p.get("section_summary"))

# ================== TOPICS ==================
with tabs[2]:
    st.header("Topics")

    df = lda_topic_modeling(st.session_state.papers)
    st.dataframe(df)

    generate_wordcloud(st.session_state.papers)

# ================== COLLECTIONS ==================
with tabs[3]:
    st.header("Collections")

    name = st.text_input("Collection name")

    if st.button("Save"):
        st.session_state.collections[name] = st.session_state.papers.copy()
        st.success("Saved")

    st.write(st.session_state.collections.keys())

# ================== TRENDS ==================
with tabs[4]:
    st.header("Trends")

    build_coauthor_graph(st.session_state.papers)

# ================== CHAT AI (NEW ADDED SECTION) ==================
with tabs[5]:
    st.header("💬 Chat with AI about your Papers")

    if st.session_state.papers:

        context = "\n\n".join([p.get("abstract", "") for p in st.session_state.papers])

        user_q = st.text_input("Ask a question")

        if st.button("Ask AI") and user_q.strip():

            throttle()

            prompt = f"""
You are an academic assistant.

Use ONLY the context below:
If answer not found say "Not enough information".

Context:
{context}

Question:
{user_q}

Answer:
"""

            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            answer = res.choices[0].message.content.strip()

            st.session_state.chat_history.append(("You", user_q))
            st.session_state.chat_history.append(("AI", answer))

        st.markdown("### Conversation")

        for role, msg in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"**🧑 You:** {msg}")
            else:
                st.markdown(f"**🤖 AI:** {msg}")

    else:
        st.info("Upload papers first")