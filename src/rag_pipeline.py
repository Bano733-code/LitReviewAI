import streamlit as st
from groq import Groq

from src.embeddings import embedding_model
from src.vector_store import VectorStore


# =====================================================
# GROQ CLIENT
# =====================================================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

vector_db = VectorStore()


# =====================================================
# CHUNKING
# =====================================================

def chunk_text(text, chunk_size=1000):

    if not text:
        return []

    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


# =====================================================
# VECTOR DATABASE
# =====================================================

def create_rag_database(papers):

    documents = []

    for paper in papers:

        chunks = chunk_text(
            paper.get("text", "")
        )

        documents.extend(chunks)

    if not documents:
        return

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=False
    )

    vector_db.build(
        embeddings,
        documents
    )


# =====================================================
# CHAT WITH PAPERS
# =====================================================

def ask_rag(question):

    query_embedding = embedding_model.encode(question)

    context = vector_db.search(query_embedding)

    context_text = "\n\n".join(context)

    prompt = f"""
You are an expert scientific research assistant.

Answer ONLY from the provided research paper.

If the answer is not available,
reply:

"I could not find this information in the uploaded paper."

Context:

{context_text}

Question:

{question}
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# =====================================================
# AI SUMMARY
# =====================================================
def summarize_abstract(abstract):

    prompt = f"""
Summarize this abstract in 5-6 sentences.

Include:
- Research problem
- Methodology
- Main findings
- Conclusion

Abstract:

{abstract[:3000]}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def generate_summary(text):

    if not text:
        return "No text available."

    prompt = f"""
Summarize the following scientific paper.

Include:

• Research objective

• Methodology

• Key findings

• Conclusion

Paper:

{text[:5000]}
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )

    return response.choices[0].message.content


# =====================================================
# RESEARCH GAPS
# =====================================================

def generate_research_gaps(text):

    if not text:
        return "No text available."

    prompt = f"""
Read this scientific paper.

Identify 3 potential future research directions
or research gaps.

Paper:

{text[:5000]}
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )

    return response.choices[0].message.content


# =====================================================
# LIMITATIONS
# =====================================================

def generate_limitations(text):

    if not text:
        return "No text available."

    prompt = f"""
Identify the limitations of this scientific study.

If limitations are not explicitly written,
infer reasonable limitations from the study.

Paper:

{text[:5000]}
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )

    return response.choices[0].message.content


# =====================================================
# COMPLETE PAPER ANALYSIS
# =====================================================

def analyze_paper(text):

    return {

        "summary":
            generate_summary(text),
 
        "research_gaps":
            generate_research_gaps(text),

        "limitations":
            generate_limitations(text)

    }