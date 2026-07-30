import streamlit as st

from groq import Groq

from src.embeddings import (
    embedding_model
)


from src.vector_store import VectorStore



client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)



vector_db = VectorStore()



def chunk_text(
        text,
        chunk_size=1000
):

    chunks=[]


    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunks.append(
            text[i:i+chunk_size]
        )


    return chunks



def create_rag_database(
        papers
):


    documents=[]


    for paper in papers:

        chunks = chunk_text(
            paper["text"]
        )


        documents.extend(
            chunks
        )



    embeddings = embedding_model.encode(
        documents
    )


    vector_db.build(
        embeddings,
        documents
    )



def ask_rag(
        question
):


    query_embedding = (
        embedding_model.encode(
            question
        )
    )


    context = vector_db.search(
        query_embedding
    )



    context_text="\n\n".join(
        context
    )


    prompt=f"""

You are a scientific research assistant.

Answer using only the provided papers.

Context:

{context_text}


Question:

{question}

"""


    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
            "role":"user",
            "content":prompt
            }
        ],

        temperature=0.2
    )


    return response.choices[0].message.content