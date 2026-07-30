from sentence_transformers import SentenceTransformer


# Load once
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_embeddings(texts):

    """
    Convert text chunks into vectors
    """

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings