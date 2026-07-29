from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


# Load embedding model once
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# Initialize KeyBERT
kw_model = KeyBERT(
    model=embedding_model
)


def extract_keywords(text, top_n=5):
    """
    Extract important keywords from research abstract
    """

    if not text:
        return []

    try:
        keywords = kw_model.extract_keywords(
            text,
            top_n=top_n
        )

        return [
            keyword[0]
            for keyword in keywords
        ]

    except Exception:
        return []