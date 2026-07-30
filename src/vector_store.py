import faiss
import numpy as np
import pickle
import os


class VectorStore:

    def __init__(
        self,
        index_path="data/vector.index",
        docs_path="data/documents.pkl"
    ):

        self.index = None
        self.documents = []

        self.index_path = index_path
        self.docs_path = docs_path


        # Load existing database
        self.load()



    # =====================================================
    # BUILD VECTOR DATABASE
    # =====================================================

    def build(
        self,
        embeddings,
        documents
    ):

        """
        Create FAISS vector database.

        embeddings:
            SentenceTransformer embeddings

        documents:
            text chunks
        """


        if len(embeddings) == 0:
            return


        embeddings = np.array(
            embeddings
        ).astype("float32")


        # Normalize for cosine similarity
        faiss.normalize_L2(
            embeddings
        )


        dimension = embeddings.shape[1]


        # Cosine similarity search
        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.index.add(
            embeddings
        )


        self.documents = documents


        self.save()



    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query_embedding,
        k=5
    ):

        if self.index is None:
            return []


        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")


        faiss.normalize_L2(
            query_embedding
        )


        distances, indices = self.index.search(
            query_embedding,
            k
        )


        results = []


        for score, idx in zip(
            distances[0],
            indices[0]
        ):


            if idx != -1:

                results.append(
                    self.documents[idx]
                )


        return results



    # =====================================================
    # SAVE DATABASE
    # =====================================================

    def save(self):

        """
        Save FAISS index + documents.
        Creates folders automatically.
        """


        # Create directories if missing

        index_dir = os.path.dirname(
            self.index_path
        )

        docs_dir = os.path.dirname(
            self.docs_path
        )


        if index_dir:

            os.makedirs(
                index_dir,
                exist_ok=True
            )


        if docs_dir:

            os.makedirs(
                docs_dir,
                exist_ok=True
            )


        # Save FAISS index

        if self.index is not None:

            faiss.write_index(
                self.index,
                self.index_path
            )


        # Save documents

        with open(
            self.docs_path,
            "wb"
        ) as f:

            pickle.dump(
                self.documents,
                f
            )



    # =====================================================
    # LOAD DATABASE
    # =====================================================

    def load(self):

        """
        Load FAISS database if available.
        """


        if os.path.exists(
            self.index_path
        ):

            self.index = faiss.read_index(
                self.index_path
            )


        if os.path.exists(
            self.docs_path
        ):

            with open(
                self.docs_path,
                "rb"
            ) as f:

                self.documents = pickle.load(f)