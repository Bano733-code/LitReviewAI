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


        # Load existing database if available
        self.load()



    def build(
        self,
        embeddings,
        documents
    ):

        """
        Build FAISS vector database

        embeddings:
            SentenceTransformer embeddings

        documents:
            text chunks
        """


        embeddings = np.array(
            embeddings
        ).astype("float32")


        # Normalize for cosine similarity
        faiss.normalize_L2(
            embeddings
        )


        dimension = embeddings.shape[1]


        # Cosine similarity index
        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.index.add(
            embeddings
        )


        self.documents = documents


        # Save database
        self.save()



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


        # Normalize query vector
        faiss.normalize_L2(
            query_embedding
        )


        distances, indices = self.index.search(
            query_embedding,
            k
        )


        results=[]


        for score, idx in zip(
            distances[0],
            indices[0]
        ):


            if idx != -1:

                results.append(
                    {
                        "text": self.documents[idx],
                        "score": float(score)
                    }
                )


        return results



    def save(self):

        """
        Save FAISS index and documents
        """


        if self.index:

            faiss.write_index(
                self.index,
                self.index_path
            )


        with open(
            self.docs_path,
            "wb"
        ) as f:

            pickle.dump(
                self.documents,
                f
            )



    def load(self):

        """
        Load existing vector database
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