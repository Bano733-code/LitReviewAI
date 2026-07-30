import faiss
import numpy as np


class VectorStore:


    def __init__(self):

        self.index = None
        self.documents = []



    def build(self, embeddings, documents):

        embeddings = np.array(
            embeddings
        ).astype("float32")


        dimension = embeddings.shape[1]


        self.index = faiss.IndexFlatL2(
            dimension
        )


        self.index.add(
            embeddings
        )


        self.documents = documents



    def search(self, query_embedding, k=5):

        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")


        distances, indices = self.index.search(
            query_embedding,
            k
        )


        results=[]


        for idx in indices[0]:

            if idx < len(self.documents):

                results.append(
                    self.documents[idx]
                )


        return results