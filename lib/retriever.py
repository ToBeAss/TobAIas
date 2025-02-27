from lib.embedding_wrapper import Embedding_Wrapper
from langchain_chroma import Chroma

class Retriever:
    def __init__(self, embedding: Embedding_Wrapper):
        self._id = self._generate_id()
        self._embedding = embedding
        self._db = self._init_db()
        self._retriever = self._init_retriever()

    def _generate_id(self):
        id = f"vectordb_{Retriever.instances}"
        Retriever.instances += 1 # Increment on each instantiation
        return id

    def _init_db(self):
        return Chroma(
            persist_directory = self._id,
            embedding_function = self._embedding,
        )
    
    def retrieve_mmr(self, prompt, k=5, fetch_k=20):
        retriever = self._db.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                "k": k,
                "fetch_k": fetch_k,
            }
        )
        result = retriever.invoke(prompt)
        return result

    def retrieve_similarity_scores(self, prompt, k=5, score_threshold=0.4):
        result = self._db.similarity_search_with_relevance_scores(
            query = prompt,
            k = k,
            score_threshold = score_threshold,
        )
        return result