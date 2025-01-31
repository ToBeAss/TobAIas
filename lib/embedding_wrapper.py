from langchain_ollama import OllamaEmbeddings

class Embedding_Wrapper:
    def __init__(self, model="ollama-nomic-embed-text"):
        self._embedding = self._init(model)
        self._prompt_cost = 0

    def _init(self, model):
        if model == "ollama-nomic-embed-text":
            return OllamaEmbeddings(
                model="nomic-embed-text",
            )