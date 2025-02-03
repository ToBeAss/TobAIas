from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

class VectorDB_Wrapper:
    def __init__(self, model="chroma"):
        self._embedding = OllamaEmbeddings(model="nomic-embed-text")
        self._db = self._init(model)
        self._retriever = self._init_retriever()
        self._document_loader = PyPDFDirectoryLoader("")
        self._text_splitter = RecursiveCharacterTextSplitter()
        self._prompt_cost = 0

    def _init(self, model):
        if model == "chroma":
            return Chroma(
                persist_directory=model,
                embedding_function=self._embedding,
            )
        else:
            raise ValueError(f"Unsupported database: {model}")
        
    def _init_retriever(self):
        return self._db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
            },
        )
        
    def _load_documents(self, path="data"):
        self._document_loader.path = path
        return self._document_loader.load()
    
    def _split_documents(self, documents: list):
        self._text_splitter._chunk_size = 800
        self._text_splitter._chunk_overlap = 400
        self._text_splitter._length_function = len
        self._text_splitter._is_separator_regex = False
        return self._text_splitter.split_documents(documents)
    
    def _embed_chunks(self, chunks: list):
        print(f"👉 Adding new documents: {len(chunks)}")
        return self._db.add_documents(chunks)
    
    def embed_documents(self):
        documents = self._load_documents()
        chunks = self._split_documents(documents)
        vectors = self._embed_chunks(chunks)
        return vectors
    
    def invoke(self, prompt):
        for method in ["invoke", "retrieve", "call", "__call__", "fetch"]:
            if hasattr(self._retriever, method):
                results = getattr(self._retriever, method)(prompt)
                return results

        raise AttributeError(f"No valid invocation method found for {type(self._retriever).__name__}.")
    
    def get_prompt_cost(self):
        return self._prompt_cost