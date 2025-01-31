from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

class VectorDB_Wrapper:
    def __init__(self, model="chroma"):
        self._embedding = OllamaEmbeddings(model="nomic-embed-text")
        self._db = self._init(model)
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
        
    def _load_documents(self, path="data"):
        self._document_loader.path = path
        return self._document_loader.load()
    
    def _split_documents(self, documents: list):
        self._text_splitter._chunk_size = 100
        self._text_splitter._chunk_overlap = 50
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