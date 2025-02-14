import os, shutil
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

class VectorDB_Wrapper:
    instances = 0 # Class variable to track instances

    def __init__(self, db="chroma", embedding="azure-text-embedding-3-large", search_type="similarity_score_threshold", k=5, fetch_k=20, score_threshold=0.4):
        load_dotenv() # Load environment variables from .env file
        self._dir = f"{db}_{VectorDB_Wrapper.instances}"
        VectorDB_Wrapper.instances += 1 # Increment on each instantiation
        self._embedding = self._init_embedding(embedding)
        self._db = self._init_db(db)
        self._retriever = self._init_retriever(search_type, k, fetch_k, score_threshold)
        self._document_loader = self._init_document_loader()
        self._text_splitter = self._init_text_splitter()

    def _init_document_loader(self):
        return PyPDFDirectoryLoader("")
    
    def _init_text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 400,
            length_function = len,
            is_separator_regex = False,
        )

    def _init_embedding(self, embedding):
        if embedding == "ollama-nomic-embed-text":
            return OllamaEmbeddings(
                model="nomic-embed-text",
            )
        elif embedding == "azure-text-embedding-3-large":
            return AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-3-large",
                api_version = "2023-05-15",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            raise ValueError(f"Unsupported embedding model: {embedding}")

    def _init_db(self, db):
        if db == "chroma":
            return Chroma(
                persist_directory=self._dir,
                embedding_function=self._embedding,
            )
        else:
            raise ValueError(f"Unsupported database: {db}")
        
    def _init_retriever(self, search_type, k, fetch_k, score_threshold):
        return {
            "search_type": search_type,
            "k": k,
            "fetch_k": fetch_k,
            "score_threshold": score_threshold,
        }
    
    def _read_files(self, path: str):
        documents = []
        for root, _, files in os.walk(path): # Recursively walk through directories
            for filename in files:
                file_path = os.path.join(root, filename)

                if os.path.isfile(file_path): # Ensure it is a file
                    try:
                        with open(file_path, "r", encoding="utf-8") as file:
                            content = file.read()
                            doc = Document(page_content=content, metadata={"source": file_path, "page": 0})
                            documents.append(doc)
                    except UnicodeDecodeError as e:
                        print(f"Unicode error in file: {file_path}\nError: {e}")
                        return  # Stop execution so we can inspect the problematic file
        return documents
        
    def _load_documents(self, path: str):
        self._document_loader.path = path
        return self._document_loader.load()
    
    def _split_documents(self, documents: list):
        return self._text_splitter.split_documents(documents)
    
    def _index_chunks(self, chunks: list):
        # Creates IDs like "dir/file.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        previous_page = None

        for chunk in chunks:
            current_page = f"{chunk.metadata.get("source")}:{chunk.metadata.get("page")}"

            if current_page == previous_page:
                current_chunk += 1
            else:
                current_chunk = 0

            chunk.id = f"{current_page}:{current_chunk}"
            previous_page = current_page
        return chunks
            
    def _embed_chunks(self, chunks: list):
        stored_ids = set(self._db.get()["ids"])
        print(f"💾 Number of stored chunks in DB: {len(stored_ids)}")

        new_chunks = []
        for chunk in chunks:
            if chunk.id not in stored_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new chunks: {len(new_chunks)}")
            return self._db.add_documents(new_chunks)
        else:
            print("✅ No new chunks to add")
            return []
        
    def embed_data(self, data: list):
        chunks = self._split_documents(data)
        indexed_chunks = self._index_chunks(chunks)
        vectors = self._embed_chunks(indexed_chunks)
        return vectors

    def embed_documents(self, path = "data"):
        documents = self._load_documents(path)
        return self.embed_data(documents)
    
    def clear_db(self):
        if os.path.exists(self._dir):
            shutil.rmtree(self._dir)
    
    def invoke(self, prompt):
        if self._retriever["search_type"] == "similarity_score_threshold":
            return self._db.similarity_search_with_relevance_scores(prompt, self._retriever["k"], score_threshold=self._retriever["score_threshold"])
        elif self._retriever["search_type"] == "mmr":
            retriever = self._db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self._retriever["k"],
                    "fetch_k": self._retriever["fetch_k"],
                },
            )
            return retriever.invoke(prompt)
        else:
            raise AttributeError(f"No valid invocation method found for {type(self._retriever).__name__}.")