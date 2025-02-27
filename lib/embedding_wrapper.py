import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

class Embedding_Wrapper:
    instances = 0 # Class variable to track instances
    
    def __init__(self, model_name="azure-text-embedding-3-large", **kwargs):
        load_dotenv() # Load environment variables from .env file
        self._model = self._init_model(model_name, **kwargs)

    def _init_model(self, model_name, **kwargs):
        if model_name == "azure-text-embedding-3-large":
            AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-3-large",
                api_version = "2023-05-15",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        
    def embed_documents(self):
        pass