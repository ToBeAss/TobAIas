import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback

class LLM_Wrapper:
    def __init__(self, model="azure-gpt-4o-mini", **kwargs):
        load_dotenv() # Load environment variables from .env file
        self._llm = self._init_llm(model, **kwargs)
        self._prompt_cost = 0

    def _init_llm(self, model, **kwargs):
        if model == "azure-gpt-4o-mini":
            return AzureChatOpenAI(
                azure_deployment="gpt-4o-mini",
                api_version="2024-08-01-preview",
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", None),
                timeout=kwargs.get("timeout", None),
                max_retries=kwargs.get("max_retries", 2),
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

    def invoke(self, prompt):
        for method in ["invoke", "generate", "call", "__call__", "chat"]:
            if hasattr(self._llm, method):
                with get_openai_callback() as callback:
                    response = getattr(self._llm, method)(prompt)
                    self._prompt_cost = callback.total_cost
                    return response

        raise AttributeError(f"No valid invocation method found for {type(self._llm).__name__}.")
    
    def get_prompt_cost(self):
        return self._prompt_cost