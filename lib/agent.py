from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper

class Agent:
    def __init__(self, name: str, llm: LLM_Wrapper, vector_db: VectorDB_Wrapper):
        self.name = name
        self._instructions = []
        self._messages = []
        self._HISTORY_LENGTH = 10
        self.add_instruction(f"Your name is {self.name}")
        self._llm = llm
        self._db = vector_db
        self._prompt_cost = 0
        self._total_cost = 0

    def _add_message(self, message: str, role: str):
        data = (role, message)
        self._messages.append(data)
        if len(self._messages) > self._HISTORY_LENGTH:
            self._messages.pop(0)

    def add_instruction(self, instruction: str):
        data = ("system", instruction)
        self._instructions.append(data)

    def invoke(self, user_input):
        # Step 0: Rephrase user input for better search
        prompt = f"Conversation context:\n{self._instructions}\n{self._messages}\n\nUser Query:\n{user_input}\n\nRephrase the question to better be used in a RAG retrieval search. Include important keywords that could be relevant to the user's question for the retrieval to work better."
        enhanced_query = self._llm.invoke(prompt).content

        # Step 1: Retrieve relevant documents from the vector database
        retrieved_chunks = self._db.invoke(enhanced_query)
        contexts = []
        sources = []
        for chunk in retrieved_chunks:
            contexts.append(chunk.page_content)
            sources.append(chunk.id)

        # Step 2: Formulate a structured prompt using retrieved context and add it to the message history
        context = f"{retrieved_chunks}" if retrieved_chunks else "I couldn't find any relevant documents. I'll answer based on my general knowledge."
        prompt = f"Context:\n{context}\n\nUser Query:\n{user_input}\n\nAnswer the question based on the context. If no context is given, answer the question based on your own training data or the context of the conversation history."
        self._add_message(prompt, "human")
        # see if I can get away with only storing the user input

        # Step 3: Get response from the LLM
        result = self._llm.invoke(self._instructions + self._messages)
        response = result.content
        self._add_message(response, "ai")

        # Optional Step: Calculate total cost
        self._prompt_cost = self._llm.get_prompt_cost()
        self._total_cost += self._prompt_cost

        # Constuct return dictionary
        return {
            "content": response,
            "input": user_input,
            "query": enhanced_query,
            "contexts": contexts,
            "sources": sources,
            "cost": self._prompt_cost,
        }

    def get_prompt_cost(self):
        return self._prompt_cost
    
    def get_total_cost(self):
        return self._total_cost