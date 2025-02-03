from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper

class Agent:
    def __init__(self, name: str, llm: LLM_Wrapper, vector_db: VectorDB_Wrapper):
        self.name = name
        self._messages = []
        self.add_instruction(f"Your name is {self.name}")
        self._llm = llm
        self._db = vector_db
        self._prompt_cost = 0
        self._total_cost = 0

    def invoke(self, user_input):
        # Step 0: Rephrase user input for better search
        prompt = f"Conversation context:\n{self._messages}\n\nUser Query:\n{user_input}\n\nRephrase the question to better be used in a RAG retrieval search. Include important keywords that could be relevant to the user's question for the retrieval to work better."
        message = ("human", prompt)
        enhanced_query = self._llm.invoke(message).content
        print(f"Enhanced query: {enhanced_query}")

        # Step 1: Retrieve relevant documents from the vector database
        retrieved_chunks = self._db.invoke(enhanced_query)
        sources = [chunk.id for chunk in retrieved_chunks]
        contexts = [chunk.page_content for chunk in retrieved_chunks]

        # Step 2: Formulate a structured prompt using retrieved context and add it to the message history
        context = f"\n{retrieved_chunks}" if retrieved_chunks else "I couldn't find any relevant documents. I'll answer based on my general knowledge."
        prompt = f"Context:\n{context}\n\nUser Query:\n{user_input}\n\nAnswer the question based on the context. If no context is given, answer the question based on your own training data or the context of the conversation history."
        message = ("human", prompt)
        self._messages.append(message)

        # Step 3: Get response from the LLM
        result = self._llm.invoke(self._messages)
        response = result.content
        message = ("ai", response)
        self._messages.append(message)

        # Optional Step: Calculate total cost
        self._prompt_cost = self._llm.get_prompt_cost()
        self._total_cost += self._prompt_cost

        # Constuct return dictionary
        result = {
            "content": response,
            "sources": sources,
            "contexts": contexts,
            "cost": self._prompt_cost,
        }

        print(f"📊: {result}")
        return result
    
    def add_instruction(self, instruction: str):
        message = ("system", instruction)
        self._messages.append(message)

    def get_prompt_cost(self):
        return self._prompt_cost
    
    def get_total_cost(self):
        return self._total_cost