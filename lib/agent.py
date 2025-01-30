from lib.llm_wrapper import LLM_Wrapper

class Agent:
    def __init__(self, name: str, llm: LLM_Wrapper, vector_db):
        self.name = name
        self._messages = []
        self.add_instruction(f"Your name is {self.name}")
        self._llm = llm
        self._vector_db = vector_db
        self._prompt_cost = 0
        self._total_cost = 0

    def invoke(self, user_input):
        #Step 1: Retrieve relevant documents from the vector database
        retrieved_docs = []

        # Step 2: Formulate a structured prompt using retrieved context and add it to the message history
        context = "\n".join(retrieved_docs) if retrieved_docs else "I couldn't find any relevant documents. I'll answer based on my general knowledge."
        prompt = f"Context:\n{context}\n\nUser Query:\n{user_input}\n\nAnswer the question based on the context. If no context is given, answer the question based on your own training data or the context of the conversation history."
        message = ("human", prompt)
        self._messages.append(message)

        # Step 3: Get response from the LLM
        response = self._llm.invoke(self._messages)

        # Extra Step: Calculate total cost
        self._prompt_cost = self._llm.get_prompt_cost() # + self._vector_db.get_prompt_cost()
        self._total_cost += self._prompt_cost

        return response
    
    def add_instruction(self, instruction: str):
        message = ("system", instruction)
        self._messages.append(message)

    def get_prompt_cost(self):
        return self._prompt_cost
    
    def get_total_cost(self):
        return self._total_cost