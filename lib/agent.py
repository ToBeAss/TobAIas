from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper
from langchain.schema.document import Document

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
        self._db._db._persist_directory += f"_{self.name}"

    def _add_message(self, message: str, role: str):
        data = (role, message)
        self._messages.append(data)
        if len(self._messages) > self._HISTORY_LENGTH:
            self._messages.pop(0)

    def add_instruction(self, instruction: str):
        data = ("system", instruction)
        self._instructions.append(data)

    def add_data(self, path: str):
        return self._db.embed_documents(path)

    def invoke(self, user_input):
        # Step 0: Rephrase user input for better search
        prompt = f"Conversation context:\n{self._instructions}\n{self._messages}\n\nUser Query:\n{user_input}\n\nRephrase the question to better be used in a RAG retrieval search. Include important keywords that could be relevant to the user's question for the retrieval to work better."
        enhanced_query = self._llm.invoke(prompt).content

        # Step 1: Retrieve relevant documents from the vector database
        retrieved_chunks = self._db.invoke(enhanced_query)
        contexts = []
        sources = []
        if "score" in retrieved_chunks:
            for chunk, score in retrieved_chunks:
                contexts.append(chunk.page_content)
                sources.append((chunk.id, format(score, ".2f")))
        else:
            for chunk in retrieved_chunks:
                contexts.append(chunk.page_content)
                sources.append(chunk.id)

        # Step 2: Formulate a structured prompt using retrieved context and add it to the message history
        context = f"{retrieved_chunks}" if retrieved_chunks else "I couldn't find any relevant documents. I'll answer based on my general knowledge."
        prompt = f"Context:\n{context}\n\nUser Query:\n{user_input}\n\nAnswer the question based on the context. If no context is given, answer the question based on your own training data or the context of the conversation history."

        # Step 3: Get response from the LLM
        result = self._llm.invoke(self._instructions + self._messages + [prompt])
        response = result.content
        self._add_message(user_input, "human")
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
    


class Orchestrator_Agent(Agent):
    def __init__(self, name: str, llm: LLM_Wrapper, vector_db: VectorDB_Wrapper):
        super().__init__(name, llm, vector_db)
        self._children = {}

    def add_child(self, agent: Agent):
        info = [Document(
            f"AI Agenten {agent.name} har følgende instruksjoner:\n{agent._instructions}", 
            metadata={"source": "agent_info", "page": agent.name}
        )]
        self._db.embed_data(info)
        self._children[agent.name] = agent

    def _invoke_child(self, child_name: str, prompt):
        return self._children[child_name].invoke(prompt)

    def invoke(self, user_input):
        # Step 0: Rephrase user input for better search
        prompt = f"Conversation context:\n{self._instructions}\n{self._messages}\n\nUser Query:\n{user_input}\n\nRephrase the question to better be used in a RAG retrieval search. Include important keywords that could be relevant to the user's question for the retrieval to work better."
        enhanced_query = self._llm.invoke(prompt).content

        # Step 1: Retrieve relevant documents from the vector database
        retrieved_chunks = self._db.invoke(enhanced_query)
        contexts = []
        sources = []
        structured_sources = []
        for chunk, score in retrieved_chunks:
            contexts.append(chunk.page_content)
            sources.append(f"{chunk.id} ({format(score, ".2f")})")
            structured_sources.append((chunk.metadata["source"], chunk.metadata["page"]))

        # Step 2: Check if any of the given sources contain "agent_info" and add the respective agent names to the call list
        call_list = []
        for source in structured_sources:
            if "agent_info" in source[0]:
                call_list.append(source[1])

        # Step 3: Call agents
        results = []
        children_sources = []
        if len(call_list):
            for call in call_list:
                print(f"🤖main: Let me ask {call} about this issue...")
                print()
                result = self._invoke_child(call, user_input)
                results.append((call, result))
            #print(f"📊children: {results}")
            for result in results:
                for source in result[1]["sources"]:
                    children_sources.append(source)
                    
        # Step 4: Formulate a structured prompt using retrieved context and add it to the message history
        context = f"{results}" if results else retrieved_chunks
        prompt = f"Context:\n{context}\n\nUser Query:\n{user_input}\n\nAnswer the question based on the context. If no context is given, answer the question based on your own training data or the context of the conversation history."
        
        # Step 5: Get response from the LLM
        result = self._llm.invoke(self._instructions + self._messages + [prompt])
        response = result.content
        self._add_message(user_input, "human")
        self._add_message(response, "ai")

        # Optional Step: Calculate total cost
        children_cost = 0
        for child in self._children.values():
            children_cost += child._prompt_cost
        self._prompt_cost = self._llm.get_prompt_cost() + children_cost
        self._total_cost += self._prompt_cost

        # Constuct return dictionary
        return {
            "content": response,
            "input": user_input,
            "query": enhanced_query,
            "contexts": contexts,
            "sources": sources + children_sources,
            "cost": self._prompt_cost,
        }