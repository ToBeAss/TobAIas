from lib.agent import Agent
from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper

class Orchestrator_Agent(Agent):
    def __init__(self, name: str, llm: LLM_Wrapper, vector_db: VectorDB_Wrapper):
        super().__init__(name, llm, vector_db)
        self._children = []

    def add_child(self, agent: Agent):
        self._children.append(agent)

    def invoke(self, user_input):
        # Step 0: Rephrase user input for better search
        prompt = f"Conversation context:\n{self._instructions}\n{self._messages}\n\nUser Query:\n{user_input}\n\nRephrase the question to better be used in a RAG retrieval search. Include important keywords that could be relevant to the user's question for the retrieval to work better."
        enhanced_query = self._llm.invoke(prompt).content

        # Step 1: Retrieve relevant documents from the vector database
        retrieved_chunks = self._db.invoke(enhanced_query)
        contexts = []
        sources = []
        for chunk, score in retrieved_chunks:
            contexts.append(chunk.page_content)
            sources.append(f"{chunk.id} ({format(score, ".2f")})")
        for source in sources:
            print(f"🔗: {source}")

        # Step 2: Check if any of the given sources contain "agent_info" and add them to the call list
        call_list = []
        for source in sources:
            if "agent_info" in source:
                if "aloe_vera" in source:
                    call_list.append("aloe_vera")

        # Step 3: Call agents
        results = []
        if len(call_list):
            for call in call_list:
                print(f"🤖: Let me ask {call} about this issue...")
                for agent in self._children:
                    if call == agent.name:
                        result = agent.invoke(user_input)
                        results.append((agent.name, result))
            print(f"📊: {results}")
            for result in results:
                for source in result[1]["sources"]:
                    print(f"🔗: {source}")

        #Step 4: Invoke the original agent with added contexts
        if len(results):
            context = result
        else:
            context = retrieved_chunks
        prompt = f"Context:\n{context}\n\nUser Query:\n{user_input}\n\nAnswer the question based on the context. If no context is given, answer the question based on your own training data or the context of the conversation history."
        return self._llm.invoke(prompt).content