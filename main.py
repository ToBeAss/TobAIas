from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper
from lib.agent import Agent

# CREATE AGENT
llm = LLM_Wrapper("azure-gpt-4o-mini")
db = VectorDB_Wrapper("chroma")
agent = Agent("TobAIas", llm, db)

db.embed_documents()

while True:
    user_input = input("👤: ")

    # INVOKE AGENT WITH USER INPUT
    response = agent.invoke(user_input)

    # PRINT OUT RESULT
    print(f"🤖: {response.content}")
    print(f"💰: ${format(agent.get_prompt_cost(), '.6f')} / ${format(agent.get_total_cost(), '.6f')}")