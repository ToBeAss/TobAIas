from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper
from lib.agent import Agent

# CREATE AGENT
llm = LLM_Wrapper("azure-gpt-4o-mini")
db = VectorDB_Wrapper("chroma", "azure-text-embedding-3-large")
agent = Agent("TobAIas", llm, db)

db.embed_documents("data")
agent.add_instruction("Du skal alltid svare på norsk.")

while True:
    print()
    user_input = input("👤: ")

    # INVOKE AGENT WITH USER INPUT
    response = agent.invoke(user_input)

    # PRINT OUT RESULT
    print(f"🔍: {response["query"]}")
    print(f"🤖: {response["content"]}")
    for source in response["sources"]:
        print(f"🔗: {source}")
    print(f"💰: ${format(response["cost"], '.6f')}")