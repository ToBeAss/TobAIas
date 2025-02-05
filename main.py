from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper
from lib.agent import Agent
from langchain.schema.document import Document

# CREATE AGENT
llm = LLM_Wrapper("azure-gpt-4o-mini")
db = VectorDB_Wrapper("chroma", "chroma", "azure-text-embedding-3-large")
agent = Agent("TobAIas", llm, db)

db.embed_documents("data")
agent.add_instruction("Du skal alltid svare på norsk.")

# Multi Agent
agent_info = Document(
    "AI Agenten 'Aloe Vera' har tilgang til kodebasen til et IoT prosjekt, hvor jordfuktigheten til en aloe vera plante blir målt og sendt over Discord via et webhook. Den kan svare på spørsmål om prosjektet og foreslå endringer i koden.", 
    metadata={"source": "agent_info", "page": "aloe_vera"})
data = [agent_info]
db.embed_data(data)

print("Aloe: ...")

aloe_vera_llm = LLM_Wrapper("azure-gpt-4o-mini")
aloe_vera_db = VectorDB_Wrapper("aloe_chroma", "chroma", "azure-text-embedding-3-large")
aloe_vera_agent = Agent("aloe_vera", aloe_vera_llm, aloe_vera_db)

aloe_vera_data = aloe_vera_db._read_files("aloe_vera_data")
aloe_vera_db.embed_data(aloe_vera_data)

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

    # Invoke aloe agent
    print("Aloe: ...")
    response = aloe_vera_agent.invoke(user_input)
    print(f"🔍: {response["query"]}")
    print(f"🤖: {response["content"]}")
    for source in response["sources"]:
        print(f"🔗: {source}")
    print(f"💰: ${format(response["cost"], '.6f')}")