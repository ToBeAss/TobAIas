from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper
from lib.agent import Agent
from lib.orchestrator_agent import Orchestrator_Agent
from langchain.schema.document import Document

# CREATE ORCHESTRATOR
llm = LLM_Wrapper("azure-gpt-4o-mini")
db = VectorDB_Wrapper("chroma", "chroma", "azure-text-embedding-3-large")
orchestrator = Orchestrator_Agent("TobAIas", llm, db)
orchestrator.add_instruction("Du skal alltid svare på norsk.")

# GIVE INFORMATION ABOUT ITS AGENTS (AUTOMATIC IN THE FUTURE)
agent_info = [
    Document(
        "AI Agenten 'Aloe Vera' har tilgang til kodebasen til et IoT prosjekt, hvor jordfuktigheten til en aloe vera plante blir målt og sendt over Discord via et webhook. Den kan svare på spørsmål om prosjektet og foreslå endringer i koden.", 
        metadata={"source": "agent_info", "page": "aloe_vera"}
    )
]
db.embed_data(agent_info)

# CREATE AGENTS AND SET THEM AS CHILDREN OF ORCHESTRATOR
aloe_llm = LLM_Wrapper("azure-gpt-4o-mini")
aloe_db = VectorDB_Wrapper("aloe_chroma", "chroma", "azure-text-embedding-3-large")
aloe_agent = Agent("aloe_vera", aloe_llm, aloe_db)
aloe_agent.add_instruction("Du har tilgang til kodebasen til et IoT prosjekt, hvor jordfuktigheten til en aloe vera plante blir målt og sendt over Discord via et webhook. Den kan svare på spørsmål om prosjektet og foreslå endringer i koden.")

aloe_data = aloe_db._read_files("aloe_vera_data")
aloe_db.embed_data(aloe_data)

orchestrator.add_child(aloe_agent)

# INVOKE ORCHESTRATOR
user_input = "Hva står det i main.py fila om discord webhooks i aloe vera kodeprosjektet?"
print(f"👤: {user_input}")
response = orchestrator.invoke(user_input)
print(f"🤖: {response}")

while True:
    print()
    user_input = input("👤: ")
    response = orchestrator.invoke(user_input)
    print(f"🤖: {response}")