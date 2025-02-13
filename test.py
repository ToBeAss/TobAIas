from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper
from lib.agent import Agent, Orchestrator_Agent

# CREATE ORCHESTRATOR
llm = LLM_Wrapper("azure-gpt-4o-mini")
db = VectorDB_Wrapper("chroma", "chroma", "azure-text-embedding-3-large", score_threshold=0.4)
orchestrator = Orchestrator_Agent("TobAIas", llm, db)
orchestrator.add_instruction("Du skal alltid svare på norsk.")

# CREATE CHILD AGENT
aloe_llm = LLM_Wrapper("azure-gpt-4o-mini")
aloe_db = VectorDB_Wrapper("aloe_chroma", "chroma", "azure-text-embedding-3-large", score_threshold=0.1)
aloe_agent = Agent("aloe_vera", aloe_llm, aloe_db)
aloe_agent.add_instruction("Du har tilgang til kodebasen til et IoT prosjekt, hvor jordfuktigheten til en aloe vera plante blir målt av en sensor og sendt over Discord via et webhook. Den kan svare på spørsmål om prosjektet og foreslå endringer i koden.")

# SPECIFY DATA PATH FOR THE AGENT
aloe_data = aloe_db._read_files("aloe_vera_data")
aloe_db.embed_data(aloe_data)

# ADD AGENT AS A CHILD OF ORCHESTRATOR LAST (THE ORCHESTRATOR EMBEDS THE AGENT'S CURRENT INSTRUCTIONS TO ITS DB)
orchestrator.add_child(aloe_agent)


while True:
    print()
    user_input = input("👤: ")

    # INVOKE ORCHESTRATOR WITH USER INPUT
    response = orchestrator.invoke(user_input)

    # PRINT OUT RESULT
    print()
    print(f"🤖main: {response["content"]}")
    for source in response["sources"]:
        print(f"🔗main: {source}")