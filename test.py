from lib.llm_wrapper import LLM_Wrapper
from lib.vectordb_wrapper import VectorDB_Wrapper
from lib.agent import Agent

llm = LLM_Wrapper()
db = VectorDB_Wrapper()
agent = Agent("TobAIas", llm, db)

agent.add_data("data")
agent.add_instruction("Du skal alltid svare på norsk.")

while True:
    print()
    user_input = input("👤: ")
    response = agent.invoke(user_input)

    print(f"🔍main: {response["query"]}")
    print(f"🤖main: {response["content"]}")
    for source in response["sources"]:
        print(f"🔗main: {source}")
    print(f"💰: ${format(response["cost"], '.6f')}")