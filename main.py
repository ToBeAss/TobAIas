from lib.llm_wrapper import LLM_Wrapper
from lib.agent import Agent

llm = LLM_Wrapper("azure-gpt-4o-mini")
agent = Agent("TobAIas", llm, None)

while True:
    user_input = input("👤: ")
    response = agent.invoke(user_input)
    print(f"🤖: {response.content}")
    print(f"💰: ${format(agent.get_prompt_cost(), '.6f')} / ${format(agent.get_total_cost(), '.6f')}")
