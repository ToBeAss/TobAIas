from lib.llm_wrapper import LLM_Wrapper
import json

llm = LLM_Wrapper()


def multiply(a: int, b: int):
    '''
    A function that multiplies two numbers, parameters a and b.
    '''
    return a * b

def divide(a: int, b: int):
    '''
    A function that divides the first parameter a, by the second parameter b.
    '''
    return a / b


tools = {
    "multiply": multiply,
    "divide": divide,
    }

llm_with_tools = llm._llm.bind_tools(list(tools.values()))

response = llm_with_tools.invoke("What is 100 multiplied by 5? Divide the result by 2. The first parameter of the second operation needs to be the result of the first operation. Name the parameter 42.")
print(response)

# Check if the response contains a tool call
if response.additional_kwargs.get("tool_calls"):
    for tool_call in response.additional_kwargs["tool_calls"]:
        function_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])  # Parse JSON arguments

        # Execute tool function
        if function_name in tools:
            result = tools[function_name](**args)
            print(f"Result: {result}")  # Print or process the result
        else:
            print(f"Error: Tool {function_name} not found.")
else:
    print(response.content)  # Print regular response if no tool was called