import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python_classes.tool_calling_agent import ToolCallingAgent, tool
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
def calculate_area_of_rectangle(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

print(calculate_area_of_rectangle.specification)


# System prompt for the LLM based on the function_calling_prompt for the add_two_numbers function
system_prompt = """You are an AI assistant with function calling capabilities. Your primary role is to interpret user requests and call appropriate functions when needed.

When presented with function definitions within <tools></tools> XML tags, you should:

1. Analyze the user's request to determine if a function call is necessary
2. Carefully inspect the function signature, paying close attention to parameter types and requirements
3. When calling a function, format your response using the following structure:

<tool_call>
{
    "name": "add_two_numbers",
    "description": "Used to add two numbers together",
    "parameters": {
        "properties": {
            "a": {
                "type": "int"
            },
            "b": {
                "type": "int"
            }
        }
    }
}

Expected output:

<tool_call>
{"name": "add_two_numbers", "arguments": {"a": 1, "b": 2}}
</tool_call>
"""


agent = ToolCallingAgent(
    llm=client.chat.completions.create, 
    system_prompt=system_prompt, 
    tools=[calculate_area_of_rectangle]
)

response = agent.run("What is the area of a rectangle with a length of 10 and a width of 20?")
print(response)