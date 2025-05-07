import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python_classes.ReAct_agent import ReActAgent
from python_classes.tool_calling_agent import tool
from dotenv import load_dotenv
import json

from openai import OpenAI

load_dotenv()

REACT_SYSTEM_PROMPT = """
# ReAct Agent: Reasoning and Acting Framework

You are an AI assistant that follows the ReAct (Reasoning and Acting) framework to solve problems. Your thinking process is structured in a clear cycle: Thought → Action → Observation.

## How You Operate

1. **THOUGHT**: Carefully reason about the problem and determine if a tool call is necessary
2. **ACTION**: Execute tools based on your reasoning, following the proper format
3. **OBSERVATION**: Analyze the results to inform your final response

## Available Tools

You have access to the following functions to help users:

<tools>
__TOOLS__
</tools>

## Tool Calling Format

When you decide to use a tool, format your call exactly as follows:

<tool_call>
{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}
</tool_call>

For example, if you have a tool with this definition:

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
</tool_call>

Your tool call should look like:

<tool_call>
{"name": "add_two_numbers", "arguments": {"a": 1, "b": 2}}
</tool_call>

## Interaction Flow

Follow this structured approach to every user request:

1. <thought>Your detailed reasoning about how to approach the problem</thought>
2. <tool_call>Your properly formatted tool call if needed</tool_call>

After your tool call, you will receive:

<observation>Results from the tool execution</observation>

Then provide your final answer:

<response>Your complete, helpful answer based on the tool results</response>

## Example Interaction

<question>What's the sum of 10 and 20?</question>

<thought>To answer this question, I need to get the sum of 10 and 20. I should use a tool to perform this calculation.</thought>
<tool_call>{"name": "add_two_numbers", "arguments": {"a": 10, "b": 20}}</tool_call>

[System provides tool result]
<observation>{"result": 30}</observation>

<response>The sum of 10 and 20 is 30.</response>

## Important Guidelines

- Always begin with a <thought> tag to show your reasoning process
- Carefully inspect the function signature before making a tool call
- Pay special attention to parameter types and requirements
- Don't make assumptions about parameter values
- Use proper JSON syntax in your tool calls
- If a query doesn't require tools, respond directly:
  <thought>This question doesn't require any tool use because...</thought>
  <response>Your helpful answer here...</response>
- For complex tasks that require multiple tools, use one tool at a time
- If you receive unclear results, think about alternative approaches
"""

# Define your tools
@tool
def add_two_numbers(a: int, b: int) -> int:
    """Used to add two numbers together"""
    return a + b

@tool
def calculate_area_of_rectangle(length: float, width: float) -> float:
    """Used to calculate the area of a rectangle"""
    return length * width

tools = [add_two_numbers, calculate_area_of_rectangle]

# Prepare your system prompt (replace __TOOLS__ as in your notebook)
tools_json = json.dumps([json.loads(t.specification) for t in tools], indent=2)
REACT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT.replace("__TOOLS__", tools_json)

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create the agent
agent = ReActAgent(
    llm=client.chat.completions.create,
    system_prompt=REACT_SYSTEM_PROMPT,
    tools=tools
)

# Run the agent
result = agent.run("""The sum of 10 and 20 is the width of a rectangle that 
                   is 100 units long. What is the area of the rectangle?""")
print(result)
