import json
import re
from typing import Callable, List, Dict, Any
from colorama import init, Fore, Style

init(autoreset=True)

class Tool:
    def __init__(self, name: str, function: Callable, specification: str):
        self.name = name
        self.function = function
        self.specification = specification

    def execute(self, **kwargs):
        return self.function(**kwargs)

def extract_function_metadata(function: Callable) -> Dict[str, Any]:
    metadata = {
        "name": function.__name__,
        "description": function.__doc__,
        "parameters": {"properties": {}}
    }
    parameter_types = {
        param_name: {"type": param_type.__name__}
        for param_name, param_type in function.__annotations__.items()
        if param_name != "return"
    }
    metadata["parameters"]["properties"] = parameter_types
    return metadata

def tool(function: Callable) -> Tool:
    metadata = extract_function_metadata(function)
    return Tool(
        name=metadata.get("name"),
        function=function,
        specification=json.dumps(metadata)
    )

class ReActAgent:
    def __init__(self, llm: Callable, system_prompt: str, tools: List[Tool], max_iterations: int = 10):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.conversation_history = []
        self.max_iterations = max_iterations

    def format_tools_for_prompt(self) -> str:
        tools_json = [json.loads(tool.specification) for tool in self.tools.values()]
        return f"<tools>\n{json.dumps(tools_json, indent=2)}\n</tools>"

    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        tool_calls = []
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                if "name" in tool_call and "arguments" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                print(f"{Fore.RED}Failed to decode tool call: {match}{Style.RESET_ALL}")
                continue
        return tool_calls

    def extract_final_response(self, response: str) -> str:
        match = re.search(r"<response>(.*?)</response>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def convert_argument_types(self, tool_call: Dict[str, Any], tool_spec: Dict[str, Any]) -> Dict[str, Any]:
        if "parameters" not in tool_spec or "properties" not in tool_spec["parameters"]:
            return tool_call
        properties = tool_spec["parameters"]["properties"]
        type_mapping = {
            "int": int,
            "str": str,
            "bool": bool,
            "float": float,
            "integer": int,
            "string": str,
            "boolean": bool,
            "number": float
        }
        for arg_name, arg_value in tool_call["arguments"].items():
            if arg_name in properties and "type" in properties[arg_name]:
                expected_type = properties[arg_name]["type"]
                if expected_type in type_mapping:
                    converter = type_mapping[expected_type]
                    try:
                        if not isinstance(arg_value, converter):
                            print(f"{Fore.BLUE}Converting argument '{arg_name}' from {type(arg_value).__name__} to {expected_type}{Style.RESET_ALL}")
                            tool_call["arguments"][arg_name] = converter(arg_value)
                    except (ValueError, TypeError) as e:
                        print(f"{Fore.RED}Type conversion error for '{arg_name}': {str(e)}{Style.RESET_ALL}")
                        pass
        return tool_call

    def execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        if tool_name not in self.tools:
            print(f"{Fore.RED}Error: Tool '{tool_name}' not found{Style.RESET_ALL}")
            return f"Error: Tool '{tool_name}' not found"
        tool = self.tools[tool_name]
        print(f"{Fore.MAGENTA}Executing tool: {tool_name} with arguments: {json.dumps(arguments)}{Style.RESET_ALL}")
        try:
            tool_spec = json.loads(tool.specification)
            validated_args = self.convert_argument_types({"arguments": arguments}, tool_spec)["arguments"]
            arguments = validated_args
        except Exception as e:
            print(f"{Fore.RED}Error validating arguments: {str(e)}{Style.RESET_ALL}")
            pass
        try:
            result = tool.execute(**arguments)
            print(f"{Fore.GREEN}Tool '{tool_name}' executed successfully. Result: {result}{Style.RESET_ALL}")
            return result
        except Exception as e:
            print(f"{Fore.RED}Error executing {tool_name}: {str(e)}{Style.RESET_ALL}")
            return f"Error executing {tool_name}: {str(e)}"

    def run(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        messages = [
            {"role": "system", "content": self.system_prompt + "\n\n" + self.format_tools_for_prompt()}
        ]
        for message in self.conversation_history:
            messages.append({"role": message["role"], "content": message["content"]})

        iterations = 0
        last_response = None

        while iterations < self.max_iterations:
            iterations += 1
            print(f"{Fore.CYAN}{Style.BRIGHT}--- Iteration {iterations} ---{Style.RESET_ALL}")

            # Get response from language model
            response = self.llm(model="gpt-4o", messages=messages)
            model_response = response.choices[0].message.content
            print(f"{Fore.YELLOW}Model response:\n{model_response}{Style.RESET_ALL}")

            # Check for final response
            final_response = self.extract_final_response(model_response)
            if final_response is not None:
                # print(f"{Fore.GREEN}{Style.BRIGHT}Final response found!{Style.RESET_ALL}")
                self.conversation_history.append({"role": "assistant", "content": model_response})
                return final_response

            # Extract tool calls from response
            tool_calls = self.extract_tool_calls(model_response)
            if not tool_calls:
                print(f"{Fore.GREEN}No tool calls found. Returning model response.{Style.RESET_ALL}")
                self.conversation_history.append({"role": "assistant", "content": model_response})
                return model_response

            # Execute tools and collect results
            tool_results = []
            for i, tool_call in enumerate(tool_calls):
                print(f"{Fore.MAGENTA}{Style.BRIGHT}Executing tool call {i+1}/{len(tool_calls)}{Style.RESET_ALL}")
                result = self.execute_tool(tool_call)
                tool_results.append({
                    "tool": tool_call.get("name"),
                    "arguments": tool_call.get("arguments"),
                    "result": result
                })

            # Format tool results for the next prompt
            results_text = "Tool results:\n"
            for res in tool_results:
                result_str = str(res["result"])
                if isinstance(res["result"], (dict, list)):
                    try:
                        result_str = json.dumps(res["result"], indent=2)
                    except Exception:
                        pass
                results_text += f"- {res['tool']}{json.dumps(res['arguments'])}: {result_str}\n"

            print(f"{Fore.BLUE}Tool results to be sent to LLM:\n{results_text}{Style.RESET_ALL}")

            # Add model response and tool results to messages for next iteration
            messages.append({"role": "assistant", "content": model_response})
            messages.append({"role": "user", "content": results_text})

        print(f"{Fore.RED}Max iterations reached without a final response.{Style.RESET_ALL}")
        return "Max iterations reached without a final response."

    def reset_conversation(self):
        print(f"{Fore.GREEN}Conversation history reset{Style.RESET_ALL}")
        self.conversation_history = []