import json
import re
from typing import List, Dict, Any, Callable
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

class ToolCallingAgent:
    """
    An agent that integrates language models with function-calling tools.
    
    This class manages the interaction between a language model and a set of tools,
    allowing the model to decide when to call functions and processing the results.
    """
    
    def __init__(self, llm: Callable, system_prompt: str, tools: List[Any]):
        """
        Initialize an AI agent with a language model and tools.
        
        Args:
            llm: Function that takes a prompt string and returns a response
            system_prompt: Instructions for guiding the language model's behavior
            tools: List of Tool objects that the agent can use
        """
        self.model = llm
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.conversation_history = []
        print(f"{Fore.GREEN}ToolCallingAgent initialized with {len(tools)} tools{Style.RESET_ALL}")
        
    def format_tools_for_prompt(self) -> str:
        """Format all available tools into a format the language model can understand."""
        tools_json = []
        for tool in self.tools.values():
            try:
                # Use the specification provided by the @tool decorator
                tools_json.append(json.loads(tool.specification))
            except (json.JSONDecodeError, AttributeError):
                # Fallback for tools without proper specification
                tools_json.append({
                    "name": tool.name,
                    "description": getattr(tool, "description", "No description available"),
                    "parameters": {"properties": {}}
                })
        
        print(f"{Fore.CYAN}Formatted {len(tools_json)} tools for LLM prompt{Style.RESET_ALL}")
        return f"<tools>\n{json.dumps(tools_json, indent=2)}\n</tools>"
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the language model's response to extract tool call requests.
        
        Args:
            response: The text response from the language model
            
        Returns:
            A list of tool call dictionaries with 'name' and 'arguments' keys
        """
        tool_calls = []
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                if "name" in tool_call and "arguments" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        if tool_calls:
            print(f"{Fore.YELLOW}Extracted {len(tool_calls)} tool call(s) from LLM response{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No tool calls found in LLM response{Style.RESET_ALL}")
                
        return tool_calls
    
    def execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute a tool based on the model's request.
        
        Args:
            tool_call: Dictionary with 'name' and 'arguments' for the tool
            
        Returns:
            The result from executing the tool
        """
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        if tool_name not in self.tools:
            print(f"{Fore.RED}Error: Tool '{tool_name}' not found{Style.RESET_ALL}")
            return f"Error: Tool '{tool_name}' not found"
            
        tool = self.tools[tool_name]
        print(f"{Fore.MAGENTA}Executing tool: {tool_name} with arguments: {json.dumps(arguments)}{Style.RESET_ALL}")
        
        # Validate and convert argument types using the tool's specification
        try:
            if hasattr(tool, "specification"):
                tool_spec = json.loads(tool.specification)
                validated_args = self.convert_argument_types(
                    {"arguments": arguments}, 
                    tool_spec
                )["arguments"]
                arguments = validated_args
                print(f"{Fore.BLUE}Arguments validated and converted to appropriate types{Style.RESET_ALL}")
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"{Fore.RED}Error validating arguments: {str(e)}{Style.RESET_ALL}")
            # Continue with original arguments if validation fails
            pass
            
        try:
            # Handle execution based on the tool interface
            # First try the execute method for tools created with the @tool decorator
            if hasattr(tool, "execute"):
                print(f"{Fore.GREEN}Calling tool.execute() method{Style.RESET_ALL}")
                return tool.execute(**arguments)
            # Then try the function attribute which is used by the @tool decorator
            elif hasattr(tool, "function"):
                print(f"{Fore.GREEN}Calling tool.function() method{Style.RESET_ALL}")
                return tool.function(**arguments)
            # Fall back to run method
            elif hasattr(tool, "run"):
                print(f"{Fore.GREEN}Calling tool.run() method{Style.RESET_ALL}")
                return tool.run(**arguments)
            # Last resort: call the tool directly if it's callable
            elif callable(tool):
                print(f"{Fore.GREEN}Calling tool directly{Style.RESET_ALL}")
                return tool(**arguments)
            else:
                print(f"{Fore.RED}Error: Tool '{tool_name}' is not callable{Style.RESET_ALL}")
                return f"Error: Tool '{tool_name}' is not callable"
        except Exception as e:
            print(f"{Fore.RED}Error executing {tool_name}: {str(e)}{Style.RESET_ALL}")
            return f"Error executing {tool_name}: {str(e)}"
    
    def convert_argument_types(self, tool_call: Dict[str, Any], tool_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert arguments to their expected types based on tool specification.
        
        Args:
            tool_call: Dictionary containing arguments to convert
            tool_spec: Tool specification with expected types
            
        Returns:
            Updated tool call with properly typed arguments
        """
        if "parameters" not in tool_spec or "properties" not in tool_spec["parameters"]:
            return tool_call
            
        properties = tool_spec["parameters"]["properties"]
        
        # Standard type converters
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
                        # Only convert if types don't match
                        if not isinstance(arg_value, converter):
                            print(f"{Fore.BLUE}Converting argument '{arg_name}' from {type(arg_value).__name__} to {expected_type}{Style.RESET_ALL}")
                            tool_call["arguments"][arg_name] = converter(arg_value)
                    except (ValueError, TypeError) as e:
                        print(f"{Fore.RED}Type conversion error for '{arg_name}': {str(e)}{Style.RESET_ALL}")
                        # Keep original value if conversion fails
                        pass
                        
        return tool_call
    
    def run(self, user_input: str) -> str:
        print(f"{Fore.WHITE}{Style.BRIGHT}=== Starting agent run with user input: '{user_input}' ==={Style.RESET_ALL}")
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Build the messages list for OpenAI API
        messages = [
            {"role": "system", "content": self.system_prompt + "\n\n" + self.format_tools_for_prompt()}
        ]
        
        # Add conversation history
        for message in self.conversation_history:
            messages.append({"role": message["role"], "content": message["content"]})

        # Get response from language model
        print(f"{Fore.CYAN}Calling language model...{Style.RESET_ALL}")
        response = self.model(model="gpt-4o", messages=messages)
        model_response = response.choices[0].message.content
        print(f"{Fore.CYAN}Received response from language model ({len(model_response)} chars){Style.RESET_ALL}")

        # Extract tool calls from response
        tool_calls = self.extract_tool_calls(model_response)
        
        # If no tool calls, return the response directly
        if not tool_calls:
            print(f"{Fore.GREEN}No tool calls needed. Returning response.{Style.RESET_ALL}")
            final_response = model_response
        else:
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
            
            # Format tool results
            results_text = "Tool results:\n"
            for res in tool_results:
                result_str = str(res["result"])
                if isinstance(res["result"], (dict, list)):
                    try:
                        result_str = json.dumps(res["result"], indent=2)
                    except:
                        pass
                    
                results_text += f"- {res['tool']}{json.dumps(res['arguments'])}: {result_str}\n"
            
            print(f"{Fore.BLUE}Formatted tool results{Style.RESET_ALL}")
            
            # Create a new message with original response and tool results
            messages.append({"role": "assistant", "content": model_response})
            messages.append({"role": "user", "content": results_text})
            
            # Get final response from language model with tool results
            print(f"{Fore.CYAN}Calling language model with tool results...{Style.RESET_ALL}")
            final_response_obj = self.model(model="gpt-4o", messages=messages)
            final_response = final_response_obj.choices[0].message.content
            print(f"{Fore.CYAN}Received final response from language model ({len(final_response)} chars){Style.RESET_ALL}")
        
        # Add final response to conversation history
        self.conversation_history.append({
            "role": "assistant", 
            "content": final_response
        })
        
        print(f"{Fore.WHITE}{Style.BRIGHT}=== Agent run completed ===={Style.RESET_ALL}")
        return final_response
        
    def reset_conversation(self):
        """Clear the conversation history."""
        print(f"{Fore.GREEN}Conversation history reset{Style.RESET_ALL}")
        self.conversation_history = []