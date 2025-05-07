from typing import Callable, Dict, Any
import json

def extract_function_metadata(function: Callable) -> Dict[str, Any]:
    """
    Creates a metadata dictionary from a function's signature.
    
    Parameters:
        function: The target function to analyze
        
    Returns:
        A dictionary containing the function's name, documentation, and parameter specifications
    """
    # Initialize the basic structure
    metadata = {
        "name": function.__name__,
        "description": function.__doc__,
        "parameters": {"properties": {}}
    }
    
    # Extract parameter types (excluding return annotation)
    parameter_types = {
        param_name: {"type": param_type.__name__}
        for param_name, param_type in function.__annotations__.items()
        if param_name != "return"
    }
    
    # Add parameters to metadata
    metadata["parameters"]["properties"] = parameter_types
    return metadata

def convert_argument_types(tool_invocation: Dict[str, Any], function_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all arguments match their expected types according to the function specification.
    
    Parameters:
        tool_invocation: Dictionary with the tool name and arguments
        function_spec: Dictionary containing the expected parameter types
        
    Returns:
        Updated tool invocation with correctly typed arguments
    """
    expected_params = function_spec["parameters"]["properties"]
    
    # Type conversion mapping
    type_converters = {
        "int": int,
        "str": str,
        "bool": bool,
        "float": float
    }
    
    # Convert each argument to its expected type if needed
    for arg_name, arg_value in tool_invocation["arguments"].items():
        target_type = expected_params[arg_name].get("type")
        if not isinstance(arg_value, type_converters[target_type]):
            tool_invocation["arguments"][arg_name] = type_converters[target_type](arg_value)
            
    return tool_invocation

class Tool:
    """
    Wrapper class that encapsulates a function as a callable tool.
    
    Attributes:
        name: Tool identifier
        function: The underlying function
        specification: JSON-formatted function metadata
    """
    def __init__(self, name: str, function: Callable, specification: str):
        self.name = name
        self.function = function
        self.specification = specification
        
    def __str__(self):
        return self.specification
        
    def execute(self, **kwargs):
        """
        Runs the wrapped function with the provided arguments.
        
        Parameters:
            **kwargs: Arguments to pass to the function
            
        Returns:
            The function's result
        """
        return self.function(**kwargs)

def tool(function: Callable) -> Tool:
    """
    Decorator that transforms a regular function into a Tool instance.
    
    Parameters:
        function: The function to convert into a tool
        
    Returns:
        A fully configured Tool object
    """
    def create_tool():
        metadata = extract_function_metadata(function)
        return Tool(
            name=metadata.get("name"),
            function=function,
            specification=json.dumps(metadata)
        )
    
    return create_tool()