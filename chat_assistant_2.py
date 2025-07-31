import json
import inspect
import markdown
from IPython.display import display, HTML

from openai import OpenAI


def shorten(text, max_length=50):
    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def generate_description(function):
    """
    Generate a tool description schema for a given function using its docstring and signature.
    """

    # Get function name and docstring
    name = function.__name__
    doc = inspect.getdoc(function) or "No description provided."

    # Get function signature
    sig = inspect.signature(function)
    properties = {}
    required = []

    for param in sig.parameters.values():
        param_name = param.name
        param_type = param.annotation if param.annotation != inspect._empty else str

        # Map Python types to JSON schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            dict: "object",
            list: "array"
        }
        json_type = type_map.get(param_type, "string")  # default to string

        properties[param_name] = {
            "type": json_type,
            "description": f"{param_name} parameter"
        }

        # Consider all parameters required unless they have a default
        if param.default == inspect._empty:
            required.append(param_name)

    return {
        "type": "function",
        "name": name,
        "description": doc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    }


class Tools:

    def __init__(self):
        self.tools = {}
        self.functions = {}
    
    def add_tool(self, function, description=None):
        """
            tool_description = {
                "type": "function",               # This identifies it as a function/tool.
                "name": "<function_name>",       # Name of the function as it will be exposed.
                "description": "<function_description>",  # A human-readable description of the function's purpose.
                "parameters": {
                    "type": "object",            # Indicates the function accepts a JSON object as input.
                    "properties": {
                        "<param_name>": {
                            "type": "<type>",    # e.g., "string", "integer", etc.
                            "description": "<description>"  # Human-readable parameter description.
                        },
                        ...
                    },
                    "required": ["<param1>", "<param2>"],  # List of required parameters.
                    "additionalProperties": False          # Disallows additional unexpected parameters.
                }
            }
        """
        if description is None:
            description = generate_description(function)
        self.tools[function.__name__] = description
        self.functions[function.__name__] = function

    def add_tools(self, instance):
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not name.startswith("_"):  # skip private and special methods
                self.add_tool(method)
    
    def get_tools(self):
        return list(self.tools.values())

    def function_call(self, tool_call_response):
        args = json.loads(tool_call_response.arguments)

        f_name = tool_call_response.name
        f = self.functions[f_name]
        
        call_id = tool_call_response.call_id
    
        results = f(**args)
        output_json = json.dumps(results)
        
        call_output = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_json,
        }
    
        return call_output

class IPythonChatInterface:

    def input(self):
        question = input('User:').strip()
        return question

    def display(self, content):
        print(content)

    def display_function_call(self, name, arguments, output):
        short_arguments = shorten(arguments)

        call_html = f"""
            <details>
                <summary>Function call: <tt>{name}({short_arguments})</tt></summary>
                <div>
                    <b>Call</b>
                    <pre>{arguments}</pre>
                </div>
                <div>
                    <b>Output</b>
                    <pre>{output}</pre>
                </div>
            </details>
        """
        display(HTML(call_html))

    def display_response(self, md_content):
        html_content = markdown.markdown(md_content)

        html = f"""
            <div>
                <div><b>Assistant:</b></div>
                <div>{html_content}</div>
            </div>
        """
        display(HTML(html))

class ChatAssistant:

    def __init__(self, tools: Tools, developer_prompt: str, interface: IPythonChatInterface, openai_client: OpenAI):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.interface = interface
        self.openai_client = openai_client

    def run(self) -> None:
        chat_messages = [
            {"role": "developer", "content": self.developer_prompt},
        ]
        
        while True: # Q&A loop
            question = self.interface.input()
        
            if question.lower() == 'stop':
                self.interface.display('chat ended')
                break
        
            chat_messages.append({"role": "user", "content": question})
        
            while True:
                response = self.openai_client.responses.create(
                    model='gpt-4o-mini',
                    input=chat_messages,
                    tools=self.tools.get_tools()
                )
        
                has_function_call = False
                
                for entry in response.output:    
                    chat_messages.append(entry)
                
                    if entry.type == 'message':
                        md_content = entry.content[0].text
                        self.interface.display_response(md_content)
        
                    if entry.type == 'function_call':
                        call_output = self.tools.function_call(entry)
        
                        name = entry.name
                        arguments = entry.arguments
                        output = call_output['output']
                        self.interface.display_function_call(name, arguments, output)
        
                        chat_messages.append(call_output)
                        has_function_call = True
        
                if not has_function_call:
                    break

