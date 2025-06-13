# From RAG to Agents: Building Smart AI Assistants

In this workshop we

- Build a RAG application on the FAQ database
- Make it agentic
- Learn about agentic search
- Give tools to our agents
- Use PydanticAI to make it easier

# Environment

* For this workshop, all you need is Python with Jupyter.
* I use GitHub Codespaces to run it (see [here](https://www.loom.com/share/80c17fbadc9442d3a4829af56514a194)) but you can use whatever environment you like.
* Also, you need an [OpenAI account](https://openai.com/) (or an alternative provider).


# Part 1: Basic RAG Implementation

## RAG

RAG consists of 3 parts:

- Search
- Prompt 
- LLM 

So in python it looks like that:

```python
def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
```

Let's implement each component step-by-step

## Search

First, we implement a basic search function that will query our FAQ database. This function takes a query string and returns relevant documents.

We will use `minsearch` for that, so let's install
it 

```bash
pip install minsearch
```

Get the documents:

```python
import requests 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

Index them:

```python
index = AppendableIndex(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)
```

Now search:

```python
def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5,
        output_ids=True
    )

    return results
```

**Explanation:**
- This function will be the foundation of our RAG system
- It will search through the FAQ database to find relevant information
- The returned documents will be used to build context for our LLM

## Prompt

We create a function to format the search results into a structured context that our LLM can use.

```python
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt
```

**Explanation:**
- Takes search results as input
- Formats each document with its section, question, and answer
- Creates a structured text that the LLM can easily parse
- Returns a clean, formatted context string


## The RAG flow
We combine our search and LLM components into a complete RAG pipeline.

```python
from openai import OpenAI
client = OpenAI()

def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
```

**Explanation:**
- `build_prompt`: Formats the search results into a prompt
- `llm`: Makes the API call to the language model
- `rag`: Combines search and LLM into a single function
- Creates a complete pipeline from query to answer

# Making it Agentic

Agents are AI systems that can:

- Make decisions about what actions to take
- Use tools to accomplish tasks
- Maintain state and context
- Learn from previous interactions
- Work towards specific goals

A typical agentic flow consists of:
1. Receiving a user request
2. Analyzing the request and available tools
3. Deciding on the next action
4. Executing the action using appropriate tools
5. Evaluating the results
6. Either completing the task or continuing with more actions

The key difference from basic RAG is that agents can:
- Make multiple search queries
- Combine information from different sources
- Decide when to stop searching
- Use their own knowledge when appropriate
- Chain multiple actions together

So agents:

- Have access to the history of previous actions
- Can make decisions independently based on the current information and the previous actions



Let's implement this step by step.

## 2.1 Creating the Agent Prompt Template
We create a more sophisticated prompt template that enables the agent to make decisions.

```python
prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.

The CONTEXT is build with the documents from our FAQ database.
SEARCH_QUERIES contains the queries that were used to retrieve the documents
from FAQ to and add them to the context.
PREVIOUS_ACTIONS contains the actions you already performed.

At the beginning the CONTEXT is empty.

You can perform the following actions:

- Search in the FAQ database to get more data for the CONTEXT
- Answer the question using the CONTEXT
- Answer the question using your own knowledge

For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 

Don't use search queries used at the previous iterations.

Don't repeat previously performed actions.

Don't perform more than {max_iterations} iterations for a given student question.
The current iteration number: {iteration_number}. If we exceed the allowed number 
of iterations, give the best possible answer with the provided information.

Output templates:

If you want to perform search, use this template:

{
"action": "SEARCH",
"reasoning": "<add your reasoning here>",
"keywords": ["search query 1", "search query 2", ...]
}

If you can answer the QUESTION using CONTEXT, use this template:

{
"action": "ANSWER_CONTEXT",
"answer": "<your answer>",
"source": "CONTEXT"
}

If the context doesn't contain the answer, use your own knowledge to answer the question

{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}

<QUESTION>
{question}
</QUESTION>

<SEARCH_QUERIES>
{search_queries}
</SEARCH_QUERIES>

<CONTEXT> 
{context}
</CONTEXT>

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>
""".strip()
```

**Explanation:**
- Defines the agent's capabilities and constraints
- Provides clear output templates for different actions
- Includes iteration limits and action history
- Enables the agent to make decisions about searching vs. answering

## 2.2 Implementing the Agentic Search
We create the main agent function that implements the decision-making loop.

```python
def agentic_search(question):
    search_queries = []
    search_results = []
    previous_actions = []

    iteration = 0
    
    while True:
        print(f'ITERATION #{iteration}...')
    
        context = build_context(search_results)
        prompt = prompt_template.format(
            question=question,
            context=context,
            search_queries="\n".join(search_queries),
            previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
            max_iterations=3,
            iteration_number=iteration
        )
    
        print(prompt)
    
        answer_json = llm(prompt)
        answer = json.loads(answer_json)
        print(json.dumps(answer, indent=2))

        previous_actions.append(answer)
    
        action = answer['action']
        if action != 'SEARCH':
            break
    
        keywords = answer['keywords']
        search_queries = list(set(search_queries) | set(keywords))

        for k in keywords:
            res = search(k)
            search_results.extend(res)
    
        search_results = dedup(search_results)
        
        iteration = iteration + 1
        if iteration >= 4:
            break
    
        print()

    return answer
```

**Explanation:**
- Maintains state of search queries and results
- Implements the decision-making loop
- Handles search actions and result aggregation
- Includes iteration limits and action history tracking

# Part 3: Implementing Chat Assistant

## 3.1 Creating the Chat Interface
We create a basic chat interface for interacting with our agent.

```python
class ChatInterface:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages
```

**Explanation:**
- Manages the conversation history
- Stores messages with their roles
- Provides access to the message history

## 3.2 Implementing the Tools Class
We create a class to manage our agent's tools.

```python
class Tools:
    def __init__(self):
        self.tools = []

    def add_tool(self, function, description):
        self.tools.append({
            "type": "function",
            "name": description["name"],
            "description": description["description"],
            "parameters": description["parameters"]
        })

    def get_tools(self):
        return self.tools
```

**Explanation:**
- Manages the available tools
- Stores tool descriptions and functions
- Provides access to tool information

## 3.3 Defining Tool Descriptions
We define the JSON schemas for our tools.

```python
search_description = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

add_entry_description = {
    "type": "function",
    "name": "add_entry",
    "description": "Add an entry to the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to be added to the FAQ database",
            },
            "answer": {
                "type": "string",
                "description": "The answer to the question",
            }
        },
        "required": ["question", "answer"],
        "additionalProperties": False
    }
}
```

**Explanation:**
- Defines the structure of each tool
- Specifies required parameters
- Includes descriptions for better understanding
- Ensures proper validation of inputs

## 3.4 Creating the Chat Assistant
We implement the main chat assistant class.

```python
class ChatAssistant:
    def __init__(self, tools, developer_prompt, chat_interface, client):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        self.client = client

    def run(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Chat ended.")
                break

            self.chat_interface.add_message("user", user_input)
            
            # Process the input and generate response
            response = self.process_input(user_input)
            
            self.chat_interface.add_message("assistant", response)
            print(f"Assistant: {response}")
```

**Explanation:**
- Manages the chat interaction
- Processes user input
- Generates responses using the agent
- Maintains conversation state

## 3.5 Adding Tools to the Agent
We register our tools with the agent.

```python
tools = chat_assistant.Tools()
tools.add_tool(search, search_description)
tools.add_tool(add_entry, add_entry_description)
```

**Explanation:**
- Creates a tools manager
- Registers each tool with its description
- Makes tools available to the agent
- Enables the agent to use the tools effectively

# Part 4: Using PydanticAI

## 4.1 Automating Tool Definitions
Instead of manually defining tool descriptions, we can use Pydantic to automate this process.

```python
@chat_agent.tool
def search_tool(ctx: RunContext, query: str) -> Dict[str, str]:
    """
    Search the FAQ for relevant entries matching the query.
    """
    return search(query)

@chat_agent.tool
def add_entry_tool(ctx: RunContext, question: str, answer: str) -> None:
    """
    Add a new question-answer entry to FAQ.
    """
    return add_entry(question, answer)
```

**Explanation:**
- Uses Pydantic decorators to automatically generate tool descriptions
- Type hints define the parameter structure
- Docstrings provide the tool descriptions
- Eliminates the need for manual JSON schema definitions

## 4.2 Benefits of Using Pydantic
- Automatic type validation
- Self-documenting code
- Reduced boilerplate
- Better IDE support
- Easier maintenance

# Example Usage

## Basic RAG Query
```python
question = "how do I prepare for the course?"
answer = rag(question)
```

## Agentic Search
```python
question = "what do I need to do to be successful at module 1?"
result = agentic_search(question)
```

## Using the Chat Assistant
```python
developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.
Use FAQ if your own knowledge is not sufficient to answer the question.
At the end of each response, ask the user a follow up question based on your answer.
""".strip()

chat_interface = chat_assistant.ChatInterface()
chat = chat_assistant.ChatAssistant(
    tools=tools,
    developer_prompt=developer_prompt,
    chat_interface=chat_interface,
    client=client
)
chat.run()
```

# Resources

* [PydanticAI Documentation](https://ai.pydantic.dev/)
* [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
* [GitHub Codespaces](https://github.com/features/codespaces)
