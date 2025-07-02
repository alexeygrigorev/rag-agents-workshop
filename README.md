# From RAG to Agents: Building Smart AI Assistants

Video: https://www.youtube.com/watch?v=GH3lrOsU3AU


In this workshop we

- Build a RAG application on the FAQ database
- Make it agentic
- Learn about agentic search
- Give tools to our agents
- Use PydanticAI to make it easier

For this workshop, we will use the following FAQ documents from [our free courses](https://datatalks.club/blog/guide-to-free-online-courses-at-datatalks-club.html):

* [Machine Learning Zoomcamp](https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit?tab=t.0) 
* [Data Engineering Zoomcamp](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit?tab=t.0#heading=h.edeyusfgl4b7)
* [MLOps Zoomcamp](https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit?tab=t.0)

# Environment

* For this workshop, all you need is Python with Jupyter.
* I use GitHub Codespaces to run it (see [here](https://www.loom.com/share/80c17fbadc9442d3a4829af56514a194)) but you can use whatever environment you like.
* Also, you need an [OpenAI account](https://openai.com/) (or an alternative provider).

## Setting up Github Codespaces

Github Codespaces is the recommended environment for this 
workshop. But you can use any other environment with
Jupyter Notebook, including your laptop and Google Colab.

* Create a repository on GitHub, initialize it with README.md
* Add the OpenAI key:
    * Go to Settings -> Secrets and Variables (under Security) -> Codespaces
    * Click "New repository secret"
    * Name: `OPENAI_API_KEY`, Secret: your key
    * Click "Add secret"
* Create a codespace
    * Click "Code" 
    * Select the "Codespaces" tab
    * "Create codespaces on main"

## Installing required libraries

Next we need to install the required libraries:

```bash
pip install jupyter openai minsearch requests
```

# Part 0: Basic RAG

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
from minsearch import AppendableIndex

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

- This function is the foundation of our RAG system
- It looks up in the FAQ to find relevant information
- The result is used to build context for the LLM

## Prompt

We create a function to format the search results into
a structured context that our LLM can use.

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

- Takes search results
- Formats each document
- Put everything in a prompt


## The RAG flow

We add a call to an LLM and combine everything
into a complete RAG pipeline:

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


# Part 1: Agentic RAG

Now let's make our flow agentic


## Agents and Agentic flows 

Agents are AI systems that can:

- Make decisions about what actions to take
- Use tools to accomplish tasks
- Maintain state and context
- Learn from previous interactions
- Work towards specific goals

Agentic flow is not necessarily a completely independent agent,
but it can still make some decisions during the flow execution

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

So in agentic RAG, the system

- has access to the history of previous actions
- makes decisions independently based on the current information
  and the previous actions

Let's implement this step by step.

## Making RAG more agentic

First, we'll take the prompt we have so far and make it 
a little more "agentic":

- Tell the LLM that it can answer the question directly or look up context
- Provide output templates
- Show clearly what's the source of the answer



```python
prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
At the beginning the context is EMPTY.

<QUESTION>
{question}
</QUESTION>

<CONTEXT> 
{context}
</CONTEXT>

If CONTEXT is EMPTY, you can use our FAQ database.
In this case, use the following output template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>"
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}
""".strip()
```


Let's use it:

```python
question = "how do I run docker on gentoo?"
context = "EMPTY"

prompt = prompt_template.format(question=question, context=context)
print(prompt)

answer = llm(prompt)
print(answer)
```

We may get something like that:

```json
{
"action": "ANSWER",
"answer": "To run Docker on Gentoo, you'll first need to ensure that you have the necessary system prerequisites and then install Docker. Follow these steps:\n\n1. **Install Docker**: You can install Docker using the Portage package management system. Open a terminal and run:\n   ```\n   sudo emerge app-emulation/docker\n   ```\n\n2. **Start the Docker service**: You'll need to start the Docker service to begin using it. You can do this with:\n   ```\n   sudo rc-service docker start\n   ```\n\n3. **Add your user to the Docker group**: This will allow you to run Docker commands without `sudo`. Run the following command:\n   ```\n   sudo usermod -aG docker $USER\n   ```\n   Log out and back in for this change to take effect.\n\n4. **Test your installation**: You can verify that Docker is running by executing:\n   ```\n   docker run hello-world\n   ```\n\nIf Docker is installed correctly, this command will download a test image and run it, displaying a confirmation message.\n\nMake sure your system is up to date and review the Gentoo Docker wiki page for any additional configurations specific to your setup.","source": "OWN_KNOWLEDGE"
}
```

But if we ask for something that it can't answer:

```python
question = "how do I join the course?"
context = "EMPTY"

prompt = prompt_template.format(question=question, context=context)
answer = llm(prompt)
print(answer)
```

We will get this:

```json
{
"action": "SEARCH",
"reasoning": "The context is empty, and I need to find information on how to join the course."
}
```

Let's implement make the search:

```python
search_results = search(question)
context = build_context(search_results)
prompt = prompt_template.format(question=question, context=context)
print(prompt)
```

Here, `build_context` is a helper function from the previous
code:

```python
def build_context(search_results):
    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    return context.strip()
```

Now we can query it again:

```python
answer = llm(prompt)
print(answer)
```

And get:

```json
{
"action": "ANSWER",
"answer": "To join the course, you need to register before the start date using the provided registration link. Even if you're unable to register before the course begins, you can still participate by submitting homework, but be mindful of project deadlines. Make sure to also join the course's Telegram channel and the DataTalks.Club's Slack for announcements and updates.",
"source": "CONTEXT"
}
```

Let's put this together:

- First attempt to answer it with our know knowledge
- If needed, do the lookup and then answer

```python
def agentic_rag_v1(question):
    context = "EMPTY"
    prompt = prompt_template.format(question=question, context=context)
    answer_json = llm(prompt)
    answer = json.loads(answer_json)
    print(answer)

    if answer['action'] == 'SEARCH':
        print('need to perform search...')
        search_results = search(question)
        context = build_context(search_results)
        
        prompt = prompt_template.format(question=question, context=context)
        answer_json = llm(prompt)
        answer = json.loads(answer_json)
        print(answer)

    return answer
```

Test it:

```python
agentic_rag_v1('how do I join the course?')
agentic_rag_v1('how patch KDE under FreeBSD?')
```


## Part 2: Agentic search

So far we had two actions only: search and answer.

But we can let our "agent" formulate one or more 
search queries - and do it for a few iterations until
we found an answer


Let's build a prompt:

- List available actions:
    - Search in FAQ
    - Answer using own knowledge
    - Answer using information extracted from FAQ 
- Provide access to the previous actions
- Have clear stop criteria (no more than X iterations)
- We also specify the output format, so it's easier to parse it

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

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>",
"keywords": ["search query 1", "search query 2", ...]
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER_CONTEXT",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}

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

Our code becomes more complicated. For the first iteration,
we have:


```python
question = "how do I join the course?"

search_queries = []
search_results = []
previous_actions = []
context = build_context(search_results)

prompt = prompt_template.format(
    question=question,
    context=context,
    search_queries="\n".join(search_queries),
    previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
    max_iterations=3,
    iteration_number=1
)
print(prompt)
```

```python
answer_json = llm(prompt)
answer = json.loads(answer_json)
print(json.dumps(answer, indent=2))
```

Output:

```json
{
  "action": "SEARCH",
  "reasoning": "I need to find specific information on how to join the course, as this information is not present in the current CONTEXT.",
  "keywords": [
    "how to join the course",
    "course enrollment process",
    "register for the course"
  ]
}
```

We need to save the actions, so let's do it:

```python
previous_actions.append(answer)
```

Save the search queries:

```python
keywords = answer['keywords']
search_queries.extend(keywords)

And perform the search:

```python
for k in keywords:
    res = search(k)
    search_results.extend(res)
```

Some of the search results will be duplicates, so we need 
to remove them:


```python
def dedup(seq):
    seen = set()
    result = []
    for el in seq:
        _id = el['_id']
        if _id in seen:
            continue
        seen.add(_id)
        result.append(el)
    return result

search_results = dedup(search_results)
```

Now let's make another iteration - use the same code as previously, but remove variable initialization
and increase the iteration number:


```python
# question = "how do I join the course?"

# search_queries = []
# search_results = []
# previous_actions = []

context = build_context(search_results)

prompt = prompt_template.format(
    question=question,
    context=context,
    search_queries="\n".join(search_queries),
    previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
    max_iterations=3,
    iteration_number=2
)
print(prompt)

answer_json = llm(prompt)
answer = json.loads(answer_json)
print(json.dumps(answer, indent=2))
```

Let's put everything together:

```python
question = "what do I need to do to be successful at module 1?"

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
```

Or, as a function:

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

Test it:

```python
agentic_search('how do I prepare for the course?')
```


# Part 3: Function calling 

## Function calling in OpenAI

We put all this logic inside our prompt. 

But OpenAI and other providers provide a convenient 
API for adding extra functionality like search.

* https://platform.openai.com/docs/guides/function-calling

It's called "function calling" - you define functions
that the model can call, and if it decides to make a call,
it returns structured output for that.

For example, let's take our `search` function:

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

We describe it like that:

```python
search_tool = {
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
```

Here we have:

- `name`: `search`
- `description`: when to use it
- `parameters`: all the arguments that the function can take 
  and their description

In order to use function calling, we'll use a newer API - 
the "responses" API (not "chat completions" as previously):

```python
question = "How do I do well in module 1?"

developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.
""".strip()

tools = [search_tool]

chat_messages = [
    {"role": "developer", "content": developer_prompt},
    {"role": "user", "content": question}
]

response = client.responses.create(
    model='gpt-4o-mini',
    input=chat_messages,
    tools=tools
)
response.output
```

If the model thinks we should make a function call, it will
tell us:

```
[ResponseFunctionToolCall(arguments='{"query":"How to do well in module 1"}', call_id='call_AwYwOak5Ljeidh4HbE3RxMZJ', name='search', type='function_call', id='fc_6848604db67881a298ec38121c1555ef0dee5fa0cdb59912', status='completed')]
```

Let's make a call to `search`:

```python
calls = response.output
call = calls[0]
call

call_id = call.call_id
call_id

f_name = call.name
f_name

arguments = json.loads(call.arguments)
arguments
```

Using `f_name` we can find the function we need:

```python
f = globals()[f_name]
```

And invoke it with the arguments:

```python
results = f(**arguments)
```

Now let's save the results as json:

```python
search_results = json.dumps(results, indent=2)
print(search_results)
```

And save both the response and the result of the function call:


```python
chat_messages.append(call)

chat_messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": search_results,
})
```

Now `chat_messages` contains both the call description 
(so it keeps track of history) and the results

Let's make another call to the model:

```python
response = client.responses.create(
    model='gpt-4o-mini',
    input=chat_messages,
    tools=tools
)
```

This time it should be the response (but also can be another call):

```python
r = response.output[0]
print(r.content[0].text)
```

## Making multiple calls

What if we want to make multiple calls? Change the developer prompt a little:

```python
developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.
If you look up something in FAQ, convert the student question into multiple queries.
""".strip()

chat_messages = [
    {"role": "developer", "content": developer_prompt},
    {"role": "user", "content": question}
]

response = client.responses.create(
    model='gpt-4o-mini',
    input=chat_messages,
    tools=tools
)
```

This time let's start to organize the code a little:


Let's organize our code a little.

First, create a function `do_call`:

```python
def do_call(tool_call_response):
    function_name = tool_call_response.name
    arguments = json.loads(tool_call_response.arguments)

    f = globals()[function_name]
    result = f(**arguments)

    return {
        "type": "function_call_output",
        "call_id": tool_call_response.call_id,
        "output": json.dumps(result, indent=2),
    }
```

Now iterate over responses:

```python
for entry in response.output:
    chat_messages.append(entry)
    print(entry.type)

    if entry.type == 'function_call':      
        result = do_call(entry)
        chat_messages.append(result)
    elif entry.type == 'message':
        print(entry.text) 
```

First call will probably be function call, so let's do another one:

```python
response = client.responses.create(
    model='gpt-4o-mini',
    input=chat_messages,
    tools=tools
)

for entry in response.output:
    chat_messages.append(entry)
    print(entry.type)
    print()

    if entry.type == 'function_call':      
        result = do_call(entry)
        chat_messages.append(result)
    elif entry.type == 'message':
        print(entry.content[0].text) 
```

This one is a text response.

## Putting everything together

But what if it's not?

Let's make two loops: 

- First is the main Q&A loop - ask question, get back the answer
- Second is the request loop - send requests until there's a message reply from the API

```python
developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.

Use FAQ if your own knowledge is not sufficient to answer the question.
When using FAQ, perform deep topic exploration: make one request to FAQ,
and then based on the results, make more requests.

At the end of each response, ask the user a follow up question based on your answer.
""".strip()

chat_messages = [
    {"role": "developer", "content": developer_prompt},
]
```

```python
while True: # main Q&A loop
    question = input() # How do I do my best for module 1?
    if question == 'stop':
        break

    message = {"role": "user", "content": question}
    chat_messages.append(message)

    while True: # request-response loop - query API till get a message
        response = client.responses.create(
            model='gpt-4o-mini',
            input=chat_messages,
            tools=tools
        )

        has_messages = False
        
        for entry in response.output:
            chat_messages.append(entry)
        
            if entry.type == 'function_call':      
                print('function_call:', entry)
                print()
                result = do_call(entry)
                chat_messages.append(result)
            elif entry.type == 'message':
                print(entry.content[0].text)
                print()
                has_messages = True

        if has_messages:
            break
```


It's also possible that there's both message and tool calls,
but we'll ignore this case for now. (It's easy to fix -
just check if there are no function calls, and only then 
ask the user for input.)


Let's make it a bit nicer using HTML:

```python
from IPython.display import display, HTML
import markdown # pip install markdown

    

developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.

Use FAQ if your own knowledge is not sufficient to answer the question.

At the end of each response, ask the user a follow up question based on your answer.
""".strip()

chat_messages = [
    {"role": "developer", "content": developer_prompt},
]

# Chat loop
while True:
    
    if question.strip().lower() == 'stop':
        print("Chat ended.")
        break
    print()

    message = {"role": "user", "content": question}
    chat_messages.append(message)

    while True:  # inner request loop
        response = client.responses.create(
            model='gpt-4o-mini',
            input=chat_messages,
            tools=tools
        )

        has_messages = False

        for entry in response.output:
            chat_messages.append(entry)

            if entry.type == "function_call":
                result = do_call(entry)
                chat_messages.append(result)
                display_function_call(entry, result)

            elif entry.type == "message":
                display_response(entry)
                has_messages = True

        if has_messages:
            break
```

## Using multiple tools

What if we also want to use this chat app to add new entries to the FAQ?
We'll need another function for it:

```python
def add_entry(question, answer):
    doc = {
        'question': question,
        'text': answer,
        'section': 'user added',
        'course': 'data-engineering-zoomcamp'
    }
    index.append(doc)
```

Description:

```python
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

We can just reuse the preivous code. But we can also clean it up
and make it more modular. 

See the result in [`chat_assistant.py`](chat_assistant.py)

You can download it using `wget`:

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/rag-agents-workshop/refs/heads/main/chat_assistant.py
```

Here we define multiple classes:

- `Tools` - manages function tools for the agent
    - `add_tool(function, description)`: Register a function with its description
    - `get_tools()`: Return list of registered tool descriptions
    - `function_call(tool_call_response)`: Execute a function call and return result
- `ChatInterface` - handles user input and display formatting
    - `input()`: Get user input
    - `display(message)`: Print a message
    - `display_function_call(entry, result)`: Show function calls in HTML format
    - `display_response(entry)`: Display AI responses with markdown
- `ChatAssistant` - main orchestrator for chat conversations.
    - `__init__(tools, developer_prompt, chat_interface, client)`: Initialize assistant
    - `gpt(chat_messages)`: Make OpenAI API calls
    - `run()`: Main chat loop handling user input and AI responses

Let's use it:

```python
import chat_assistant

tools = chat_assistant.Tools()
tools.add_tool(search, search_tool)

tools.get_tools()

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
```

And run it:

```python
chat.run()
```

Now let's add the new tool:

```python
tools.add_tool(add_entry, add_entry_description)
tools.get_tools()
```

And talk with the assistant:

- How do I do well for module 1?
- Add this back to FAQ

And check that it's in the index:

```python
index.docs[-1]
```


# Part 4: Using PydanticAI

## Installing and using PydanticAI

There are frameworks that make it easier for us to create
agents

One of them is [PydanticAI](https://ai.pydantic.dev/agents/):

```bash
pip install pydantic-ai
```

Let's import it:

```python
from pydantic_ai import Agent, RunContext
```

And create an agent:

```python
chat_agent = Agent(  
    'openai:gpt-4o-mini',
    system_prompt=developer_prompt
)
```

Now we can use it to automate tool description:


```python
from typing import Dict


@chat_agent.tool
def search_tool(ctx: RunContext, query: str) -> Dict[str, str]:
    """
    Search the FAQ for relevant entries matching the query.

    Parameters
    ----------
    query : str
        The search query string provided by the user.

    Returns
    -------
    list
        A list of search results (up to 5), each containing relevance information 
        and associated output IDs.
    """
    print(f"search('{query}')")
    return search(query)


@chat_agent.tool
def add_entry_tool(ctx: RunContext, question: str, answer: str) -> None:
    """
    Add a new question-answer entry to FAQ.

    This function creates a document with the given question and answer, 
    tagging it as user-added content.

    Parameters
    ----------
    question : str
        The question text to be added to the index.

    answer : str
        The answer or explanation corresponding to the question.

    Returns
    -------
    None
    """
    return add_entry(question, answer)
```

It reads the functions' docstrings to automatically
create function definition, so we don't need to worry about it.

Let's use it:

```python
user_prompt = "I just discovered the course. Can I join now?"
agent_run = await chat_agent.run(user_prompt)
print(agent_run.output)
```

If want to learn more about implementing chat
applications with Pydantic AI:

- https://ai.pydantic.dev/message-history/
- https://ai.pydantic.dev/examples/chat-app/


# Wrap up

In this workshop, we took our RAG application
and made it agentic, by first tweaking the prompts,
and then using the "function calling" functionality
from OpenAI.

At the end, we put all the logic into the `chat_assistant.py ` script, and also explored PydanticAI to make it simpler.

What's next:

- MCP
- Agent deployment
- Agent monitoring

