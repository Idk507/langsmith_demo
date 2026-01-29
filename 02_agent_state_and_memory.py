import uuid 
from typing import Annotated,TypedDict,Any

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.store.memory import InMemoryStore

class AgentState(TypedDict):
    messages : Annotated[list, add_messages]
    # additional fields for agent 
    user_id : str 
    last_answer : str | None
# Create a short-term memory saver
short_term_saver = InMemorySaver()
print("Short-term memory saver initialized:", short_term_saver)
#Short-term memory via a checkpointer saves the entire graph state after every step, enabling multi-turn context.
# Long-term memory store
long_term_store = InMemoryStore()
print("Long-term memory store initialized:", long_term_store)


namespace = ("user_1", "global")
#Long-term memory stores JSON documents organized by namespace and key.
# build graph with memory components

def add_message_node(state : AgentState) -> dict[str, Any]:
    return {
        "messages": state["messages"],
    }


def add_sample_answer(state : AgentState) -> dict[str, Any]:
    return {
        "last_answer": state["last_answer"],
    }


builder = StateGraph(AgentState)

builder.add_node("add_message", add_message_node)
builder.add_node("sample_answer", add_sample_answer)
builder.add_edge(START, "add_message")
#edge fixed sequence
builder.add_edge("add_message", "sample_answer")

graph = builder.compile(
    checkpointer = short_term_saver,
    store = long_term_store,

)
# test short term memory 

thread_id = str(uuid.uuid4())

# First invocation
config = {"configurable": {"thread_id": thread_id}}

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hello agent"}], 
     "user_id": "user_1", "last_answer": None},
    config
)

print("After first invocation:", result)

# Second invocation on same thread continues history
result2 = graph.invoke(
    {"messages": [{"role": "user", "content": "Another query"}], 
     "user_id": "user_1", "last_answer": None},
    config
)

print("After second invocation (short-term memory):", result2)

# test long term memory

# Save long-term info
key = "greeting_fact"
value = {"fact": "user likes pizza"}

long_term_store.put(namespace, key, value)
print("Stored long-term memory:", value)

# Retrieve it
loaded = long_term_store.get(namespace, key)
print("Retrieved long-term memory:", loaded)

def read_long_term_memory(user_id: str, key: str) -> Any:
    try:
        return long_term_store.get((user_id, "global"), key)
    except Exception as e:
        print("Memory read fail:", e)
        return None

print("Read greeting_fact:", read_long_term_memory("user_1", "greeting_fact"))
