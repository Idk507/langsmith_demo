# User Input

#    ↓

# CLARITY CHECK AGENT   (new)

#    ↓ (if clear)

# Planner

#    ↓

# Action (memory/LLM first, web only if needed)

#    ↓

# Thought (1 pass)

#    ↓

# Final Answer

#    ↓

# Memory Update

Suffix: """
"""
Purpose:
- Maintain user profile memory (name, preferences, etc.)
- Maintain conversation/topic memory
- Retrieve memory before answering
- Update memory after each run
- Support follow-up questions
- Low latency (single action-thought cycle)

Architecture:
User Input → Load Memory → Action → Final Answer → Memory Update → END
"""

import os
import uuid
from typing import TypedDict, Annotated, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient

from huggingface_hub import InferenceClient
from ddgs import DDGS

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

assert os.getenv("HF_TOKEN"), "HF_TOKEN must be set"
assert os.getenv("LANGCHAIN_PROJECT"), "LANGCHAIN_PROJECT must be set"

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
langsmith_client = LangSmithClient()
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

print("LangSmith project:", PROJECT_NAME)

# ============================================================================
# CONSTANTS
# ============================================================================

PROFILE_NS = ("user_profile",)
CONVO_NS = ("user_conversation",)

# ============================================================================
# LLM WRAPPER
# ============================================================================

def call_llm(prompt: str, *, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=512) -> str:
    """Call HuggingFace LLM with LangSmith tracing"""
    with trace(
        name="llm_call",
        project_name=PROJECT_NAME,
        metadata={"model": model},
        tags=["llm"],
    ):
        resp = hf_client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content.strip()





# ============================================================================
# WEB SEARCH
# ============================================================================

def web_search(query: str, k: int = 2) -> str:
    """Search DuckDuckGo with LangSmith tracing"""
    with trace(
        name="web_search",
        project_name=PROJECT_NAME,
        metadata={"query": query},
        tags=["tool", "web"],
    ):
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=k):
                    results.append(
                        f"{r.get('title','')}\n{r.get('body','')}\nSource: {r.get('href','')}"
                    )
        except Exception as e:
            return f"Search error: {str(e)}"

        return "\n\n".join(results) if results else "No results found."

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    profile_memory: Optional[str]
    conversation_memory: Optional[str]
    observation: Optional[str]
    final_answer: Optional[str]
    profile_update: Optional[str]
    conversation_update: Optional[str]
    user_id: str
# ============================================================================
# MEMORY HELPERS
# ============================================================================

def get_memory(store: InMemoryStore, namespace: tuple, user_id: str) -> str:
    try:
        item = store.get(namespace, user_id)
        if item is None:
            return ""
        if hasattr(item, "value"):
            return item.value
        return item
    except Exception:
        return ""

def update_memory(store: InMemoryStore, namespace: tuple, user_id: str, new_value: str, existing: str):
    if not new_value or new_value.strip().lower() == "none":
        return
    combined = (existing + "\n" + new_value).strip() if existing else new_value
    store.put(namespace, user_id, combined)

# ============================================================================
# GRAPH NODES
# ============================================================================

def memory_load_node(state: AgentState, store: InMemoryStore) -> Dict[str, Any]:
    profile = get_memory(store, PROFILE_NS, state["user_id"])
    convo = get_memory(store, CONVO_NS, state["user_id"])
    return {
        "profile_memory": profile,
        "conversation_memory": convo,
    }

def action_node(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    user_query = last_message.content if isinstance(last_message, HumanMessage) else last_message["content"]

    profile_memory = state.get("profile_memory", "")
    convo_memory = state.get("conversation_memory", "")

    prompt = f"""
You are a helpful AI assistant with memory.

User profile memory:
{profile_memory or "None"}

Conversation memory:
{convo_memory or "None"}

User question:
{user_query}

Instructions:
- Use memory if relevant
- If you cannot answer confidently, respond ONLY with the word: INSUFFICIENT
"""

    llm_answer = call_llm(prompt)

    if "insufficient" in llm_answer.lower().strip():
        observation = web_search(user_query)
    else:
        observation = llm_answer

    return {"observation": observation}

def final_answer_node(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    user_query = last_message.content if isinstance(last_message, HumanMessage) else last_message["content"]

    profile_memory = state.get("profile_memory", "")
    convo_memory = state.get("conversation_memory", "")
    observation = state.get("observation", "")

    answer_prompt = f"""
User profile memory:
{profile_memory or "None"}

Conversation memory:
{convo_memory or "None"}

User question:
{user_query}

Information:
{observation}

Provide a clear helpful answer:
"""

    answer = call_llm(answer_prompt)

    # PROFILE MEMORY EXTRACTION
    profile_prompt = f"""
Extract ONLY explicit personal facts about the user.

Existing profile memory:
{profile_memory or "None"}

User message:
{user_query}

Rules:
- Only personal facts (name, location, preference)
- No topic summaries
- If nothing new, return NONE

New profile memory:
"""

    profile_update = call_llm(profile_prompt)

    # CONVERSATION MEMORY EXTRACTION
    conversation_prompt = f"""
Summarize the user's question and assistant answer in ONE factual sentence.

User question:
{user_query}

Assistant answer:
{answer}

Rules:
- Do NOT include personal data
- Do NOT hallucinate
- Write as: "User asked about ..."

Conversation memory:
"""

    conversation_update = call_llm(conversation_prompt)

    return {
        "final_answer": answer,
        "profile_update": profile_update,
        "conversation_update": conversation_update,
    }

def memory_persist_node(state: AgentState, store: InMemoryStore) -> Dict[str, Any]:
    profile_existing = state.get("profile_memory", "")
    convo_existing = state.get("conversation_memory", "")

    update_memory(store, PROFILE_NS, state["user_id"], state.get("profile_update"), profile_existing)
    update_memory(store, CONVO_NS, state["user_id"], state.get("conversation_update"), convo_existing)

    return {}


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_graph_with_memory():
    store = InMemoryStore()
    builder = StateGraph(AgentState)

    def _memory_load(state: AgentState):
        return memory_load_node(state, store)

    def _memory_persist(state: AgentState):
        return memory_persist_node(state, store)

    builder.add_node("memory_load", _memory_load)
    builder.add_node("action", action_node)
    builder.add_node("final_answer", final_answer_node)
    builder.add_node("memory_persist", _memory_persist)

    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "action")
    builder.add_edge("action", "final_answer")
    builder.add_edge("final_answer", "memory_persist")
    builder.add_edge("memory_persist", END)

    graph = builder.compile(checkpointer=MemorySaver(), store=store)
    return graph, store

# ============================================================================
# RUN AGENT
# ============================================================================

def run_agent(query: str, graph, store, thread_id: str, user_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "profile_memory": None,
        "conversation_memory": None,
        "observation": None,
        "final_answer": None,
        "profile_update": None,
        "conversation_update": None,
        "user_id": user_id,
    }

    result = graph.invoke(initial_state, config)

    print("\nFINAL ANSWER:\n", result["final_answer"])

    print("\nPROFILE MEMORY:\n", get_memory(store, PROFILE_NS, user_id) or "None")
    print("\nCONVERSATION MEMORY:\n", get_memory(store, CONVO_NS, user_id) or "None")

    return result


if __name__ == "__main__":
    graph, store = build_graph_with_memory()

    thread_id = str(uuid.uuid4())
    user_id = "user_1"

    print("LONG-TERM MEMORY AGENT (DUAL MEMORY)")
    print("Type 'quit' to exit")

    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() == "quit":
            break
        run_agent(user_query, graph, store, thread_id, user_id)
