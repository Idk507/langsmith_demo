import os
import uuid
import json
import hashlib
import time
from typing import TypedDict, Annotated, Optional, Dict, Any

import shelve
import numpy as np
from diskcache import Cache as DiskCache
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from huggingface_hub import InferenceClient
from ddgs import DDGS

from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient


# =========================
# ENV
# =========================

assert os.getenv("HF_TOKEN")
assert os.getenv("LANGCHAIN_PROJECT")
assert os.getenv("LANGCHAIN_API_KEY")

PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
langsmith_client = LangSmithClient()

PROFILE_NS = ("user_profile",)
CONVO_NS = ("user_conversation",)

CACHE_TTL = 3600
SHELVE_PATH = "./query_cache.db"
MEMORY_PATH = "./persistent_memory.db"

disk_cache = DiskCache("./cache_dir")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# UTILS
# =========================

def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def compute_query_hash(query: str, user_id: str) -> str:
    key = f"{query.lower().strip()}|{user_id}"
    return hashlib.md5(key.encode()).hexdigest()


# =========================
# CACHE
# =========================

def semantic_cache_lookup(query: str, user_id: str, threshold: float = 0.8):
    key = compute_query_hash(query, user_id)

    if key in disk_cache:
        return disk_cache[key]

    query_emb = embedding_model.encode(query)

    with shelve.open(SHELVE_PATH) as db:
        for _, cached in db.items():
            if cached["user_id"] != user_id:
                continue
            sim = cosine_similarity(query_emb, np.array(cached["embedding"]))
            if sim >= threshold:
                return cached
    return None


def semantic_cache_write(query: str, user_id: str, answer: str):
    emb = embedding_model.encode(query).tolist()
    key = compute_query_hash(query, user_id)

    data = {
        "query": query,
        "embedding": emb,
        "user_id": user_id,
        "final_answer": answer,
        "timestamp": time.time(),
    }

    with shelve.open(SHELVE_PATH) as db:
        db[key] = data

    disk_cache.set(key, data, expire=CACHE_TTL)


# =========================
# MEMORY STORE
# =========================

class PersistentMemoryStore:
    def __init__(self, path=MEMORY_PATH):
        self.path = path

    def _key(self, ns, user_id):
        return f"{':'.join(ns)}|{user_id}"

    def get(self, ns, user_id):
        with shelve.open(self.path) as db:
            return db.get(self._key(ns, user_id), "")

    def put(self, ns, user_id, val):
        with shelve.open(self.path) as db:
            db[self._key(ns, user_id)] = val

    def clear_user(self, user_id):
        with shelve.open(self.path) as db:
            for k in list(db.keys()):
                if k.endswith(f"|{user_id}"):
                    del db[k]


# =========================
# LLM
# =========================

def call_llm(prompt: str):
    with trace(name="llm_call", project_name=PROJECT_NAME):
        resp = hf_client.chat_completion(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()


def web_search(query: str):
    with trace(name="web_search", project_name=PROJECT_NAME):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=2):
                results.append(f"{r.get('title')}\n{r.get('body')}\n{r.get('href')}")
        return "\n\n".join(results) if results else "No results found."


# =========================
# STATE
# =========================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    profile_memory: Optional[str]
    conversation_memory: Optional[str]
    observation: Optional[str]
    final_answer: Optional[str]
    profile_update: Optional[str]
    conversation_update: Optional[str]
    user_id: str
    cache_hit: Optional[bool]


# =========================
# NODES
# =========================

def memory_load_node(state: AgentState, store: PersistentMemoryStore):
    return {
        "profile_memory": store.get(PROFILE_NS, state["user_id"]),
        "conversation_memory": store.get(CONVO_NS, state["user_id"]),
        "cache_hit": False,
    }


def action_node(state: AgentState):
    user_query = state["messages"][-1].content
    cached = semantic_cache_lookup(user_query, state["user_id"])

    if cached:
        return {
            "observation": cached["final_answer"],
            "final_answer": cached["final_answer"],
            "cache_hit": True,
        }

    prompt = f"""
User memory:
{state.get("profile_memory")}

Conversation memory:
{state.get("conversation_memory")}

Question:
{user_query}

If unsure respond: INSUFFICIENT
"""
    llm_answer = call_llm(prompt)

    if "insufficient" in llm_answer.lower():
        observation = web_search(user_query)
    else:
        observation = llm_answer

    return {"observation": observation, "cache_hit": False}


def final_answer_node(state: AgentState):
    if state.get("cache_hit"):
        return {"final_answer": state["final_answer"]}

    user_query = state["messages"][-1].content

    answer_prompt = f"""
User memory:
{state.get("profile_memory")}

Conversation memory:
{state.get("conversation_memory")}

Info:
{state.get("observation")}

Answer:
"""
    answer = call_llm(answer_prompt)

    profile_update = call_llm(
        f"Extract only personal facts. If none return NONE.\n{user_query}"
    )

    convo_update = call_llm(
        f"Summarize interaction in one sentence.\nUser:{user_query}\nAssistant:{answer}"
    )

    semantic_cache_write(user_query, state["user_id"], answer)

    return {
        "final_answer": answer,
        "profile_update": profile_update,
        "conversation_update": convo_update,
    }


def memory_persist_node(state: AgentState, store: PersistentMemoryStore):
    if not state.get("cache_hit"):
        if state.get("profile_update") and state["profile_update"].lower() != "none":
            store.put(PROFILE_NS, state["user_id"], state["profile_update"])
        if state.get("conversation_update"):
            store.put(CONVO_NS, state["user_id"], state["conversation_update"])
    return {}


# =========================
# BUILD GRAPH
# =========================

def build_graph():
    store = PersistentMemoryStore()
    builder = StateGraph(AgentState)

    builder.add_node("memory_load", lambda s: memory_load_node(s, store))
    builder.add_node("action", action_node)
    builder.add_node("final_answer", final_answer_node)
    builder.add_node("memory_persist", lambda s: memory_persist_node(s, store))

    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "action")
    builder.add_edge("action", "final_answer")
    builder.add_edge("final_answer", "memory_persist")
    builder.add_edge("memory_persist", END)

    return builder.compile(checkpointer=MemorySaver()), store


# =========================
# STREAMING RUNNER
# =========================

def run_agent_stream(query: str, graph, store, user_id: str):
    state = {
        "messages": [HumanMessage(content=query)],
        "profile_memory": None,
        "conversation_memory": None,
        "observation": None,
        "final_answer": None,
        "profile_update": None,
        "conversation_update": None,
        "user_id": user_id,
        "cache_hit": False,
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("\n--- STREAM START ---")

    for event in graph.stream(state, config):
        for node, output in event.items():
            print(f"\n[{node.upper()} OUTPUT]")
            print(output)

    print("\n--- STREAM END ---")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    graph, store = build_graph()
    user_id = "user_1"

    print("STREAMING AGENT WITH MEMORY + CACHE")
    print("Type 'quit' to exit")

    while True:
        q = input("\nYour question: ").strip()
        if q == "quit":
            break
        run_agent_stream(q, graph, store, user_id)
