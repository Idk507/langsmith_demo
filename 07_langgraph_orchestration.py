import os
import uuid
import json
import re
from typing import TypedDict, Annotated, List, Dict, Any,Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient

from huggingface_hub import InferenceClient
from ddgs import DDGS
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
langsmith_client = LangSmithClient()
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

print("LangSmith project:", PROJECT_NAME)
# ----------------------------
# LLM Wrapper (optimized)
# ----------------------------
def call_llm(prompt: str, *, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=1024) -> str:
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

# ----------------------------
# Web Search (unchanged)
# ----------------------------
def web_search(query: str, k: int = 2) -> str:
    with trace(
        name="web_search",
        project_name=PROJECT_NAME,
        metadata={"query": query},
        tags=["tool", "web"],
    ):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                title = r.get("title","")
                snippet = r.get("body","")
                link = r.get("href","")
                results.append(f"{title}\n{snippet}\nSource: {link}")
        return "\n\n".join(results) if results else "No results found."

# ----------------------------
# JSON extractor (fast)
# ----------------------------
def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    try:
        return json.loads(text)
    except Exception:
        return [text]

# =========================
# STATE
# =========================

class OrchestrationState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: Optional[List[str]]
    current_step: int
    observations: List[str]
    thought: Optional[str]
    final_answer: Optional[str]
    user_id: str


# =========================
# PLANNER
# =========================

def planner_node(state: OrchestrationState) -> Dict[str, Any]:
    last = state["messages"][-1]
    user_query = last["content"] if isinstance(last, dict) else last.content

    prompt = f"""
Break the question into at most 2 research steps.

Question:
{user_query}

Return ONLY a JSON list.
"""

    raw = call_llm(prompt)
    plan = extract_json(raw)

    if not isinstance(plan, list):
        plan = [user_query]

    plan = plan[:2]

    return {"plan": plan, "current_step": 0, "observations": []}

# =========================
# ACTION
# =========================

def action_node(state: OrchestrationState) -> Dict[str, Any]:
    step = state["plan"][state["current_step"]]

    prompt = f"""
For this step:
{step}

Return ONLY JSON:
{{"search_query": "..."}}
"""

    raw = call_llm(prompt)
    decision = extract_json(raw)

    search_query = decision.get("search_query", step)

    observation = web_search(search_query)

    if observation == "NO_RESULTS":
        fallback_prompt = f"Answer using general knowledge: {step}"
        observation = call_llm(fallback_prompt, max_new_tokens=300)

    return {"observations": state["observations"] + [observation]}



# =========================
# THOUGHT
# =========================

def thought_node(state: OrchestrationState) -> Dict[str, Any]:
    step = state["plan"][state["current_step"]]
    observation = state["observations"][-1]

    prompt = f"""
You are a Thought Agent.

Step:
{step}

Observation:
{observation[:400]}

Decide:
Is this sufficient to answer the step?

Answer ONLY:
"Sufficient" or "Insufficient" with one short reason.
"""

    thought = call_llm(prompt, max_new_tokens=80)
    print("[THOUGHT]:", thought)

    return {"thought": thought}



# =========================
# STEP UPDATE
# =========================

def update_step_node(state: OrchestrationState) -> Dict[str, Any]:
    return {"current_step": state["current_step"] + 1}

# =========================
# ROUTER
# =========================

def router(state: OrchestrationState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "action"
    return "final_answer"
# =========================
# FINAL ANSWER
# =========================

def final_answer_node(state: OrchestrationState) -> Dict[str, Any]:
    last = state["messages"][-1]
    user_query = last["content"] if isinstance(last, dict) else last.content

    all_info = "\n\n".join(state["observations"])
    thought = state.get("thought", "")

    prompt = f"""
Answer the question strictly using the information.

Question:
{user_query}

Information:
{all_info}

Thought:
{thought}

Rules:
- Do not invent facts.
- If information is insufficient, say so clearly.
- Be concise and accurate.
"""

    answer = call_llm(prompt)

    return {"final_answer": answer}

# =========================
# BUILD GRAPH
# =========================

def build_graph():
    builder = StateGraph(OrchestrationState)

    builder.add_node("planner", planner_node)
    builder.add_node("action", action_node)
    builder.add_node("thought", thought_node)
    builder.add_node("update_step", update_step_node)
    builder.add_node("final_answer", final_answer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "action")
    builder.add_edge("action", "thought")
    builder.add_edge("thought", "update_step")

    builder.add_conditional_edges(
        "update_step",
        router,
        {
            "action": "action",
            "final_answer": "final_answer",
        },
    )

    builder.add_edge("final_answer", END)

    return builder.compile(
        checkpointer=MemorySaver(),
        store=InMemoryStore(),
    )



graph = build_graph()

# ----------------------------
# Run
# ----------------------------
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

user_query = input("Ask a question: ")

initial_state: OrchestrationState = {
    "messages": [{"role": "user", "content": user_query}],
    "plan": None,
    "current_step": 0,
    "observations": [],
    "thought": None,
    "final_answer": None,
    "user_id": "user_1",
}

result = graph.invoke(initial_state, config)

print(result["final_answer"])
