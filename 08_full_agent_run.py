# 08_full_agent_run
# This notebook runs the full agent:
# User Input → Planner → ReAct / AoT → Final Answer  
# with LangGraph orchestration, web search middleware (DDGS), and LangSmith tracing.

# Imports & Setup

import os
import uuid
import json
import re
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient

from huggingface_hub import InferenceClient
from ddgs import DDGS  # DuckDuckGo search provider

# Validate environment
assert os.getenv("HF_TOKEN"), "HF_TOKEN must be set in your environment"
assert os.getenv("LANGCHAIN_PROJECT"), "LANGCHAIN_PROJECT must be set"

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
langsmith_client = LangSmithClient()
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

print("LangSmith project:", PROJECT_NAME)

#   2 — LLM Wrapper

def call_llm(prompt: str, *, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=512) -> str:
    """
    Standardized LLM call wrapper with LangSmith tracing paired with Hugging Face Inference.
    """
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
        # Extract the content safely
        return resp.choices[0].message.content.strip()

#   3 — Web Search Middleware (DDGS)

def web_search(query: str, k: int = 2) -> str:
    """
    Searches via DDGS and returns normalized results.
    """
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
                    title = r.get("title", "")
                    snippet = r.get("body", "")
                    link = r.get("href", "")
                    results.append(f"{title}\n{snippet}\nSource: {link}")
        except Exception as e:
            # If search fails completely, return an empty string
            return f"Search failed: {str(e)}"
        return "\n\n".join(results) if results else "No results found."

#   4 — JSON Extractor (safe)

def extract_json(text: str) -> Any:
    """
    Extract JSON from possibly markdown–wrapped responses.
    """
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        # Remove leading fences
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("[") or candidate.startswith("{"):
                text = candidate
                break
    try:
        return json.loads(text)
    except Exception:
        # Fallback to searching for JSON patterns
        json_pattern = r'(\[.*?\]|\{.*?\})'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except Exception:
                continue
    # Return raw text in a list if all else fails
    return [text]

#   5 — Agent State Definition

class OrchestrationState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: Optional[List[str]]
    current_step: int
    observations: List[str]
    thought: Optional[str]
    final_answer: Optional[str]
    user_id: str

#   6 — Planner Agent

def planner_node(state: OrchestrationState) -> Dict[str, Any]:
    """
    Planner splits the user query into at most 2 actionable research steps.
    """
    last = state["messages"][-1]
    user_query = last["content"] if isinstance(last, dict) else last.content

    prompt = f"""
You are a Planner Agent.

Break this question into at most 2 clear research steps
and return ONLY a JSON array.

Question:
{user_query}
"""

    raw = call_llm(prompt, max_new_tokens=128)
    plan = extract_json(raw)
    if not isinstance(plan, list):
        plan = [user_query]
    plan = plan[:2]  # Cap plan length
    return {"plan": plan, "current_step": 0, "observations": []}

#   7 — Action Agent

def action_node(state: OrchestrationState) -> Dict[str, Any]:
    """
    Action agent decides what action to take for the current step
    and executes the web search if needed.
    """
    step = state["plan"][state["current_step"]]

    action_prompt = f"""
You are an Action Agent.

For this step:
{step}

Return ONLY JSON:
{{"search_query": "..."}}
"""

    raw = call_llm(action_prompt, max_new_tokens=64)
    decision = extract_json(raw)

    if isinstance(decision, dict):
        search_query = decision.get("search_query", step)
    else:
        search_query = step

    observation = web_search(search_query, k=2)
    return {"observations": state["observations"] + [observation]}

#   8 — Thought Agent

def thought_node(state: OrchestrationState) -> Dict[str, Any]:
    """
    Thought agent reflects on the latest observation and
    provides a short assessment of sufficiency.
    """
    step = state["plan"][state["current_step"]]
    observation = state["observations"][-1]

    prompt = f"""
You are a Thought Agent.

Step executed:
{step}

Observation:
{observation[:400]}

Decide if the step was answered sufficiently:
Respond ONLY with "Sufficient" or "Insufficient" and a short reason.
"""

    thought = call_llm(prompt, max_new_tokens=80)
    return {"thought": thought}

#   9 — Step Update Node

def update_step_node(state: OrchestrationState) -> Dict[str, Any]:
    """
    Move to the next plan step.
    """
    return {"current_step": state["current_step"] + 1}

#   10 — Router Function

def router(state: OrchestrationState) -> str:
    """
    Decide the next node based on the current progress.
    """
    if state["current_step"] < len(state["plan"]):
        return "action"
    return "final_answer"

#   11 — Final Answer Agent

def final_answer_node(state: OrchestrationState) -> Dict[str, Any]:
    """
    Synthesize all observations into a final answer for the user.
    """
    last = state["messages"][-1]
    user_query = last["content"] if isinstance(last, dict) else last.content
    all_info = "\n\n".join(state["observations"])
    thought = state.get("thought", "")

    prompt = f"""
You are a Synthesis Agent.

Using ONLY the information below, answer the user's question:

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

Final answer:
"""
    answer = call_llm(prompt, max_new_tokens=512)
    return {"final_answer": answer}

#   12 — Build LangGraph Orchestration

def build_full_graph():
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

graph = build_full_graph()

#   13 — Execute Full Agent Run

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

user_query = input("Ask your question: ")

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

print("\n=== FINAL ANSWER ===\n")
print(result["final_answer"])
