# 04_react_aot_loop.ipynb

# Purpose:
# - Implement an execution node that follows a ReAct / Atom-of-Thought (AoT) loop.
# - Execute each step in a plan produced by the planner.
# - Demonstrate calling tools via middleware (e.g., web search, memory).
# - Integrate with LangSmith tracing for observability.
import json
import os
import time
from typing import Any, Annotated,TypedDict,Dict,List
import uuid
from langgraph.graph import StateGraph, START ,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import  add_messages
from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient

from huggingface_hub import InferenceClient


hf_client = InferenceClient(
    token=os.getenv("HF_TOKEN")
)

langsmith_client = LangSmithClient()

PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

print("Hugging Face client initialized")
print(" LangSmith client initialized")
print("LangSmith Project:", PROJECT_NAME)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list | None
    step_index: int
    observations: list
    user_id: str
    last_answer: str | None

class AoTState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    plan: List[str]      # Steps from planner
    current_step: int
    observations: List[str]
    thoughts: List[str]
    final_answer: str | None

type(AgentState)


def call_llm(
    prompt: str,
    *,
    model: str = "zai-org/GLM-4.7-Flash",
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    metadata: Dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """
    Canonical LLM call wrapper using:
    - Hugging Face Inference chat_completion
    - LangSmith tracing
    - Chat-based models (GLM / OpenAI-style)
    """
    metadata = metadata or {}
    tags = tags or []

    with trace(
        name="hf_llm_call",
        project_name=PROJECT_NAME,
        metadata={
            **metadata,
            "model": model,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        tags=tags,
    ):
        start = time.time()

        response = hf_client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        latency = round(time.time() - start, 3)

        # Optional: attach latency as metadata
        trace_context = {
            "latency_sec": latency,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            
        }

        return response

def get_assistant_text(response) -> str :
    # if response.choices[0].message.reasoning_content:
    #     return response.choices[0].message.reasoning_content
    if response.choices[0].message.content:
        return response.choices[0].message.content
# Action Node (Too Executor Stub)

This simulates tools like search, calculator, API clients, etc..
def action_node(state: AoTState) -> Dict[str, Any]:
    """Execute actual tools based on the current step"""
    step = state["plan"][state["current_step"]]
    
    # Use LLM to actually gather information about the step
    prompt = f"""
You are a research assistant. Provide detailed information about: {step}

Be specific, factual, and comprehensive. Include key facts, dates, and achievements.
"""
    
    response = call_llm(
        prompt, 
        max_new_tokens=1024,  # More tokens for detailed info
        metadata={"node": "action", "step": step}, 
        tags=["tool_execution"]
    )
    
    observation = get_assistant_text(response).strip()
    
    return {
        "observations": state["observations"] + [observation]
    }

# def action_node(state: dict) -> dict:
#     step = state["plan"][state["current_step"]]
#     observation = middleware_action_for_step(step, top_k=3)
#     return {"observations": state["observations"] + [observation]}

# Thought Node (LLM Reasoning Based on Observation)
def thought_node(state: AoTState) -> Dict[str, Any]:
    """Reflect on the observation and plan next steps"""
    current_obs = state["observations"][-1] if state["observations"] else ""
    current_step_num = state["current_step"]
    total_steps = len(state["plan"])
    
    prompt = f"""
You are analyzing information gathering progress.

Current Step: {current_step_num + 1}/{total_steps}
Latest Information Gathered:
{current_obs}

Write a brief 1-2 sentence reflection on what was learned and what's needed next.
"""
    
    response = call_llm(
        prompt, 
        max_new_tokens=128,
        metadata={"node": "thought"}, 
        tags=["react_thought"]
    )
    
    thought_text = get_assistant_text(response).strip()
    return {"thoughts": state["thoughts"] + [thought_text]}

# Update Step Node

Increment step counter and decide to loop or finish.
def update_step_node(state: AoTState) -> Dict[str, Any]:
    return {"current_step": state["current_step"] + 1}

def router(state: AoTState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "action"
    return "end"

def final_answer_node(state: AoTState) -> Dict[str, Any]:
    """Generate comprehensive final answer"""
    
    # Combine all observations
    all_info = "\n\n".join([
        f"**Step {i+1}:** {state['plan'][i]}\n{obs}" 
        for i, obs in enumerate(state["observations"])
    ])
    
    prompt = f"""
Based on the following research, provide a comprehensive, well-structured summary:

{all_info}

Create a cohesive narrative that synthesizes all the information gathered.
"""
    
    response = call_llm(
        prompt, 
        max_new_tokens=1024,  # Allow longer final answer
        metadata={"node": "final_answer"}, 
        tags=["final_answer"]
    )
    
    return {"final_answer": get_assistant_text(response).strip()}

# Build the AoT Loop Graph

# Action → Observation → Thought → Decision → (loop or stop)

# https://docs.langchain.com/oss/python/langgraph/graph-api?utm_source=chatgpt.com
short_term_saver = InMemorySaver()
long_term_store = InMemoryStore()
builder = StateGraph(AoTState)
builder.add_node("action", action_node)
builder.add_node("thought", thought_node)
builder.add_node("update_step", update_step_node)
builder.add_node("final_answer", final_answer_node)
builder.add_edge(START, "action")
builder.add_edge("action", "thought")
builder.add_edge("thought", "update_step")
builder.add_edge("final_answer", END)
builder.add_conditional_edges(
    "update_step",
    router,
    {
        "action": "action",
        "end": "final_answer",  # Changed from END to "final_answer"
    },
)

# builder.add_conditional_edges(
#     "update_step",
#     router,
#     {
#         "action": "action",
#         "end": END,
#     },
# )

graph = builder.compile(
    checkpointer=short_term_saver,
    store=long_term_store,
)

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id
    }
}
initial_state: AoTState = {
    "messages": [],
    "user_id": "user1",
    "plan": [
        "Summary about Sam Altman",
        "His work at Loopt",
        "His work at Y Combinator",
        "His work at OpenAI",


  ],
    "current_step": 0,
    "observations": [],
    "thoughts": [],
    "final_answer": None,
}

result = graph.invoke(initial_state, config)

print("Observations:")
for o in result["observations"]:
    print("-", o)
result
print("\nThoughts:")
for t in result["thoughts"]:
    print("-", t)
# User Input
  
#       ↓

# Planner (03)
  
#       ↓

# AoT Loop Executor (04)
  
#       ↓

# Final Answer

# user_plan = input("Enter comma-separated steps: ")

# plan_list = [s.strip() for s in user_plan.split(",")]

# state = {
#     "messages": [],
#     "user_id": "user1",
#     "plan": plan_list,
#     "current_step": 0,
#     "observations": [],
#     "thoughts": [],
#     "final_answer": None,
# }

# result = graph.invoke(state, config)
# print("Observations:", result["observations"])
# print("Thoughts:", result["thoughts"])

# print("\nThoughts:")
# for t in result["thoughts"]:
#     print("-", t)
print("Final Answer:", result["final_answer"])