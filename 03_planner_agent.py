# 03_planner_agent.ipynb

# Purpose:
# - Implement the Planner node
# - Integrate it into the existing LangGraph workflow
# - Test planning logic using a real LLM
# - Store and inspect the generated plan


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
    user_id: str
    last_answer: str | None



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
    if response.choices[0].message.reasoning_content:
        return response.choices[0].message.reasoning_content
    if response.choices[0].message.content:
        return response.choices[0].message.content
# planner agent 
def planner_node(state: AgentState) -> dict[str, Any]:
    """
    Planner node:
    - Reads the last user message
    - Uses LLM to generate a structured plan (JSON list of steps)
    - Writes plan into state
    """

    messages = state.get("messages", [])
    if not messages:
        return {"plan": []}

    last_message = messages[-1]

    # LangGraph converts dicts → HumanMessage
    if isinstance(last_message, dict):
        user_query = last_message["content"]
    else:
        user_query = last_message.content

    planner_prompt = f"""
You are a Planner Agent in an agentic AI system.

Your job is to break the user query into clear, minimal steps
that another agent can execute.

User Query:
{user_query}

Return ONLY a JSON array of steps.
Each step should be a short string.
Do NOT add explanations.
Do NOT add markdown.
"""

    response = call_llm(
        planner_prompt,
        metadata={"node": "planner"},
        tags=["planner"]
    )

    text = get_assistant_text(response)

    try:
        plan = json.loads(text)
        if not isinstance(plan, list):
            raise ValueError("Plan is not a list")
    except Exception:
        plan = [text.strip()]

    return {"plan": plan}

# build graph with planner
short_term_saver = InMemorySaver()
long_term_store = InMemoryStore()

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)

builder.add_edge(START, "planner")

graph = builder.compile(
    checkpointer=short_term_saver,
    store=long_term_store
)

print(" Planner graph compiled")

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id
    }
}

initial_state: AgentState = {
    "messages": [{"role": "user", "content": "Explain how to build a RAG system"}],
    "plan": None,
    "user_id": "user_1"
}

result = graph.invoke(initial_state, config)

print("Generated plan:")
print(result["plan"])
