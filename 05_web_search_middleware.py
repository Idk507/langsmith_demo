# 05_web_search_middleware.ipynb

# Purpose:
# - Implement and validate web search middleware for agent tools
# - DuckDuckGo Search
# - Sanitize and normalize results
# - Use an in-memory TTL cache
# - Trace calls with LangSmith
# - Provide example integration with action_node from AoT loop

import os
import time
import html
import requests
from typing import List, Dict, Any, Optional, Tuple,TypedDict,Annotated
from dataclasses import dataclass, field
import uuid
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

print("DuckDuckGo middleware initialized")



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
#In-Memory TTL Cache
@dataclass
class TTLCache:
    ttl_seconds: int = 1800
    _store: Dict[str, Tuple[float, Any]] = field(default_factory=dict)

    def get(self, key: str) -> Optional[Any]:
        record = self._store.get(key)
        if not record:
            return None
        ts, value = record
        if time.time() - ts > self.ttl_seconds:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any):
        self._store[key] = (time.time(), value)

    def clear(self):
        self._store.clear()


web_cache = TTLCache()
print("TTL cache initialized")

#  Helpers: sanitize & normalize text using BeautifulSoup or fallback
def sanitize_html_to_text(html_content: str, max_chars: int = 1000) -> str:
    """
    Convert HTML to clean text. If BeautifulSoup available use it,
    otherwise fall back to naive html.unescape and truncate.
    """
    if not html_content:
        return ""
    if BS4_AVAILABLE:
        soup = BeautifulSoup(html_content, "html.parser")
        # remove script/style
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = soup.get_text(separator="\n", strip=True)
    else:
        text = html.unescape(html_content)
    # collapse whitespace
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text

def truncate(text: str, n: int = 800) -> str:
    if text is None:
        return ""
    t = text.strip()
    return t if len(t) <= n else t[: n - 3].rstrip() + "..."

def duckduckgo_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Uses DuckDuckGo via DDGS (official API).
    Returns normalized search results.
    """
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            title = r.get("title", "")
            snippet = r.get("body", "")
            link = r.get("href", "")

            results.append({
                "title": truncate(title, 200),
                "snippet": truncate(snippet, 800),
                "link": link,
                "source": "duckduckgo",
                "raw": r
            })

    return results

query = "What is the capital of France?"
result = DDGS().text(query, max_results=5)
for r in result:
    print(r)

# Cache Key Builder
def build_cache_key(query: str, num_results: int) -> str:
    return f"web:duckduckgo:{num_results}:{abs(hash(query))}"

#Unified Middleware Function (with LangSmith tracing)
def web_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    cache_key = build_cache_key(query, num_results)
    cached = web_cache.get(cache_key)
    if cached:
        return cached

    with trace(
        name="web_search",
        project_name=PROJECT_NAME,
        metadata={"query": query},
        tags=["duckduckgo", "web_search"],
    ):
        results = DDGS().text(query, max_results=num_results)

    web_cache.set(cache_key, results)
    return results

def fetch_and_extract(url: str, max_chars: int = 1200) -> str:
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "agent-middleware/1.0"})
        if resp.status_code != 200:
            return ""
        return sanitize_html_to_text(resp.text, max_chars=max_chars)
    except Exception:
        return ""

#Middleware Tool for AoT action_node

def middleware_action_for_step(step_text: str, top_k: int = 3) -> str:
    """
    Given a plan step, perform web search and return a formatted observation.
    """
    results = web_search(step_text, num_results=top_k)

    parts = []
    for i, r in enumerate(results):
        part = f"{i+1}. {r['title']}\n{r['snippet']}\nSource: {r['link']}"
        parts.append(part)

    observation_text = "\n\n".join(parts)
    return truncate(observation_text, 2000)

query = input("Enter a search query: ")
results = web_search(query, num_results=3)

for i, r in enumerate(results, 1):
    print(f"\n[{i}] {r['title']}")
    print(r["body"])
    print("URL:", r["href"])
