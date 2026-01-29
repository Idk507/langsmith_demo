# 06_kv_caching.ipynb

# Purpose:
# - Implement a robust KV caching layer for LLM and web middleware outputs.
# - Provide both in-memory TTL cache and optional persistent disk cache (shelve or diskcache).
# - Cache normalized, serializable outputs only (assistant_text + usage + metadata).
# - Version cache keys by model, prompt version, and other parameters.
# - Provide decorators and helper functions to integrate caching with call_llm and middleware.
# - Log hits/misses to LangSmith for observability.
# - Include smoke tests.

import os
import time
import json
import hashlib
import pickle
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import shelve
import diskcache 
from diskcache import  Cache as DiskCache
from langsmith import Client as LangSmithClient 
from langsmith.run_helpers import trace

#  Design notes (runtime)
"""
Design decisions (runtime):
- Primary in-memory TTL cache for fastest lookups in current process.
- Optional durable cache using `shelve` (stdlib) or `diskcache` if installed.
- Cache ONLY normalized, serializable dicts:
    {
       "assistant_text": str,
       "usage": dict or None,
       "model": str,
       "created_at": iso8601 str,
       "metadata": {...}
    }
  Avoid caching raw HF response objects to prevent non-serializable issues.
- Cache keys: sha256(versioned_prefix + JSON(sorted_inputs))
  where versioned_prefix includes model name and cache version tags.
- Provide decorator `@kv_cache` to wrap callables that produce serializable results.
- Trace hits/misses in LangSmith (if available).
"""

LANGSMITH_AVAILABLE = False
try:
    from langsmith import Client as LangSmithClient
    from langsmith.run_helpers import trace
    LANGSMITH_AVAILABLE = True
except ImportError:
    pass
#  In-memory TTL cache implementation
@dataclass
class InMemoryTTL:
    ttl_seconds: int = 3600
    _store: Dict[str, Tuple[float, Any]] = field(default_factory=dict)

    def get(self, key: str) -> Optional[Any]:
        rec = self._store.get(key)
        if not rec:
            return None
        ts, val = rec
        if time.time() - ts > self.ttl_seconds:
            # expired
            del self._store[key]
            return None
        return val

    def set(self, key: str, value: Any):
        self._store[key] = (time.time(), value)

    def delete(self, key: str):
        if key in self._store:
            del self._store[key]

    def clear(self):
        self._store.clear()

    def stats(self):
        return {"entries": len(self._store)}

# instantiate a module-level cache (adjust TTL via env)
DEFAULT_TTL = int(os.getenv("KV_CACHE_TTL", "3600"))
_inmem_cache = InMemoryTTL(ttl_seconds=DEFAULT_TTL)
print("In-memory TTL cache initialized with TTL seconds:", DEFAULT_TTL)

# Optional persistent cache (shelve) wrapper (safe serializable store)
# We use a simple file-backed shelf if available. It stores pickled values under keys.
SHELVE_PATH = os.getenv("KV_SHELVE_PATH", "kv_cache_shelf.db")

class ShelveCache:
    def __init__(self, path=SHELVE_PATH):
        self.path = path
        # shelve opens files lazily as needed

    def get(self, key: str) -> Optional[Any]:
        try:
            with shelve.open(self.path) as db:
                return db.get(key)
        except Exception:
            return None

    def set(self, key: str, value: Any):
        try:
            with shelve.open(self.path) as db:
                db[key] = value
        except Exception:
            pass

    def delete(self, key: str):
        try:
            with shelve.open(self.path) as db:
                if key in db:
                    del db[key]
        except Exception:
            pass

    def clear(self):
        try:
            with shelve.open(self.path) as db:
                for k in list(db.keys()):
                    del db[k]
        except Exception:
            pass

_shelve_cache = ShelveCache()
print("Shelve cache wrapper ready (path):", SHELVE_PATH )

#  Optional diskcache integration (higher performance, if installed)
_diskcache = None
try:
    _diskcache = DiskCache(os.getenv("KV_DISKCACHE_DIR", "kv_diskcache"))
    print("DiskCache initialized at", _diskcache.directory)
except Exception as e:
    _diskcache = None
    print("DiskCache init failed:", e)

#  Key generation utilities (versioned, deterministic)
def _stable_serialize(obj: Any) -> str:
    # Ensure deterministic JSON: sort keys, handle bytes
    return json.dumps(obj, default=lambda o: repr(o), sort_keys=True, separators=(",", ":"))

def build_kv_key(prefix: str, model: str, input_payload: Any, version: str = "v1") -> str:
    """
    Build a stable key for cache lookup.
    prefix: e.g., "llm" or "web"
    model: model identifier (include model version)
    input_payload: prompt text or dict (will be serialized deterministically)
    version: cache version token (bump on prompt template changes)
    """
    payload_serial = _stable_serialize(input_payload)
    raw = f"{prefix}|{model}|{version}|{payload_serial}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"kv:{prefix}:{model}:{version}:{h}"

# Normalizer for LLM results (serializable and minimal)
def normalize_llm_output(hf_response) -> Dict[str, Any]:
    """
    Given a HuggingFace chat_completion response object,
    extract assistant_text and usage info into a serializable dict.
    Assumes response.choices[0].message.content exists.
    """
    try:
        assistant_text = None
        usage = None
        # Some HF response types are objects; guard with attr access
        if hasattr(hf_response, "choices") and len(hf_response.choices) > 0:
            msg = hf_response.choices[0].message
            # prefer content over reasoning content for observations
            assistant_text = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None)
        else:
            # If it's already a dict-like
            if isinstance(hf_response, dict):
                assistant_text = hf_response.get("assistant_text") or hf_response.get("content")
                usage = hf_response.get("usage")
        # usage may be an object — convert to dict if possible
        if hasattr(hf_response, "usage"):
            try:
                usage = vars(hf_response.usage)
            except Exception:
                # fallback: try to access fields
                usage = {
                    "prompt_tokens": getattr(hf_response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(hf_response.usage, "completion_tokens", None),
                    "total_tokens": getattr(hf_response.usage, "total_tokens", None),
                }
        # Build result
        res = {
            "assistant_text": assistant_text or "",
            "usage": usage,
            "model": getattr(hf_response, "model", None) or "unknown",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        return res
    except Exception as e:
        # fallback minimal result
        return {"assistant_text": str(hf_response), "usage": None, "model": "unknown", "created_at": datetime.utcnow().isoformat() + "Z"}

# KV caching decorator and direct helpers
# This decorator supports 3-layer cache: in-memory TTL, diskcache (optional), shelve (optional)
def kv_cache(prefix: str, model: str, version: str = "v1", ttl_seconds: Optional[int] = None):
    """
    Decorator to cache function outputs based on arguments.
    The wrapped function must return a serializable dict (not raw HF objects).
    Example usage:
        @kv_cache("llm", model="mymodel", version="v1")
        def call_and_return_serializable(prompt): ...
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Build stable payload from args & kwargs
            payload = {"args": args, "kwargs": kwargs}
            key = build_kv_key(prefix, model, payload, version=version)
            # Check in-memory
            cached = _inmem_cache.get(key)
            if cached is not None:
                # trace hit
                if LANGSMITH_AVAILABLE:
                    with trace(name="kv_cache_hit", project_name=os.getenv("LANGCHAIN_PROJECT"), metadata={"key": key, "prefix": prefix, "model": model}, tags=["kv_cache"]):
                        pass
                return cached
            # diskcache
            if _diskcache is not None:
                try:
                    v = _diskcache.get(key, default=None)
                    if v is not None:
                        _inmem_cache.set(key, v)
                        if LANGSMITH_AVAILABLE:
                            with trace(name="kv_cache_hit_diskcache", project_name=os.getenv("LANGCHAIN_PROJECT"), metadata={"key": key}, tags=["kv_cache"]):
                                pass
                        return v
                except Exception:
                    pass
            # shelve
            
            v = _shelve_cache.get(key)
            if v is not None:
                _inmem_cache.set(key, v)
                if LANGSMITH_AVAILABLE:
                    with trace(name="kv_cache_hit_shelve", project_name=os.getenv("LANGCHAIN_PROJECT"), metadata={"key": key}, tags=["kv_cache"]):
                        pass
                return v
            # Miss -> call function
            result = func(*args, **kwargs)
            # Ensure serializable (we expect dict)
            try:
                json.dumps(result, default=lambda o: repr(o))
            except Exception:
                # try to coerce if huggingface response
                if hasattr(result, "choices"):
                    result = normalize_llm_output(result)
                else:
                    # fallback to stringify
                    result = {"value": str(result)}
            # Write to caches
            _inmem_cache.set(key, result)
            if _diskcache is not None:
                try:
                    _diskcache.set(key, result, expire=ttl_seconds or DEFAULT_TTL)
                except Exception:
                    pass
            
            _shelve_cache.set(key, result)
            # trace miss
            if LANGSMITH_AVAILABLE:
                with trace(name="kv_cache_miss", project_name=os.getenv("LANGCHAIN_PROJECT"), metadata={"key": key, "prefix": prefix, "model": model}, tags=["kv_cache"]):
                    pass
            return result
        return wrapper
    return decorator

# Direct helpers (without decorator) for manual get/set
def kv_get(prefix: str, model: str, payload: Any, version: str = "v1"):
    key = build_kv_key(prefix, model, payload, version=version)
    v = _inmem_cache.get(key)
    if v is not None:
        return v
    if _diskcache is not None:
        try:
            v = _diskcache.get(key, default=None)
            if v is not None:
                _inmem_cache.set(key, v)
                return v
        except Exception:
            pass
    
    v = _shelve_cache.get(key)
    if v is not None:
        _inmem_cache.set(key, v)
        return v
    return None

def kv_set(prefix: str, model: str, payload: Any, value: Any, version: str = "v1", ttl: Optional[int] = None):
    key = build_kv_key(prefix, model, payload, version=version)
    _inmem_cache.set(key, value)
    if _diskcache is not None:
        try:
            _diskcache.set(key, value, expire=ttl or DEFAULT_TTL)
        except Exception:
            pass
    
    _shelve_cache.set(key, value)
    if LANGSMITH_AVAILABLE:
        with trace(name="kv_cache_set", project_name=os.getenv("LANGCHAIN_PROJECT"), metadata={"key": key}, tags=["kv_cache"]):
            pass

#  LLM wrapper integration example (safe caching)
# This assumes you have an hf_client.chat_completion wrapper elsewhere.
# We'll construct a cached wrapper that normalizes HF output before storing.

# Example: safe function to call HF and return serializable dict
def _call_llm_and_normalize(prompt: str, model: str = "zai-org/GLM-4.7-Flash", **kwargs):
    """
    Calls hf_client.chat_completion and returns normalized output dict.
    This function returns a dict that is safe to cache.
    """
    # Ensure hf_client exists (user may have created it in prior kernel)
    from huggingface_hub import InferenceClient
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set in environment for LLM calls")
    client = InferenceClient(token=hf_token)
    resp = client.chat_completion(model=model, messages=[{"role":"user","content":prompt}], **kwargs)
    # Normalize into serializable dict
    norm = normalize_llm_output(resp)
    return norm

# Decorated cached version
@kv_cache(prefix="llm", model="zai-org/GLM-4.7-Flash", version="v1")
def cached_llm_call(prompt: str, model: str = "zai-org/GLM-4.7-Flash", **kwargs):
    return _call_llm_and_normalize(prompt, model=model, **kwargs)

#  Middleware integration example (web_search caching)
# Suppose your web_search(query) returns a list of dicts (serializable).
# Wrap that with kv_cache as well (prefix "web", model denotes provider string).

def _raw_web_search_ddg(query: str, num_results: int = 3):
    # Minimal implementation using DDGS; assumes DDGS imported elsewhere
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append({"title": r.get("title"), "body": r.get("body"), "href": r.get("href")})
    return results

@kv_cache(prefix="web", model="duckduckgo", version="v1")
def cached_web_search(query: str, num_results: int = 3):
    return _raw_web_search_ddg(query, num_results=num_results)

#  Quick smoke tests (run these to validate)
# 1) Test in-memory cache behavior for arbitrary payload
payload = {"q":"What is the capital of France?"}
k = build_kv_key("test", "mymodel", payload, version="v1")
print("Test key:", k)
print("Inmem stats before:", _inmem_cache.stats())

_inmem_cache.set(k, {"value":"paris", "ts":datetime.utcnow().isoformat()})
print("Get immediately:", _inmem_cache.get(k))

# 2) Test decorated cached_web_search (will use DDGS; ensure DDGS available)
try:
    res = cached_web_search("capital of France", num_results=2)
    print("cached_web_search result (first call):", res)
    # call again to assert cache hit (short TTL)
    res2 = cached_web_search("capital of France", num_results=2)
    print("cached_web_search result (second call, should be cached):", res2)
except Exception as e:
    print("cached_web_search error (DDGS may not be installed):", e)

# 3) Test cached_llm_call (requires HF_TOKEN)
try:
    sample_prompt = "Say hi and confirm cache test."
    llm_res1 = cached_llm_call(sample_prompt, model="zai-org/GLM-4.7-Flash")
    print("LLM normalized result (first call):", llm_res1.get("assistant_text", "")[:200])
    # call again — should be cache hit and return same dict quickly
    llm_res2 = cached_llm_call(sample_prompt, model="zai-org/GLM-4.7-Flash")
    print("LLM result from cache (second call):", llm_res2.get("assistant_text", "")[:200])
except Exception as e:
    print("cached_llm_call error (HF token/model access may be required):", e)
