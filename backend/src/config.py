"""Configuration for the Research & Search Agent."""
from pydantic_settings import BaseSettings
from typing import Optional
import logging

COMPLEXITY_LIMITS = {
    "simple":   {"max_queries": 2, "max_results_per_query": 3},
    "moderate": {"max_queries": 3, "max_results_per_query": 5},
    "complex":  {"max_queries": 5, "max_results_per_query": 5},
}

class Settings(BaseSettings):
    # LLM
    model_provider: str = "ollama"          # ollama | openai | gemini | llamacpp | groq | nvidia
    model_name: str = "qwen3:8b"
    fast_model_name: str = "llama3.2:3b"    # cheap model for extraction/credibility-adjacent tasks
    summarization_model: str = "openai/gpt-oss-120b"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com"
    google_api_key: Optional[str] = None
    gemini_verifier_api_key: Optional[str] = None
    llamacpp_base_url: str = "http://localhost:8080"
    groq_api_key: Optional[str] = None
    groq_verifier_api_key: Optional[str] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    router_provider: str = "groq"
    router_model: str = "llama-3.1-8b-instant"
    router_api_key: Optional[str] = None

    planner_provider: str = "gemini"
    planner_model: str = "gemini-3.5-flash-lite"
    planner_api_key: Optional[str] = None

    claim_extraction_provider: str = "gemini"
    claim_extraction_model: str = "gemini-3.5-flash-lite"
    claim_extraction_api_key: Optional[str] = None

    verifier_provider: str = "gemini"
    verifier_model: str = "gemini-3.5-flash-lite"
    verifier_api_key: Optional[str] = None

    synthesizer_provider: str = "groq"
    synthesizer_model: str = "openai/gpt-oss-120b"
    synthesizer_api_key: Optional[str] = None

    summarizer_map_provider: str = "groq"
    summarizer_map_model: str = "openai/gpt-oss-20b"
    summarizer_map_api_key: Optional[str] = None

    summarizer_reduce_provider: str = "groq"
    summarizer_reduce_model: str = "openai/gpt-oss-120b"
    summarizer_reduce_api_key: Optional[str] = None

    fast_local_provider: str = "groq"
    fast_local_model: str = "openai/gpt-oss-20b"
    fast_local_api_key: Optional[str] = None
    # Legacy model settings retained for compatibility with external callers.
    claim_extraction_model_legacy: str = "qwen2.5:3b-instruct"
    gemini_planner_model: str = "gemini-3.5-flash-lite"
    gemini_claim_extraction_model: str = "gemini-3.5-flash-lite"
    gemini_verifier_model: str = "gemini-3.5-flash-lite"
    nvidia_api_key: Optional[str] = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_verifier_model: str = "openai/gpt-oss-20b"

    verifier_max_concurrent_calls: int = 3 

    # Search
    search_provider: str = "tavily"         # tavily | searxng
    tavily_api_key: Optional[str] = None
    max_search_queries: int = 5
    max_search_results_per_query: int = 5
    min_credibility_score: int = 40
    max_docs_for_extraction: int = 8

    chunk_relevance_keep_ratio: float = 0.70
    chunk_relevance_min_survivors: int = 4

    # Verification
    min_claims_verified_ratio: float = 0.6  # trigger retry loop if below this
    max_retries: int = 2

    # Claim extraction — safety ceiling on LLM parsing/latency, NOT a quality-selection cap.
    claim_extraction_safety_cap: int = 15

    # Relevance / selection — real claim selection happens here using the cross-encoder score across the FULL pool.
    relevance_keep_ratio: float = 0.60
    relevance_min_survivors: int = 5
    relevance_global_budget: int = 25       # total claims allowed through, across all sources combined
    relevance_min_per_source: int = 1       # guaranteed floor of claims per source
 
    # Report
    max_report_sections: int = 8
    citation_style: str = "apa"

    # Local RAG (Phase 2)
    local_rag_enabled: bool = False

    # Summarization
    summary_map_threshold_tokens: int = 6000   # docs under this go single-pass
    summary_map_chunk_tokens: int = 4000       # window size per map call for larger docs
    summary_max_concurrent_map_calls: int = 2  # tune for provider token limits

    # Port
    port: int = 8001

    class Config:
        env_file = ".env"


config = Settings()
logging.getLogger(__name__).info(
    f"[startup] planner_provider={config.planner_provider!r} planner_model={config.planner_model!r}"
)