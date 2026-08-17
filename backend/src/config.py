"""Configuration for the Research & Search Agent."""
from pydantic_settings import BaseSettings
from typing import Optional

COMPLEXITY_LIMITS = {
    "simple":   {"max_queries": 2, "max_results_per_query": 3},
    "moderate": {"max_queries": 3, "max_results_per_query": 5},
    "complex":  {"max_queries": 5, "max_results_per_query": 5},
}

class Settings(BaseSettings):
    # LLM
    model_provider: str = "ollama"          # ollama | openai | gemini | llamacpp | groq
    model_name: str = "qwen3:8b"
    fast_model_name: str = "llama3.2:3b"    # cheap model for extraction/credibility-adjacent tasks
    summarization_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com"
    google_api_key: Optional[str] = None
    llamacpp_base_url: str = "http://localhost:8080"
    groq_api_key: Optional[str] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    claim_extraction_model: str = "qwen2.5:3b-instruct"

    # Search
    search_provider: str = "tavily"         # tavily | searxng
    tavily_api_key: Optional[str] = None
    max_search_queries: int = 5
    max_search_results_per_query: int = 5
    min_credibility_score: int = 40
    max_docs_for_extraction: int = 8

    # Verification
    min_claims_verified_ratio: float = 0.6  # trigger retry loop if below this
    max_retries: int = 2


    relevance_keep_ratio: float = 0.60   # keep top 50% of each batch by relevance rank
    relevance_min_survivors: int = 5   # safety floor — never drop below this many claims
 
    # Report
    max_report_sections: int = 8
    citation_style: str = "apa"

    # Local RAG (Phase 2)
    local_rag_enabled: bool = False

    #Port
    port: int = 8001

    class Config:
        env_file = ".env"


config = Settings()