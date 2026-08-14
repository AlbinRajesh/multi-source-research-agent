"""State management for the Research & Search Agent.

Adapted from a LangGraph reference research-agent pattern, extended with
Claim Extraction + Verification stages for claim-level groundedness checking
(not just source-level credibility scoring).
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
from src.utils.relevance import filter_by_relevance


# =============================================================================
# Planning
# =============================================================================

class SearchQuery(BaseModel):
    query: str = Field(description="The search query text")
    purpose: str = Field(description="Why this query is being made")
    source_hint: Literal["web", "local", "both"] = Field(
        default="web", description="Which source type this query should target"
    )
    completed: bool = Field(default=False)

class ResearchPlan(BaseModel):
    topic: str
    objectives: List[str]
    search_queries: List[SearchQuery]
    report_outline: List[str]
    complexity: Literal["simple", "moderate", "complex"] = "moderate"


# =============================================================================
# Search / Retrieval
# =============================================================================

class SearchResult(BaseModel):
    query: str
    title: str
    url: str
    snippet: str
    content: Optional[str] = Field(default=None, description="Full extracted content")
    source_type: Literal["web", "local"] = Field(default="web")
    source_name: Optional[str] = Field(default=None, description="Domain or filename")


# =============================================================================
# Claim Extraction + Verification  (our added stages)
# =============================================================================

class Claim(BaseModel):
    """An atomic, checkable factual statement extracted from a document."""
    id: str = Field(description="Unique claim id")
    text: str = Field(description="The atomic claim statement")
    source_url: str = Field(description="URL/id of the document this claim was extracted from")
    source_index: int = Field(description="Index into search_results this claim came from")


class VerificationVerdict(BaseModel):
    claim_id: str
    is_grounded: bool = Field(description="Whether the source text actually supports this claim")
    confidence: Literal["verified", "single_source", "unconfirmed", "conflicting"] = Field(
        description=(
            "verified: grounded + corroborated by 2+ independent sources. "
            "single_source: grounded but only one source supports it. "
            "unconfirmed: not clearly grounded in any retrieved source. "
            "conflicting: sources disagree on this claim."
        )
    )
    corroborating_source_indices: List[int] = Field(default_factory=list)
    reasoning: Optional[str] = Field(default=None, description="Brief justification, for audit trail")


# =============================================================================
# Report
# =============================================================================

class ReportSection(BaseModel):
    title: str
    content: str
    sources: List[str] = Field(default_factory=list)


# =============================================================================
# Overall workflow state
# =============================================================================

class ResearchState(BaseModel):
    # Input
    research_topic: str

    # Sources available this run (Phase 1: web only; Phase 2 adds "local")
    sources_available: List[Literal["web", "local"]] = Field(default_factory=lambda: ["web"])

    # Planning
    plan: Optional[ResearchPlan] = Field(default=None)

    # Search
    search_results: List[SearchResult] = Field(default_factory=list)
    credibility_scores: List[Dict] = Field(default_factory=list)

    # Added Workflow Tracking Fields
    processed_result_indices: List[int] = Field(default_factory=list)
    route_decision: Optional[Literal["search", "synthesize", "plan", "refine_search"]] = Field(default=None)

    # Claim extraction + verification (our added stages)
    claims: List[Claim] = Field(default_factory=list)
    verified_claims: List[VerificationVerdict] = Field(default_factory=list)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=2)
    retry_feedback: list[str] = []      
    confirmed_claims: list = []  
    weak_claims_to_resolve: list = []

    # Synthesis / report
    key_findings: List[str] = Field(default_factory=list)
    report_sections: List[ReportSection] = Field(default_factory=list)
    final_report: Optional[str] = Field(default=None)
    citations: List[Dict] = Field(default_factory=list)

    #Tracker
    token_tracker: Any = None

    # Workflow control
    current_stage: Literal[
        "planning", "searching", "extracting_claims", "verifying",
        "synthesizing", "reporting", "complete"
    ] = Field(default="planning")
    error: Optional[str] = Field(default=None)
    iterations: int = Field(default=0)

    # LLM tracking
    stage_timings: Dict[str, float] = Field(default_factory=dict)
    llm_calls: int = Field(default=0)
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)
    llm_call_details: List[Dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True