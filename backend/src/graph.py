"""LangGraph workflow for the Research & Search Agent.

Adapted from a reference deep-research-agent pattern (StateGraph +
SqliteSaver checkpointing + conditional routing), extended with:
  - claim extraction + verification stages (the accuracy differentiator)
  - a conditional retry loop: if too many claims are unconfirmed, loop
    back to search with a refined plan instead of synthesizing on weak
    evidence
"""
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from src.config import config
import time
from functools import wraps

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.state import ResearchState
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.claim_extractor import ClaimExtractionAgent
from src.agents.verifier import VerificationAgent
from src.agents.synthesizer import SynthesizerAgent
from src.exceptions import ResearchAgentError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Checkpointing & Instrumentation
# =============================================================================

def timed_node(stage_name: str):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(state: ResearchState):
            start = time.perf_counter()
            result = await fn(state)
            elapsed = time.perf_counter() - start
            if isinstance(result, dict) and "error" not in result:
                existing = dict(state.stage_timings)
                existing[stage_name] = existing.get(stage_name, 0.0) + elapsed
                result["stage_timings"] = existing
            logger.info(f"[timing] {stage_name}: {elapsed:.2f}s")
            return result
        return wrapper
    return decorator

def get_checkpoint_path() -> Path:
    cache_dir = Path(".cache/checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "research_checkpoints.db"


@asynccontextmanager
async def create_sqlite_checkpointer():
    async with AsyncSqliteSaver.from_conn_string(str(get_checkpoint_path())) as checkpointer:
        yield checkpointer


# =============================================================================
# Graph construction
# =============================================================================

def create_research_graph(checkpointer=None):
    planner = PlannerAgent()
    retriever = RetrieverAgent()
    claim_extractor = ClaimExtractionAgent()
    verifier = VerificationAgent()
    synthesizer = SynthesizerAgent()

    workflow = StateGraph(ResearchState)

    workflow.add_node("plan", timed_node("plan")(planner.plan))
    workflow.add_node("search", timed_node("search")(retriever.search))
    workflow.add_node("extract_claims", timed_node("extract_claims")(claim_extractor.extract))
    workflow.add_node("verify", timed_node("verify")(verifier.verify))
    workflow.add_node("check_retry", check_retry)
    workflow.add_node("synthesize", timed_node("synthesize")(synthesizer.synthesize))

    workflow.add_edge(START, "plan")

    def after_plan(state: ResearchState) -> str:
        if state.error or not state.plan or not state.plan.search_queries:
            logger.error(f"Planning invalid: {state.error}")
            return END
        return "search"

    def after_search(state: ResearchState) -> str:
        if state.error or not state.search_results:
            logger.error(f"Search invalid: {state.error}")
            return END
        return "extract_claims"

    def after_extract_claims(state: ResearchState) -> str:
        if state.error or not state.claims:
            logger.warning("No claims extracted — nothing to verify, going straight to synthesize")
            return "synthesize"
        return "verify"

    workflow.add_conditional_edges("plan", after_plan, {"search": "search", END: END})
    workflow.add_conditional_edges("search", after_search, {"extract_claims": "extract_claims", END: END})
    workflow.add_conditional_edges("extract_claims", after_extract_claims, {"verify": "verify", "synthesize": "synthesize"})
    
    workflow.add_edge("verify", "check_retry")
    workflow.add_conditional_edges(
        "check_retry", lambda s: s.route_decision, {"search": "search", "synthesize": "synthesize"}
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# Retry Logic Node
# =============================================================================

def check_retry(state: ResearchState) -> dict:
    total = len(state.verified_claims) or 1
    weak = sum(1 for v in state.verified_claims if v.confidence in ("unconfirmed", "conflicting"))
    weak_ratio = weak / total
    threshold = 1 - config.min_claims_verified_ratio

    if state.error:
        return {"route_decision": "synthesize"}
    if weak_ratio > threshold and state.retry_count < state.max_retries:
        logger.info(f"{weak_ratio:.0%} weak, retry {state.retry_count + 1}/{state.max_retries}")
        return {"retry_count": state.retry_count + 1, "route_decision": "search"}
    return {"route_decision": "synthesize"}


# =============================================================================
# Execution entry points
# =============================================================================

async def run_research(
    topic: str,
    sources_available: Optional[list] = None,
    use_checkpoints: bool = True,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info(f"Starting research on: {topic}")

    initial_state = ResearchState(
        research_topic=topic,
        sources_available=sources_available or ["web"],
    )

    run_config: Dict[str, Any] = {}
    checkpointer = None
    if use_checkpoints:
        checkpointer = MemorySaver()
        tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
        run_config["configurable"] = {"thread_id": tid}
        logger.info(f"thread_id: {tid}")

    graph = create_research_graph(checkpointer=checkpointer)

    try:
        final_state = await graph.ainvoke(initial_state, config=run_config or None)
        
        timings = final_state.get("stage_timings", {}) if isinstance(final_state, dict) else {}
        if timings:
            total = sum(timings.values())
            logger.info("=== Stage timing breakdown ===")
            for stage, t in sorted(timings.items(), key=lambda x: -x[1]):
                logger.info(f"  {stage}: {t:.2f}s ({t/total:.0%})")
            logger.info(f"  TOTAL: {total:.2f}s")

    except Exception as e:
        logger.error(f"Research workflow failed: {e}")
        raise

    return final_state


async def run_research_with_persistence(topic: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """SQLite-persisted version — survives process restarts."""
    initial_state = ResearchState(research_topic=topic)
    tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
    run_config = {"configurable": {"thread_id": tid}}

    async with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)
        try:
            return await graph.ainvoke(initial_state, config=run_config)
        except Exception as e:
            logger.error(f"Failed. Resume with thread_id: {tid}")
            raise


async def resume_research(thread_id: str) -> Dict[str, Any]:
    run_config = {"configurable": {"thread_id": thread_id}}
    async with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)
        state = await graph.aget_state(run_config)
        if not state or not state.values:
            raise ResearchAgentError(f"No checkpoint found for thread_id: {thread_id}")
        return await graph.ainvoke(None, config=run_config)