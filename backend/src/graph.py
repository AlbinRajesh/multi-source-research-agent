"""LangGraph workflow for the Research & Search Agent.

Adapted from a reference deep-research-agent pattern (StateGraph +
SqliteSaver checkpointing + conditional routing), extended with:
  - claim extraction + verification stages (the accuracy differentiator)
  - relevance filtering to strip off-topic extracted claims
  - a conditional retry loop: if too many claims are unconfirmed, loop
    back to refining search queries instead of blindly re-searching,
    and only weak claims are re-verified on subsequent passes
"""
import uuid
import logging
from metrics.token_counter import TokenTracker
from src.processing.chunk_relevance import filter_chunks_by_relevance
from src.agents.fast_local_agent import FastLocalAgent
from src.rag.vector_store import has_any_documents
from src.agents.router import RouterAgent
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from src.config import config, COMPLEXITY_LIMITS
import time
from functools import wraps

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.state import ResearchState, SearchQuery    
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.claim_extractor import ClaimExtractionAgent
from src.agents.verifier import VerificationAgent
from src.agents.synthesizer import SynthesizerAgent
from src.exceptions import ResearchAgentError
from src.utils.relevance import filter_by_relevance
from src.config import config

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
    router = RouterAgent()
    planner = PlannerAgent()
    retriever = RetrieverAgent()
    claim_extractor = ClaimExtractionAgent()
    verifier = VerificationAgent()
    synthesizer = SynthesizerAgent()
    fast_local_agent = FastLocalAgent()

    workflow = StateGraph(ResearchState)

    workflow.add_node("route", timed_node("route")(router.route))
    workflow.add_node("plan", timed_node("plan")(planner.plan))
    workflow.add_node("fast_local_answer", timed_node("fast_local_answer")(fast_local_agent.answer))
    workflow.add_node("search", timed_node("search")(retriever.search))
    workflow.add_node("chunk_relevance_filter", timed_node("chunk_relevance_filter")(chunk_relevance_filter))
    workflow.add_node("extract_claims", timed_node("extract_claims")(claim_extractor.extract))
    workflow.add_node("relevance_filter", timed_node("relevance_filter")(relevance_filter))
    workflow.add_node("verify", timed_node("verify")(verify_only_weak(verifier.verify)))
    workflow.add_node("check_retry", check_retry)
    workflow.add_node("refine_search", timed_node("refine_search")(refine_search_queries))
    workflow.add_node("synthesize", timed_node("synthesize")(synthesizer.synthesize))

    workflow.add_edge(START, "route")

    def after_route(state: ResearchState) -> str:
        return END if state.is_casual else "plan"

    def after_plan(state: ResearchState) -> str:
        if state.error or not state.plan or not state.plan.search_queries:
            logger.error(f"Planning invalid: {state.error}")
            return END
        if getattr(state.plan, "mode", None) == "fast_local" and state.plan.complexity == "simple":
            return "fast_local_answer"
        return "search"

    def after_search(state: ResearchState) -> str:
        if state.error or not state.search_results:
            logger.error(f"Search invalid: {state.error}")
            return END
        return "chunk_relevance_filter"

    workflow.add_conditional_edges("route", after_route, {"plan": "plan", END: END})
    workflow.add_conditional_edges("plan", after_plan, {"search": "search", "fast_local_answer": "fast_local_answer", END: END})
    workflow.add_conditional_edges(
        "fast_local_answer",
        lambda s: s.route_decision,
        {"done": END, "escalate": "search"}
    )
    workflow.add_conditional_edges("search", after_search, {"chunk_relevance_filter": "chunk_relevance_filter", END: END})
    workflow.add_edge("chunk_relevance_filter", "extract_claims")
    workflow.add_conditional_edges("extract_claims", after_extract_claims, {"relevance_filter": "relevance_filter", "synthesize": "synthesize"})
    workflow.add_conditional_edges("relevance_filter", after_relevance_filter, {"verify": "verify", "synthesize": "synthesize"})

    workflow.add_edge("verify", "check_retry")
    workflow.add_conditional_edges(
        "check_retry", lambda s: s.route_decision,
        {"refine_search": "refine_search", "synthesize": "synthesize"}
    )
    workflow.add_conditional_edges(
        "refine_search",
        lambda s: s.route_decision,
        {"search": "search", "synthesize": "synthesize"}
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile(checkpointer=checkpointer)

# =============================================================================
# Relevance Filter Node & Routing Edges
# =============================================================================


async def chunk_relevance_filter(state: ResearchState) -> dict:
    if not state.search_results:
        return {}
    kept, scores = filter_chunks_by_relevance(
        state.search_results,
        state.research_topic,
        keep_ratio=config.chunk_relevance_keep_ratio,
        min_survivors=config.chunk_relevance_min_survivors,
        global_budget=config.max_docs_for_extraction,
    )
    if scores:
        logger.info(
            f"[chunk_relevance_dist] min={min(scores):.2f} max={max(scores):.2f} "
            f"avg={sum(scores)/len(scores):.2f} n={len(scores)}"
        )
    logger.info(f"Chunk relevance filter: {len(state.search_results)} -> {len(kept)} results")
    return {"search_results": kept}


async def relevance_filter(state: ResearchState) -> dict:
    if not state.claims:
        return {}

    kept, scores = filter_by_relevance(
        state.claims,
        state.research_topic,
        keep_ratio=config.relevance_keep_ratio,
        min_survivors=config.relevance_min_survivors,
        global_budget=config.relevance_global_budget,
        min_per_source=config.relevance_min_per_source,
    )

    if scores:
        logger.info(
            f"[relevance_dist] min={min(scores):.2f} max={max(scores):.2f} "
            f"avg={sum(scores)/len(scores):.2f} n={len(scores)}"
        )
    logger.info(f"Relevance filter: {len(state.claims)} -> {len(kept)} claims")

    return {"claims": kept}


def after_extract_claims(state: ResearchState) -> str:
    if state.error or not state.claims:
        logger.warning("No claims extracted — nothing to verify, going straight to synthesize")
        return "synthesize"
    return "relevance_filter"


def after_relevance_filter(state: ResearchState) -> str:
    if not state.claims:
        logger.warning("No claims survived relevance filter — going straight to synthesize")
        return "synthesize"
    return "verify"


# =============================================================================
# Targeted re-verification wrapper
# =============================================================================

def verify_only_weak(verify_fn):
    """
    On the first pass, verify everything. On retries, only the newly
    extracted claims need to go through verify.verify() — claims already
    confirmed on a prior pass are banked in state.confirmed_claims and
    skipped, instead of being re-sent to the verifier every loop.
    """
    @wraps(verify_fn)
    async def wrapper(state: ResearchState):
        result = await verify_fn(state)
        if not isinstance(result, dict) or "error" in result:
            return result

        newly_verified = result.get("verified_claims", [])
        merged = list(state.confirmed_claims) + list(newly_verified)
        result["verified_claims"] = merged
        return result
    return wrapper


# =============================================================================
# Retry Logic Node
# =============================================================================

def check_retry(state: ResearchState) -> dict:
    if state.error:
        return {"route_decision": "synthesize"}

    total = len(state.verified_claims) or 1
    weak_claims = [
        v for v in state.verified_claims
        if v.confidence in ("unconfirmed", "conflicting")
    ]
    weak_ratio = len(weak_claims) / total
    threshold = 1 - config.min_claims_verified_ratio

    if weak_ratio > threshold and state.retry_count < state.max_retries:
        confirmed = [
            v for v in state.verified_claims
            if v.confidence not in ("unconfirmed", "conflicting")
        ]

        claim_by_id = {c.id: c for c in state.claims}
        weak_claim_texts = [
            claim_by_id[v.claim_id].text
            for v in weak_claims
            if v.claim_id in claim_by_id
        ]

        logger.info(
            f"{weak_ratio:.0%} weak ({len(weak_claims)}/{total}), "
            f"retry {state.retry_count + 1}/{state.max_retries} — "
            f"banking {len(confirmed)} confirmed claims, "
            f"targeting {len(weak_claim_texts)} weak claims only"
        )
        return {
            "retry_count": state.retry_count + 1,
            "route_decision": "refine_search",
            "confirmed_claims": confirmed,
            "verified_claims": confirmed,
            "claims": [],
            "weak_claims_to_resolve": weak_claim_texts,
        }

    return {"route_decision": "synthesize"}


async def refine_search_queries(state: ResearchState) -> dict:
    weak_texts = state.weak_claims_to_resolve or []
    already_tried = set(state.tried_queries or [])

    seen = set()
    deduped = []
    for q in weak_texts:
        q = q[:200]
        if q not in seen and q not in already_tried:
            seen.add(q)
            deduped.append(q)

    if not deduped:
        logger.warning("No resolvable queries from weak claims, skipping refine_search")
        return {"route_decision": "synthesize"}

    complexity = getattr(state.plan, "complexity", "moderate")
    tier = COMPLEXITY_LIMITS.get(complexity, COMPLEXITY_LIMITS["moderate"])
    query_cap = tier["max_queries"]

    if len(deduped) > query_cap:
        logger.info(
            f"[retry_cap] {len(deduped)} weak claims -> capping retry queries to "
            f"{query_cap} (complexity={complexity})"
        )
        deduped = deduped[:query_cap]

    logger.info(f"Refined retry queries ({len(deduped)}): {deduped}")

    updated_plan = state.plan.model_copy(update={
        "search_queries": [
            SearchQuery(
                query=q,
                purpose="Retry: resolve unconfirmed claim",
                source_hint="both" if "local" in state.sources_available else "web",
            )
            for q in deduped
        ]
    })
    return {
        "plan": updated_plan,
        "route_decision": "search",
        "tried_queries": list(already_tried | seen),
    }


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

    checkpointer = None
    run_config: Dict[str, Any] = {}
    if use_checkpoints:
        checkpointer = MemorySaver()
        tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
        run_config["configurable"] = {"thread_id": tid}
        logger.info(f"thread_id: {tid}")

    graph = create_research_graph(checkpointer=checkpointer)

    # Resume prior turns if this thread already has state
    prior_history = []
    if use_checkpoints and thread_id:
        existing = await graph.aget_state(run_config)
        if existing and existing.values:
            prior_history = existing.values.get("conversation_history", [])

    sources = sources_available or ["web"]
    if "local" not in sources and has_any_documents():
        sources = sources + ["local"]

    initial_state = ResearchState(
        research_topic=topic,
        sources_available=sources,
        token_tracker=TokenTracker(),
        conversation_history=prior_history + [{"role": "user", "content": topic}],
    )

    try:
        final_state = await graph.ainvoke(initial_state, config=run_config or None)

        tracker = final_state.get("token_tracker") if isinstance(final_state, dict) else None
        if tracker:
            token_summary = tracker.summary()
            logger.info(
                f"[tokens] run total: {token_summary['total_tokens']} "
                f"({token_summary['total_input_tokens']} in / {token_summary['total_output_tokens']} out)"
            )
            if isinstance(final_state, dict):
                final_state["token_summary"] = token_summary

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

async def run_research_with_persistence(topic: str, sources_available: Optional[list] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """SQLite-persisted version — survives process restarts."""
    sources = sources_available or ["web"]
    if "local" not in sources and has_any_documents():
        sources = sources + ["local"]

    initial_state = ResearchState(
        research_topic=topic,
        sources_available=sources,
        token_tracker=TokenTracker(),
    )
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