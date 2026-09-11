import logging
import json
import uuid
from typing import Optional, List, AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langgraph.checkpoint.memory import MemorySaver

from src.graph import create_sqlite_checkpointer, run_research, run_research_with_persistence, resume_research, create_research_graph
from src.state import ResearchState
from src.exceptions import DeepResearchError, ResearchAgentError
from src.config import config
from src.api.upload import router as upload_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Create the app FIRST
app = FastAPI(
    title="Research & Search Agent",
    description="Multi-source research assistant with claim-level groundedness verification",
    version="0.1.0",
)

# 2. THEN add middleware (app must already exist)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount routers
app.include_router(upload_router)

# Persistent checkpointer and compiled graph instance across requests
_checkpointer = MemorySaver()
_graph = create_research_graph(checkpointer=_checkpointer)


# 4. Request/response models
class ResearchRequest(BaseModel):
    topic: str
    sources: Optional[List[str]] = None
    selected_doc_ids: Optional[List[str]] = None
    source_mode: Optional[Literal["web", "local", "hybrid"]] = None
    persist: bool = False
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str


def _resolve_source_mode(req: ResearchRequest) -> Literal["web", "local", "hybrid"]:
    if req.source_mode:
        return req.source_mode
    requested_sources = set(req.sources or ["web"])
    if requested_sources == {"local"}:
        return "local"
    if {"web", "local"}.issubset(requested_sources):
        return "hybrid"
    return "web"


@app.on_event("startup")
async def preload_models():
    from src.rag.embeddings import warmup as warmup_embeddings
    from src.rag.reranker import warmup as warmup_reranker
    from src.utils.relevance import warmup as warmup_relevance

    logger.info("Preloading models at startup...")
    warmup_embeddings()
    warmup_reranker()
    warmup_relevance()
    logger.info("Model preload complete.")
    
# 5. All routes
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/research")
async def research(req: ResearchRequest):
    tid = req.thread_id or f"research-{uuid.uuid4().hex[:8]}"
    run_config = {"configurable": {"thread_id": tid}}

    source_mode = _resolve_source_mode(req)
    sources = ["web", "local"] if source_mode == "hybrid" else [source_mode]

    try:
        if req.persist:
            async with create_sqlite_checkpointer() as checkpointer:
                graph = create_research_graph(checkpointer=checkpointer)

                prior_history = []
                existing = await graph.aget_state(run_config)
                if existing and existing.values:
                    prior_history = existing.values.get("conversation_history", [])

                initial_state = ResearchState(
                    research_topic=req.topic,
                    sources_available=sources,
                    source_mode=source_mode,
                    selected_doc_ids=req.selected_doc_ids or [],
                    conversation_history=prior_history + [{"role": "user", "content": req.topic}],
                )
                final_state = await graph.ainvoke(initial_state, config=run_config)
        else:
            prior_history = []
            existing = await _graph.aget_state(run_config)
            if existing and existing.values:
                prior_history = existing.values.get("conversation_history", [])

            initial_state = ResearchState(
                research_topic=req.topic,
                sources_available=sources,
                source_mode=source_mode,
                selected_doc_ids=req.selected_doc_ids or [],
                conversation_history=prior_history + [{"role": "user", "content": req.topic}],
            )
            final_state = await _graph.ainvoke(initial_state, config=run_config)

        return {"thread_id": tid, "result": final_state}
    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/resume")
async def research_resume(req: ResumeRequest):
    try:
        final_state = await resume_research(req.thread_id)
        return {"thread_id": req.thread_id, "result": final_state}
    except Exception as e:
        logger.error(f"Research resume failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/stream")
async def research_stream(req: ResearchRequest):
    tid = req.thread_id or f"research-{uuid.uuid4().hex[:8]}"
    run_config = {"configurable": {"thread_id": tid}}

    prior_history = []
    existing = await _graph.aget_state(run_config)
    if existing and existing.values:
        prior_history = existing.values.get("conversation_history", [])

    source_mode = _resolve_source_mode(req)
    sources = ["web", "local"] if source_mode == "hybrid" else [source_mode]

    initial_state = ResearchState(
        research_topic=req.topic,
        sources_available=sources,
        source_mode=source_mode,
        selected_doc_ids=req.selected_doc_ids or [],
        conversation_history=prior_history + [{"role": "user", "content": req.topic}],
    )
    graph = _graph

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            async for update in graph.astream(initial_state, config=run_config, stream_mode="updates"):
                node_name = list(update.keys())[0]
                yield {
                    "event": "node_update",
                    "data": json.dumps(_summarize(node_name, update[node_name]), default=str),
                }
            yield {"event": "done", "data": json.dumps({"status": "complete", "thread_id": tid})}
        except Exception as e:
            logger.error(f"Streaming research failed: {e}")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())

def _summarize(node_name: str, output: dict) -> dict:
    if node_name == "route":
        if output.get("is_casual"):
            return {
                "node": "route",
                "is_casual": True,
                "final_report": output.get("final_report", ""),
                "citations": [],
            }
        return {"node": "route", "is_casual": False}
    if node_name == "plan":
        plan = output.get("plan")
        return {
            "node": node_name,
            "objectives": plan.objectives if plan else [],
            "search_queries": [q.query for q in plan.search_queries] if plan else [],
        }
    if node_name == "search":
        if output.get("error"):
            return {"node": node_name, "error": output["error"]}
        results = output.get("search_results", [])
        return {
            "node": node_name,
            "result_count": len(results),
            "local_count": sum(1 for r in results if getattr(r, "source_type", None) == "local"),
        }
    if node_name == "extract_claims":
        return {"node": node_name, "claim_count": len(output.get("claims", []))}
    if node_name == "verify":
        verified = output.get("verified_claims", [])
        return {
            "node": node_name,
            "verified_count": sum(1 for v in verified if v.confidence in ("verified", "single_source")),
            "total_count": len(verified),
        }
    if node_name == "check_retry":
        return {"node": node_name, "route": output.get("route_decision")}
    if node_name == "fast_local_answer":
        return {
            "node": node_name,
            "final_report": output.get("final_report", ""),
            "citations": output.get("citations", []),
            "route_decision": output.get("route_decision"),
        }
    if node_name == "synthesize":
        if "error" in output:
            return {"node": node_name, "error": output["error"]}
        return {
            "node": node_name,
            "final_report": output.get("final_report", ""),
            "citations": output.get("citations", []),
        }
    return {"node": node_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.port, reload=True)