import logging
import json
import uuid
from typing import Optional, List, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langgraph.checkpoint.memory import MemorySaver

from src.graph import run_research, run_research_with_persistence, resume_research, create_research_graph
from src.state import ResearchState
from src.exceptions import DeepResearchError, ResearchAgentError
from src.config import config

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

# Persistent checkpointer and compiled graph instance across requests
_checkpointer = MemorySaver()
_graph = create_research_graph(checkpointer=_checkpointer)


# 3. Request/response models
class ResearchRequest(BaseModel):
    topic: str
    sources: Optional[List[str]] = None
    persist: bool = False
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str


# 4. All routes
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/research")
async def research(req: ResearchRequest):
    ...


@app.post("/research/resume")
async def research_resume(req: ResumeRequest):
    ...


@app.post("/research/stream")
async def research_stream(req: ResearchRequest):
    tid = req.thread_id or f"research-{uuid.uuid4().hex[:8]}"
    run_config = {"configurable": {"thread_id": tid}}

    prior_history = []
    existing = await _graph.aget_state(run_config)
    if existing and existing.values:
        prior_history = existing.values.get("conversation_history", [])

    initial_state = ResearchState(
        research_topic=req.topic,
        sources_available=req.sources or ["web"],
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
        return {"node": node_name, "result_count": len(output.get("search_results", []))}
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