"""FastAPI entry point for the Research & Search Agent.

Exposes the LangGraph research pipeline over HTTP. Run with:
    uvicorn main:app --reload --port 8001
"""
import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.graph import run_research, run_research_with_persistence, resume_research
from src.exceptions import DeepResearchError, ResearchAgentError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research & Search Agent",
    description="Multi-source research assistant with claim-level groundedness verification",
    version="0.1.0",
)


# =============================================================================
# Request / Response models
# =============================================================================

class ResearchRequest(BaseModel):
    topic: str
    sources: Optional[List[str]] = None       # e.g. ["web"], later ["web", "local"]
    persist: bool = False                       # use SQLite checkpointing vs in-memory
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str


# =============================================================================
# Routes
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/research")
async def research(req: ResearchRequest):
    try:
        if req.persist:
            result = await run_research_with_persistence(req.topic, thread_id=req.thread_id)
        else:
            result = await run_research(
                topic=req.topic,
                sources_available=req.sources,
                use_checkpoints=True,
                thread_id=req.thread_id,
            )
    except ResearchAgentError as e:
        logger.error(f"Research error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal error during research")

    # result is a ResearchState-shaped dict/object depending on LangGraph version
    return {
        "final_report": result.get("final_report") if isinstance(result, dict) else result.final_report,
        "citations": result.get("citations") if isinstance(result, dict) else result.citations,
        "error": result.get("error") if isinstance(result, dict) else result.error,
    }


@app.post("/research/resume")
async def research_resume(req: ResumeRequest):
    try:
        result = await resume_research(req.thread_id)
    except ResearchAgentError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Resume failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during resume")

    return {
        "final_report": result.get("final_report") if isinstance(result, dict) else result.final_report,
        "citations": result.get("citations") if isinstance(result, dict) else result.citations,
    }