import json
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    ResearchRequest, DocumentInfo, UploadResponse, DeleteResponse,
)
from src.graph import create_research_graph
from src.state import ResearchState
from src.config import config
from src.rag.chunker import chunk_text
from src.rag.vector_store import (
    add_chunks, delete_doc, list_all_documents, VectorStoreError,
)
from src.processing.file_extract import extract_text  # ADJUST if real module name differs
from metrics.token_counter import TokenTracker

logger = logging.getLogger(__name__)

router = APIRouter()
graph = create_research_graph()  # compiled once at import time, no checkpointer for streaming runs


# =============================================================================
# Research streaming
# =============================================================================

@router.post("/research/stream")
async def stream_research(payload: ResearchRequest):
    sources = ["web"]
    if config.local_rag_enabled and payload.doc_ids:
        sources.append("local")

    initial_state = ResearchState(
        research_topic=payload.query,
        sources_available=sources,
        selected_doc_ids=payload.doc_ids or [],
        token_tracker=TokenTracker(),
        conversation_history=[{"role": "user", "content": payload.query}],
    )

    async def event_generator():
        try:
            async for update in graph.astream(initial_state, stream_mode="updates"):
                node_name = list(update.keys())[0]
                node_output = update[node_name]
                yield {
                    "event": "node_update",
                    "data": json.dumps({
                        "node": node_name,
                        "output": _summarize(node_name, node_output),
                    }),
                }
            yield {"event": "done", "data": json.dumps({"status": "complete"})}
        except Exception as e:
            logger.error(f"Research stream failed: {e}")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())


def _summarize(node_name: str, output: dict) -> dict:
    if node_name == "plan":
        return {"sub_queries": [q.query for q in getattr(output.get("plan"), "search_queries", [])]}
    if node_name == "synthesize":
        return {
            "final_answer": output.get("final_report", ""),
            "citations": output.get("citations", []),
        }
    if node_name == "verify":
        return {"verified_count": len(output.get("verified_claims", []))}
    return {"keys": list(output.keys())}


# =============================================================================
# Document upload / list / delete
# =============================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        text = extract_text(raw, file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text: {e}")

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in file.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no chunks.")

    doc_id = str(uuid.uuid4())
    try:
        count = add_chunks(
            [c.text for c in chunks],
            doc_id=doc_id,
            source_filename=file.filename,
        )
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadResponse(
        doc_id=doc_id, filename=file.filename, chunk_count=count, status="indexed"
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    return [DocumentInfo(**d) for d in list_all_documents()]


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    deleted = delete_doc(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found.")
    return DeleteResponse(doc_id=doc_id, deleted=True)


@router.get("/upload/status/{doc_id}")
async def upload_status(doc_id: str):
    # Upload is synchronous above (no background job) — status is always
    # "done" if this is reachable. Kept only for frontend compatibility.
    return {"doc_id": doc_id, "status": "done"}