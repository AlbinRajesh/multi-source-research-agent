import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from api.schemas import ResearchRequest
from src.graph import build_graph  # your compiled StateGraph

router = APIRouter()
graph = build_graph()  # compile once at import time

@router.post("/research/stream")
async def stream_research(payload: ResearchRequest):
    initial_state = {
        "query": payload.query,
        "sub_queries": [],
        "sources_to_use": ["web"],
        "raw_results": [],
        "fetched_docs": [],
        "deduped_docs": [],
        "claims": [],
        "verified_claims": [],
        "retry_count": 0,
        "final_answer": "",
        "citations": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    async def event_generator():
        try:
            async for update in graph.astream(initial_state, stream_mode="updates"):
                # `update` is like {"planner": {...state changes...}}
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
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())


def _summarize(node_name: str, output: dict) -> dict:
    """Trim node output to what the UI actually needs to display."""
    if node_name == "planner":
        return {"sub_queries": output.get("sub_queries", [])}
    if node_name == "synthesizer":
        return {
            "final_answer": output.get("final_answer", ""),
            "citations": output.get("citations", []),
        }
    if node_name == "verifier":
        return {"verified_count": len(output.get("verified_claims", []))}
    return {"keys": list(output.keys())}