"""Citation formatting — plain code, no LLM call needed."""
from typing import List, Dict, Optional, Set


def format_citations(
    usable_verdicts,
    claim_by_id,
    citation_index_map,
    search_results,
    style: str = "apa",
    cited_indices: Optional[Set[int]] = None,
) -> List[Dict]:
    """Build the reference list for the final answer, deduplicated by URL
    so unique sources map cleanly to [n] markers.

    cited_indices: if provided, only sources whose index actually appears
    in the final answer text are included. Without this filter, every
    source backing a usable claim gets listed even if the LLM's prose
    never referenced it — which is misleading in the UI (shows sources
    the user never actually saw cited).
    """
    seen_urls = {}
    citations = []

    for v in usable_verdicts:
        claim = claim_by_id.get(v.claim_id)
        if not claim:
            continue
        doc = search_results[claim.source_index] if claim.source_index < len(search_results) else None
        if not doc:
            continue

        idx = citation_index_map.get(doc.url)
        if not idx or any(c["index"] == idx for c in citations):
            continue

        if cited_indices is not None and idx not in cited_indices:
            continue

        entry = {
            "index": idx,
            "confidence": v.confidence,
            "url": doc.url,
            "title": doc.title,
            "source_type": doc.source_type,
        }
        if doc.source_type == "local":
            entry["citation_text"] = f"[{idx}] Your document: {doc.source_name or doc.title}"
        else:
            entry["citation_text"] = f"[{idx}] {doc.title}. {doc.url}"

        citations.append(entry)

    citations.sort(key=lambda x: x["index"])
    return citations