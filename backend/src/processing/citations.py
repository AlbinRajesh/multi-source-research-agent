"""Citation formatting — plain code, no LLM call needed."""
from typing import List, Dict


def format_citations(usable_verdicts, claim_by_id, citation_index_map, search_results, style: str = "apa") -> List[Dict]:
    """Build the reference list for the final answer, deduplicated by URL
    so unique sources map cleanly to [n] markers."""
    seen_urls = {}
    citations = []
    
    for v in usable_verdicts:
        claim = claim_by_id.get(v.claim_id)
        if not claim:
            continue
        doc = search_results[claim.source_index] if claim.source_index < len(search_results) else None
        if not doc:
            continue
            
        # Get the assigned index (now mapped by URL in synthesizer)
        idx = citation_index_map.get(doc.url)
        if not idx or any(c["index"] == idx for c in citations):
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

    # Sort citations by index number for clean chronological output
    citations.sort(key=lambda x: x["index"])
    return citations