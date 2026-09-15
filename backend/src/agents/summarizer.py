"""Summarizer Agent — produces a summary of one or more selected local
documents using map-reduce over ALL chunks of the document (not
query-based top-k retrieval), bypassing claim-extraction/verification
entirely — summarization doesn't need per-claim groundedness checks
against web sources.
"""
import asyncio
import logging
import re
from typing import Dict, Any, List

import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.state import ResearchState
from src.rag.vector_store import get_all_chunks
from src.utils.llm_factory import get_llm
from src.config import config
from src.prompts.summary_prompt import MAP_PROMPT, REDUCE_PROMPT, SINGLE_PASS_PROMPT

logger = logging.getLogger(__name__)

_encoding = tiktoken.get_encoding("cl100k_base")

_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _parse_length_constraint(query: str) -> str:
    """Deterministic regex match — same pattern as the planner's
    format-constraint detection. An LLM asked to 'guess' the requested
    length is exactly the kind of instruction that silently drifts, so
    this is matched directly instead of relying on prompt-following."""
    q = query.strip().lower()

    m = re.search(r"\bin\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?\b", q)
    if m:
        n = m.group(1)
        n = _WORD_NUMS.get(n, n)
        return f"Exactly {n} sentence{'s' if str(n) != '1' else ''}. No more, no less."

    m = re.search(r"\bin\s+(\d+)\s+words?\b", q)
    if m:
        return f"Approximately {m.group(1)} words."

    m = re.search(r"\bin\s+(\d+|one|two|three|four|five)\s+(bullet\s*points?|points?)\b", q)
    if m:
        n = m.group(1)
        n = _WORD_NUMS.get(n, n)
        return f"Exactly {n} bullet points."

    if re.search(r"\b(one[- ]liner|tl;?dr|very short|super short)\b", q):
        return "1-2 sentences maximum."

    if re.search(r"\b(brief|short|concise|quick)\b", q):
        return "A brief summary, 3-5 sentences."

    if re.search(r"\b(detailed|long|comprehensive|in-depth|thorough)\b", q):
        return "A detailed summary, 4-6 paragraphs, covering all major points."

    return "A clear, well-organized summary, 150-250 words."


def _count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _chunk_for_map(full_text: str, chunk_tokens: int) -> List[str]:
    """Splits full document text into large, non-overlapping windows for
    the map step. No overlap needed (unlike retrieval chunking) — a hard
    cut here doesn't lose meaning the way similarity search would, and
    the reduce step re-synthesizes across the cut anyway."""
    tokens = _encoding.encode(full_text)
    windows = []
    for start in range(0, len(tokens), chunk_tokens):
        window_tokens = tokens[start:start + chunk_tokens]
        windows.append(_encoding.decode(window_tokens))
    return windows


class SummarizerAgent:
    MAX_CONCURRENT_MAP_CALLS = 4

    def __init__(self, llm=None):
        self.llm = llm or get_llm(
            temperature=0.3,
            model_override=config.summarization_model,
        )

    async def _call_llm(self, prompt_template: str, **kwargs) -> str:
        prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke(kwargs)

    async def _summarize_single_doc(self, doc_id: str, length_instruction: str) -> Dict[str, Any]:
        try:
            chunks = await asyncio.to_thread(get_all_chunks, doc_id)
        except Exception as e:
            logger.error(f"[summarize] failed to load doc_id={doc_id}: {e}")
            return {"doc_id": doc_id, "error": f"Could not load document: {e}"}

        if not chunks:
            logger.warning(f"[summarize] doc_id={doc_id} has no indexed chunks")
            return {"doc_id": doc_id, "error": "No content found for this document."}

        source_name = chunks[0].metadata.get("source", "uploaded document")
        full_text = "\n\n".join(c.text for c in chunks)
        total_tokens = _count_tokens(full_text)

        if total_tokens <= config.summary_map_threshold_tokens:
            try:
                summary = await self._call_llm(
                    SINGLE_PASS_PROMPT,
                    document=full_text,
                    length_instruction=length_instruction,
                )
            except Exception as e:
                logger.error(f"[summarize] single-pass failed for doc_id={doc_id}: {e}")
                return {"doc_id": doc_id, "error": f"Summarization failed: {e}"}
            return {"doc_id": doc_id, "source_name": source_name, "summary": summary.strip()}

        # Document too large for one pass — map-reduce.
        windows = _chunk_for_map(full_text, config.summary_map_chunk_tokens)
        logger.info(
            f"[summarize] doc_id={doc_id} ({total_tokens} tokens) -> "
            f"{len(windows)} map window(s)"
        )

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_MAP_CALLS)

        async def map_one(window: str, idx: int) -> str:
            async with semaphore:
                try:
                    return await self._call_llm(MAP_PROMPT, section=window)
                except Exception as e:
                    logger.warning(f"[summarize] map call {idx} failed for doc_id={doc_id}: {e}")
                    return ""  # dropped section degrades quality, doesn't abort the whole summary

        partial_summaries = await asyncio.gather(*[
            map_one(w, i) for i, w in enumerate(windows)
        ])
        partial_summaries = [s for s in partial_summaries if s.strip()]

        if not partial_summaries:
            return {"doc_id": doc_id, "error": "Summarization failed for all sections of this document."}

        combined = "\n\n".join(f"[Section {i+1}] {s}" for i, s in enumerate(partial_summaries))
        try:
            final_summary = await self._call_llm(
                REDUCE_PROMPT,
                partial_summaries=combined,
                length_instruction=length_instruction,
            )
        except Exception as e:
            logger.error(f"[summarize] reduce step failed for doc_id={doc_id}: {e}")
            return {"doc_id": doc_id, "error": f"Failed to combine section summaries: {e}"}

        return {"doc_id": doc_id, "source_name": source_name, "summary": final_summary.strip()}

    async def summarize(self, state: ResearchState) -> Dict[str, Any]:
        doc_ids = state.selected_doc_ids
        if not doc_ids:
            return {
                "error": "No document selected to summarize.",
                "final_report": "Please select or upload a document first, then ask me to summarize it.",
                "citations": [],
                "current_stage": "complete",
            }

        length_instruction = _parse_length_constraint(state.research_topic)

        results = await asyncio.gather(*[
            self._summarize_single_doc(doc_id, length_instruction) for doc_id in doc_ids
        ])

        succeeded = [r for r in results if "summary" in r]
        failed = [r for r in results if "error" in r]

        if not succeeded:
            error_detail = "; ".join(f"{r['doc_id']}: {r['error']}" for r in failed)
            return {
                "error": f"Summarization failed for all selected document(s): {error_detail}",
                "final_report": "I couldn't generate a summary — none of the selected documents had readable content.",
                "citations": [],
                "current_stage": "complete",
            }

        if len(succeeded) == 1 and not failed:
            final_report = succeeded[0]["summary"]
        else:
            # Multiple docs (or partial failure) — label each summary by
            # source so the user can tell which content came from where.
            parts = [f"**{r['source_name']}**\n{r['summary']}" for r in succeeded]
            if failed:
                parts.append(
                    "\n_Note: could not summarize "
                    + ", ".join(r["doc_id"] for r in failed)
                    + " — no readable content found._"
                )
            final_report = "\n\n---\n\n".join(parts)

        citations = [
            {"source_name": r["source_name"], "doc_id": r["doc_id"], "confidence": "summary"}
            for r in succeeded
        ]

        logger.info(
            f"[summarize] completed {len(succeeded)}/{len(doc_ids)} document(s), "
            f"{len(failed)} failed"
        )

        return {
            "final_report": final_report,
            "citations": citations,
            "current_stage": "complete",
        }