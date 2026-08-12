"""Claim Extraction Agent — breaks retrieved documents into atomic,
independently checkable claims. New stage vs. the reference project;
this is what makes claim-level verification (next stage) possible.
"""
import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.state import ResearchState, Claim
from src.prompts.claim_extraction_prompt import CLAIM_EXTRACTION_SYSTEM_PROMPT, CLAIM_EXTRACTION_USER_TEMPLATE
from src.utils.llm_factory import get_llm
from src.exceptions import ClaimExtractionError

logger = logging.getLogger(__name__)


class ClaimExtractionAgent:
    def __init__(self, llm=None, max_concurrent: int = 5):
        # fast/cheap model — this stage runs once per document, keep it light
        self.llm = llm or get_llm(temperature=0.0, model_override_key="fast_model_name")
        self.max_concurrent = max_concurrent

    async def extract(self, state: ResearchState) -> Dict[str, Any]:
        if not state.search_results:
            return {"claims": [], "current_stage": "synthesizing"}

        # dedup should already have run inside the retriever/processing step;
        # here we assume state.search_results is already deduped
        prompt = ChatPromptTemplate.from_messages(
            [("system", CLAIM_EXTRACTION_SYSTEM_PROMPT), ("human", CLAIM_EXTRACTION_USER_TEMPLATE)]
        )
        chain = prompt | self.llm | StrOutputParser()

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def extract_one(idx: int, doc) -> List[Claim]:
            text = doc.content or doc.snippet
            if not text:
                return []
            async with semaphore:
                try:
                    raw = await chain.ainvoke({
                        "source_name": doc.source_name or doc.url,
                        "document_text": text[:6000],  # bound input size
                    })
                    parsed = self._parse_json_array(raw)
                    return [
                        Claim(
                            id=str(uuid.uuid4())[:8],
                            text=item["text"],
                            source_url=doc.url,
                            source_index=idx,
                        )
                        for item in parsed
                        if item.get("text")
                    ]
                except Exception as e:
                    logger.warning(f"Claim extraction failed for {doc.url}: {e}")
                    return []

        new_indices = [i for i in range(len(state.search_results)) if i not in set(state.processed_result_indices)]
        if not new_indices:
            return {"current_stage": "verifying"}

        try:
            results = await asyncio.gather(
                *[extract_one(i, state.search_results[i]) for i in new_indices]
            )
            new_claims = [c for sub in results for c in sub]
            logger.info(f"Extracted {len(new_claims)} claims from {len(new_indices)} new docs")

            return {
                "claims": state.claims + new_claims,
                "processed_result_indices": state.processed_result_indices + new_indices,
                "current_stage": "verifying",
                "iterations": state.iterations + 1,
            }
        except Exception as e:
            logger.error(f"Claim extraction stage failed: {e}")
            return {"error": f"Claim extraction failed: {e}"}

    @staticmethod
    def _parse_json_array(raw: str) -> List[dict]:
        raw = raw.strip()
        # strip markdown fences if the model added them despite instructions
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            data = json.loads(raw, strict=False)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError as e:
            raise ClaimExtractionError("Failed to parse claim JSON", details=str(e))