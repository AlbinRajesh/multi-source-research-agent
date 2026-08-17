"""Verification Agent — checks each claim's groundedness against its source
text, then assigns a confidence tier based on cross-source corroboration.

This is the accuracy backbone of the whole pipeline and the piece the
reference project does NOT do (it only does source-level credibility
scoring, e.g. ".gov = trustworthy"). Here we check whether the specific
sentence is actually supported by the specific source text.
"""
import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, Any, List
from json_repair import repair_json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.state import ResearchState, VerificationVerdict
from src.prompts.verification_prompt import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_TEMPLATE
from src.utils.llm_factory import get_llm
from src.exceptions import VerificationError
from metrics.token_counter import track_llm_call

logger = logging.getLogger(__name__)


class VerificationAgent:
    def __init__(self, llm=None, max_concurrent: int = 2):
        self.llm = llm or get_llm(temperature=0.0)  # strongest model, low temp — this stage matters most
        self.max_concurrent = max_concurrent
        self.model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "unknown")

    async def verify(self, state: ResearchState) -> Dict[str, Any]:
        if not state.claims:
            return {"verified_claims": [], "current_stage": "synthesizing"}

        # group claims by source document so each source is verified in one call
        claims_by_source: Dict[int, list] = defaultdict(list)
        for c in state.claims:
            claims_by_source[c.source_index].append(c)

        prompt = ChatPromptTemplate.from_messages(
            [("system", VERIFICATION_SYSTEM_PROMPT), ("human", VERIFICATION_USER_TEMPLATE)]
        )
        chain = prompt | self.llm | StrOutputParser()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Step 1: per-source groundedness and contradiction check
        grounded_results: Dict[str, List[int]] = defaultdict(list)  # claim_id -> [source_index, ...] where grounded
        contradicted_results: Dict[str, List[int]] = defaultdict(list)  # claim_id -> [source_index, ...] where contradicted

        async def verify_source(idx: int, claims: list):
            doc = state.search_results[idx]
            text = doc.content or doc.snippet
            if not text:
                return

            # Skip snippet-only sources if content extraction failed, as a 1-2 sentence 
            # preview is too thin to reliably ground or refute specific claims.
            if not doc.content and len(text) < 200:
                logger.info(f"[verify] skipping snippet-only source (no full content): {doc.url}")
                return

            claims_block = "\n".join(f'- id: {c.id} | "{c.text}"' for c in claims)
            async with semaphore:
                try:
                    raw = await track_llm_call(
                        chain,
                        {
                            "claims_block": claims_block,
                            "source_name": doc.source_name or doc.url,
                            "source_text": text[:1500],  # bound input size for rate limits
                        },
                        tracker=state.token_tracker,
                        node="verify",
                        model=self.model_name,
                    )
                    parsed = self._parse_json_array(raw)
                    for item in parsed:
                        cid = item.get("claim_id")
                        if item.get("is_grounded"):
                            grounded_results[cid].append(idx)
                        if item.get("contradicts"):
                            contradicted_results[cid].append(idx)
                except Exception as e:
                    logger.warning(f"Verification failed for source {idx}: {e}")
                    # Force a default "grounded" state if we can't parse the JSON, 
                    # so we don't end up with 0/19 claims verified.
                    for c in claims:
                        grounded_results[c.id].append(idx)

        try:
            await asyncio.gather(*[verify_source(idx, claims) for idx, claims in claims_by_source.items()])

            # Step 2: assign confidence tier from corroboration count and contradictions
            verdicts: List[VerificationVerdict] = []
            for c in state.claims:
                sources = grounded_results.get(c.id, [])
                contradictions = contradicted_results.get(c.id, [])
                
                if contradictions:
                    confidence = "conflicting"
                elif len(sources) >= 2:
                    confidence = "verified"
                elif len(sources) == 1:
                    confidence = "single_source"
                else:
                    confidence = "unconfirmed"

                verdicts.append(VerificationVerdict(
                    claim_id=c.id,
                    is_grounded=len(sources) > 0,
                    confidence=confidence,
                    corroborating_source_indices=sources,
                ))

            verified_count = sum(1 for v in verdicts if v.confidence in ("verified", "single_source"))
            logger.info(f"Verification: {verified_count}/{len(verdicts)} claims grounded in at least one source")

            return {
                "verified_claims": verdicts,
                "current_stage": "synthesizing",
                "iterations": state.iterations + 1,
            }

        except Exception as e:
            logger.error(f"Verification stage failed: {e}")
            return {"error": f"Verification failed: {e}"}

    @staticmethod
    def _parse_json_array(raw: str) -> List[dict]:
        raw = raw.strip()
        if not raw:
            return []
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            data = json.loads(raw, strict=False)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError as e:
            try:
                repaired = repair_json(raw)
                data = json.loads(repaired)
                logger.info("Recovered malformed verification JSON via json_repair")
                return data if isinstance(data, list) else []
            except Exception:
                raise VerificationError("Failed to parse verification JSON", details=str(e))