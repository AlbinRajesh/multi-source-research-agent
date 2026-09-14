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
from src.config import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.state import ResearchState, VerificationVerdict
from src.prompts.verification_prompt import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_TEMPLATE
from src.utils.llm_factory import get_llm
from src.exceptions import VerificationError
from metrics.token_counter import track_llm_call

logger = logging.getLogger(__name__)


class VerificationAgent:
    def __init__(self, llm=None, max_concurrent: int = 6):
        self.llm = llm or get_llm(
            temperature=0.0,
            model_override=config.nvidia_verifier_model,
            provider_override="nvidia",
            max_tokens=1200,
        )
        self.max_concurrent = max_concurrent
        self.model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "unknown")

    async def verify(self, state: ResearchState) -> Dict[str, Any]:
        if not state.claims:
            return {"verified_claims": [], "current_stage": "synthesizing"}

        # Group by source URL, not source_index — search_results gets
        # replaced on every retry, so a claim's original source_index can
        # point at a different (or out-of-range) document by the time
        # verify runs. URL is the stable identifier for lookup.
        url_to_doc = {doc.url: doc for doc in state.search_results}
        claims_by_url: Dict[str, list] = defaultdict(list)
        for c in state.claims:
            claims_by_url[c.source_url].append(c)

        prompt = ChatPromptTemplate.from_messages(
            [("system", VERIFICATION_SYSTEM_PROMPT), ("human", VERIFICATION_USER_TEMPLATE)]
        )
        chain = prompt | self.llm | StrOutputParser()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        grounded_results: Dict[str, List[str]] = defaultdict(list)  # claim_id -> [url, ...]
        contradicted_results: Dict[str, List[str]] = defaultdict(list)

        async def verify_source(url: str, claims: list):
            doc = url_to_doc.get(url)
            if not doc:
                logger.info(f"[verify] source no longer available for claims: {url}")
                return

            text = doc.content or doc.snippet
            if not text:
                return
            if not doc.content and len(text) < 200:
                logger.info(f"[verify] skipping snippet-only source (no full content): {url}")
                return

            claims_block = "\n".join(f'- id: {c.id} | "{c.text}"' for c in claims)
            logger.info(f"[debug] {url} content_len={len(text)} truncated={len(text) > 6000}")
            async with semaphore:
                try:
                    raw = await track_llm_call(
                    chain,
                    {
                        "claims_block": claims_block,
                        "source_name": doc.source_name or doc.url,
                        "source_text": self._get_relevant_excerpt(text, claims),
                    },
                    tracker=state.token_tracker,
                    node="verify",
                    model=self.model_name,
                    provider="nvidia",
                )
                    logger.info(f"[raw_length] verify source={url} chars={len(raw)}")
                    parsed = self._parse_json_array(raw)
                    for item in parsed:
                        cid = item.get("claim_id")
                        if item.get("is_grounded"):
                            grounded_results[cid].append(url)
                        else:
                            logger.info(f"[debug] UNGROUNDED claim_id={cid} content_len={len(text)}")
                        if item.get("contradicts"):
                            contradicted_results[cid].append(url)
                except Exception as e:
                    logger.warning(f"Verification failed for source {url}: {e}")

        try:
            await asyncio.gather(*[verify_source(url, claims) for url, claims in claims_by_url.items()])

            # Resolve URLs -> current index positions only here, for
            # backward compatibility with corroborating_source_indices'
            # existing List[int] type (used downstream by citations.py).
            url_to_current_index = {doc.url: i for i, doc in enumerate(state.search_results)}

            verdicts: List[VerificationVerdict] = []
            for c in state.claims:
                source_urls = grounded_results.get(c.id, [])
                contradictions = contradicted_results.get(c.id, [])

                if contradictions:
                    confidence = "conflicting"
                elif len(source_urls) >= 2:
                    confidence = "verified"
                elif len(source_urls) == 1:
                    sole_doc = url_to_doc.get(source_urls[0])
                    if sole_doc and sole_doc.source_type == "local":
                        confidence = "verified"  # user's own doc — trusted once grounded, no 2nd-source requirement
                    else:
                        confidence = "single_source"
                else:
                    confidence = "unconfirmed"

                resolved_indices = [
                    url_to_current_index[u] for u in source_urls if u in url_to_current_index
                ]

                verdicts.append(VerificationVerdict(
                    claim_id=c.id,
                    is_grounded=len(source_urls) > 0,
                    confidence=confidence,
                    corroborating_source_indices=resolved_indices,
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
        except json.JSONDecodeError as e:
            try:
                repaired = repair_json(raw)
                data = json.loads(repaired)
                logger.info("Recovered malformed claim JSON via json_repair")
            except Exception:
                raise VerificationError("Failed to parse verification JSON", details=str(e))

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("verdicts", "items", "results", "data"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            logger.warning(f"[verification_parse] got JSON object with no known array key, keys={list(data.keys())}")
        return []

    @staticmethod
    def _get_relevant_excerpt(text: str, claims: list, window: int = 2000, max_total: int = 4000) -> str:
        """Pull text windows around claim-keyword matches instead of blind truncation."""
        if len(text) <= max_total:
            return text
        spans = []
        text_lower = text.lower()
        for c in claims:
            words = [w.strip('.,;:()"\'') for w in c.text.split()]
            words = [w for w in words if len(w) > 2]
            if not words:
                continue
            # anchor on the longest word — more distinctive than the first word,
            # less likely to match a generic early mention
            anchor = max(words, key=len)
            idx = text_lower.find(anchor.lower())
            if idx != -1:
                spans.append((max(0, idx - 200), min(len(text), idx + window)))
        if not spans:
            return text[:max_total]
        spans.sort()
        merged, cur = [], list(spans[0])
        for s in spans[1:]:
            if s[0] <= cur[1]:
                cur[1] = max(cur[1], s[1])
            else:
                merged.append(tuple(cur))
                cur = list(s)
        merged.append(tuple(cur))
        excerpt = " ... ".join(text[a:b] for a, b in merged)
        return excerpt[:max_total]