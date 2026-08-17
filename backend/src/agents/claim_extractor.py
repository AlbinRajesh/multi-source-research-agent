"""Claim Extraction Agent — breaks retrieved documents into atomic,
independently checkable claims. New stage vs. the reference project;
this is what makes claim-level verification (next stage) possible.
"""
import re
import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Optional
from json_repair import repair_json
from src.config import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.state import ResearchState, Claim
from src.prompts.claim_extraction_prompt import CLAIM_EXTRACTION_SYSTEM_PROMPT, CLAIM_EXTRACTION_USER_TEMPLATE
from src.utils.llm_factory import get_llm
from src.exceptions import ClaimExtractionError
from metrics.token_counter import track_llm_call

logger = logging.getLogger(__name__)


class ClaimExtractionAgent:
    MAX_CLAIMS_PER_DOC = 6  # deterministic ceiling — do not rely on prompt compliance alone

    BOILERPLATE_PATTERNS = [
        r"^\[.*\]\(#.*\)",
        r"\bsubscribe\b.*\bpro\b",
        r"__source=|tpcc=|utm_",
        r"^(cookie|privacy) (policy|notice|settings)",
        r"^(home|menu|navigation|sign in|log in|sign up)$",
        r"\[.*\]\(https?://\w{2,3}\.\w+\.(org|com)/",
    ]

    def __init__(self, llm=None, max_concurrent: int = 2):
        # fast/cheap model — this stage runs once per document, keep it light
        self.llm = llm or get_llm(
            temperature=0.0,
            model_override=config.claim_extraction_model,
            provider_override="ollama",
            max_tokens=600,  # bounds generation length -> bounds both latency and claim count
        )
        self.max_concurrent = max_concurrent
        self.model_name = config.claim_extraction_model
        self._boilerplate_re = re.compile("|".join(self.BOILERPLATE_PATTERNS), re.IGNORECASE)

    def _is_boilerplate(self, text: str) -> bool:
        if not text or len(text.strip()) < 15:
            return True  # too short to be a real claim
        return bool(self._boilerplate_re.search(text))

    async def extract(self, state: ResearchState) -> Dict[str, Any]:
        if not state.search_results:
            return {"claims": [], "current_stage": "synthesizing"}

        # Track processed documents by stable URL instead of fragile list indices,
        # so retries correctly pick up newly added search results.
        processed_urls = {c.source_url for c in state.claims}
        docs_to_process = [
            (i, doc) for i, doc in enumerate(state.search_results)
            if doc.url not in processed_urls
        ]

        if not docs_to_process:
            return {"current_stage": "verifying"}

        prompt = ChatPromptTemplate.from_messages(
            [("system", CLAIM_EXTRACTION_SYSTEM_PROMPT), ("human", CLAIM_EXTRACTION_USER_TEMPLATE)]
        )
        chain = prompt | self.llm | StrOutputParser()

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def extract_one(idx: int, doc) -> List[Claim]:
            text = doc.content or doc.snippet
            logger.info(f"[debug] {doc.url} content_len={len(text)} preview={text[:300]!r}")
            if not text:
                return []
            async with semaphore:
                raw = None
                last_error = None
                for attempt in range(3):  # 3 attempts with short exponential backoff for resilience
                    try:
                        raw = await track_llm_call(
                            chain,
                            {
                                "source_name": doc.source_name or doc.url,
                                "document_text": text[:3000],
                            },
                            tracker=state.token_tracker,
                            node="extract_claims",
                            model=self.model_name,
                            provider="ollama",
                        )
                        break  # success
                    except Exception as e:
                        last_error = e
                        if attempt < 2:
                            logger.warning(f"[retry] extraction attempt {attempt+1} failed for {doc.url}: {e}")
                            await asyncio.sleep(1.5 * (attempt + 1))  # 1.5s, then 3s

                if raw is None:
                    logger.warning(f"Claim extraction failed for {doc.url} after 3 attempts: {last_error}")
                    return []

                try:
                    parsed = self._parse_json_array(raw)
                    if not parsed:
                        logger.warning(f"[claim_parse] zero claims parsed from {doc.url} — raw response: {raw[:200]}")

                    # Filter boilerplate BEFORE capping, so the cap counts real claims, not junk
                    parsed = [item for item in parsed if not self._is_boilerplate(item.get("text", ""))]

                    # HARD CAP — deterministic, independent of prompt compliance.
                    if len(parsed) > self.MAX_CLAIMS_PER_DOC:
                        logger.info(
                            f"[claim_cap] {doc.url}: model returned {len(parsed)} valid claims, "
                            f"capping to {self.MAX_CLAIMS_PER_DOC}"
                        )
                        parsed = parsed[: self.MAX_CLAIMS_PER_DOC]

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
                    logger.warning(f"Claim extraction failed parsing for {doc.url}: {e}")
                    return []

        try:
            results = await asyncio.gather(
                *[extract_one(i, doc) for i, doc in docs_to_process]
            )
            new_claims = [c for sub in results for c in sub]
            logger.info(f"Extracted {len(new_claims)} claims from {len(docs_to_process)} new docs")

            # Map all current search result indices for index tracking completeness
            all_indices = list(range(len(state.search_results)))

            return {
                "claims": state.claims + new_claims,
                "processed_result_indices": all_indices,
                "current_stage": "verifying",
                "iterations": state.iterations + 1,
            }
        except Exception as e:
            logger.error(f"Claim extraction stage failed: {e}")
            return {"error": f"Claim extraction failed: {e}"}

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
                raise ClaimExtractionError("Failed to parse claim JSON", details=str(e))

        if isinstance(data, list):
            # Occasionally the model nests claims by category, returning
            # a list of lists instead of a flat array — flatten one level
            # before validating, and drop anything that isn't a proper
            # {"text": ...} claim dict. Previously this returned `data`
            # unchecked, so a nested list reached `item.get("text", "")`
            # downstream and crashed with 'list' object has no attribute
            # 'get', silently losing that whole document's claims.
            if data and all(isinstance(item, list) for item in data):
                data = [item for sub in data for item in sub]
            return [item for item in data if isinstance(item, dict) and "text" in item]

        if isinstance(data, dict):
            found = ClaimExtractionAgent._find_claim_list(data)
            if found is not None:
                return found
            logger.warning(f"[claim_parse] got JSON object with no usable claim list, keys={list(data.keys())}")

        return []

    @staticmethod
    def _find_claim_list(obj, depth: int = 0) -> Optional[List[dict]]:
        if depth > 3:
            return None
        if isinstance(obj, list):
            if obj and all(isinstance(item, dict) and "text" in item for item in obj):
                return obj
            return None
        if isinstance(obj, dict):
            for value in obj.values():
                result = ClaimExtractionAgent._find_claim_list(value, depth + 1)
                if result is not None:
                    return result
        return None