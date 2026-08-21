import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import config

from src.state import ResearchState
from src.utils.llm_factory import get_llm
from src.processing.citations import format_citations
from src.prompts.synthesis_prompt import (
    SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE,
    SYNTHESIS_SIMPLE_SYSTEM_PROMPT, SYNTHESIS_SIMPLE_USER_TEMPLATE,
)
from metrics.token_counter import track_llm_call

logger = logging.getLogger(__name__)

# Complexity tiers that get the short-answer treatment. Anything not in
# this set (moderate, complex, or an unexpected/missing value) falls
# through to the structured report — the SAFER default for an enterprise
# pipeline, since an over-long answer is a UX annoyance, but an
# under-explained answer on a genuinely complex topic is a quality/trust
# problem. Fail toward more structure, not less.
SIMPLE_COMPLEXITY_TIERS = {"simple"}


class SynthesizerAgent:
    MIN_USABLE_CLAIMS = 2  # below this, evidence is too thin to synthesize
    # a meaningful answer from — return an honest "insufficient evidence"
    # response instead of forcing the LLM to produce prose from whatever
    # scraps survived. This is a general safeguard, not tied to any
    # particular topic — it fires whenever the pipeline's upstream stages
    # (search/credibility/relevance/verify) collectively leave too little
    # real evidence, for any subject.

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.3, max_tokens=2000)
        self.model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "unknown")

    async def synthesize(self, state: ResearchState) -> Dict[str, Any]:
        try:
            claim_by_id = {c.id: c for c in state.claims}
            usable = [
                v for v in state.verified_claims if v.confidence in ("verified", "single_source")
            ]
            unconfirmed = [v for v in state.verified_claims if v.confidence == "unconfirmed"]

            # Low-evidence floor — fires for ANY topic where too few
            # claims survived to verify/relevance/credibility filtering,
            # not a subject-specific check.
            if len(usable) < self.MIN_USABLE_CLAIMS:
                logger.warning(
                    f"[synthesis_floor] only {len(usable)} usable claim(s) survived "
                    f"(min={self.MIN_USABLE_CLAIMS}) — returning insufficient-evidence response"
                )
                fallback_text = (
                    f"I wasn't able to find enough verified information to answer "
                    f"\"{state.research_topic}\" with confidence. "
                )
                if usable:
                    claim = claim_by_id.get(usable[0].claim_id)
                    if claim:
                        fallback_text += f"The only verifiable detail found was: {claim.text}"
                return {
                    "final_report": fallback_text,
                    "citations": [],
                    "current_stage": "complete",
                    "iterations": state.iterations + 1,
                }

            citation_index_map = {}
            url_to_index = {}
            next_citation_idx = 1

            claims_lines = []
            for v in usable:
                claim = claim_by_id.get(v.claim_id)
                if not claim:
                    continue
                doc = state.search_results[claim.source_index] if claim.source_index < len(state.search_results) else None
                if not doc:
                    continue

                if doc.url not in url_to_index:
                    url_to_index[doc.url] = next_citation_idx
                    next_citation_idx += 1

                cite_idx = url_to_index[doc.url]
                citation_index_map[doc.url] = cite_idx

                tag = "VERIFIED" if v.confidence == "verified" else "single-source"
                claims_lines.append(f'[{cite_idx}] ({tag}) {claim.text}')

            logger.info(f"[debug] usable={len(usable)} claims_lines={len(claims_lines)} skipped={len(usable)-len(claims_lines)}")

            unconfirmed_lines = [
                f'- {claim_by_id[v.claim_id].text}' for v in unconfirmed if v.claim_id in claim_by_id
            ][:10]

            current_date = datetime.now(timezone.utc).strftime('%B %d, %Y')

            complexity = getattr(state.plan, "complexity", None) if state.plan else None
            use_simple = complexity in SIMPLE_COMPLEXITY_TIERS

            if use_simple:
                logger.info(f"[synthesis_mode] complexity='{complexity}' -> simple direct-answer format")
                system_prompt = SYNTHESIS_SIMPLE_SYSTEM_PROMPT
                user_template = SYNTHESIS_SIMPLE_USER_TEMPLATE
                input_vars = {
                    "topic": state.research_topic,
                    "claims_block": "\n".join(claims_lines) or "(none)",
                    "unconfirmed_block": "\n".join(unconfirmed_lines) or "(none)",
                    "current_date": current_date,
                }
            else:
                logger.info(f"[synthesis_mode] complexity='{complexity}' -> structured report format")
                system_prompt = SYNTHESIS_SYSTEM_PROMPT
                user_template = SYNTHESIS_USER_TEMPLATE
                objectives = getattr(state.plan, "objectives", None) or []
                objectives_block = "\n".join(f"{i+1}. {obj}" for i, obj in enumerate(objectives)) or "(none specified)"
                input_vars = {
                    "topic": state.research_topic,
                    "objectives_block": objectives_block,
                    "claims_block": "\n".join(claims_lines) or "(none)",
                    "unconfirmed_block": "\n".join(unconfirmed_lines) or "(none)",
                    "current_date": current_date,
                }

            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", user_template)]
            )
            chain = prompt | self.llm | StrOutputParser()

            answer = await track_llm_call(
                chain,
                input_vars,
                tracker=getattr(state, "token_tracker", None),
                node="synthesize",
                model=self.model_name,
                provider=config.model_provider,
            )

            # Added debug log here to inspect the raw LLM output
            logger.info(f"[debug] synth answer len={len(answer)} preview={answer[:200]!r}")

            citations = format_citations(
                usable, claim_by_id, citation_index_map, state.search_results, style="apa"
            )

            return {
                "final_report": answer,
                "citations": citations,
                "current_stage": "complete",
                "iterations": state.iterations + 1,
            }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {"error": f"Synthesis failed: {e}"}