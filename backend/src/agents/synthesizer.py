import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import config

from src.state import ResearchState
import re
from src.utils.llm_factory import get_llm
from src.processing.citations import format_citations
from src.prompts.synthesis_prompt import (
    SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE,
    SYNTHESIS_SIMPLE_SYSTEM_PROMPT, SYNTHESIS_SIMPLE_USER_TEMPLATE,
)
from metrics.token_counter import track_llm_call
from src.processing.output_format import (
    enforce_output_format,
    format_instruction,
)

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
        # Two separate instances: simple mode's prompt is short (1-4 sentence
        # answer, few claims) and fits comfortably in a small budget. Structured
        # mode's prompt is much longer and the model is a reasoning model —
        # hidden reasoning tokens draw from the same max_tokens budget as the
        # visible answer, so a tight ceiling here silently truncates the
        # completion to nothing before any output text is written.
        self.llm = llm or get_llm(temperature=0.3, max_tokens=2000)
        self.structured_llm = llm or get_llm(temperature=0.3, max_tokens=8000)
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
            use_simple = complexity in SIMPLE_COMPLEXITY_TIERS or state.force_simple_format
            output_format = state.output_format or {"style": "default"}
            requested_instruction = format_instruction(output_format)

            if use_simple:
                logger.info(f"[synthesis_mode] complexity='{complexity}' -> simple direct-answer format")
                system_prompt = SYNTHESIS_SIMPLE_SYSTEM_PROMPT
                user_template = SYNTHESIS_SIMPLE_USER_TEMPLATE
                input_vars = {
                    "topic": state.research_topic,
                    "claims_block": "\n".join(claims_lines) or "(none)",
                    "unconfirmed_block": "\n".join(unconfirmed_lines) or "(none)",
                    "current_date": current_date,
                    "format_instruction": requested_instruction,
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
                    "format_instruction": requested_instruction,
                }

            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", user_template)]
            )
            active_llm = self.structured_llm if not use_simple else self.llm
            chain = prompt | active_llm | StrOutputParser()

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

            if not answer or not answer.strip():
                logger.error(
                    f"[synthesis_empty] LLM returned empty completion "
                    f"(mode={'simple' if use_simple else 'structured'}, "
                    f"claims={len(claims_lines)}) — surfacing as error instead "
                    f"of an empty report"
                )
                return {
                    "error": (
                        "The research completed but the answer generation step "
                        "returned empty — this usually means the response was cut "
                        "off. Please try again."
                    )
                }

            formatted = enforce_output_format(answer, output_format)

            cited_indices = {int(n) for n in re.findall(r"\[(\d+)\]", formatted["text"])}

            citations = format_citations(
                usable, claim_by_id, citation_index_map, state.search_results,
                style="apa", cited_indices=cited_indices,
            )

            return {
                "final_report": formatted["text"],
                "citations": citations,
                "output_format": formatted,
                "current_stage": "complete",
                "iterations": state.iterations + 1,
            }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {"error": f"Synthesis failed: {e}"}