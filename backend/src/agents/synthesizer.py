import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.state import ResearchState
from src.utils.llm_factory import get_llm
from src.processing.citations import format_citations
from src.prompts.synthesis_prompt import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE
from metrics.token_counter import track_llm_call

logger = logging.getLogger(__name__)


class SynthesizerAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.3)
        # model name for token/cost attribution — adjust attribute if your
        # provider wrapper exposes it differently (ChatOpenAI uses
        # .model_name, some wrappers use .model)
        self.model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "unknown")

    async def synthesize(self, state: ResearchState) -> Dict[str, Any]:
        try:
            claim_by_id = {c.id: c for c in state.claims}
            usable = [
                v for v in state.verified_claims if v.confidence in ("verified", "single_source")
            ]
            unconfirmed = [v for v in state.verified_claims if v.confidence == "unconfirmed"]

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

            unconfirmed_lines = [
                f'- {claim_by_id[v.claim_id].text}' for v in unconfirmed if v.claim_id in claim_by_id
            ][:10]

            # NOTE: guessing state.plan.objectives as the field name — the
            # "Research Objectives" section in the new prompt needs this.
            # If your Plan model names it differently, fix this line.
            objectives = getattr(state.plan, "objectives", None) or []
            objectives_block = "\n".join(f"{i+1}. {obj}" for i, obj in enumerate(objectives)) or "(none specified)"

            prompt = ChatPromptTemplate.from_messages(
                [("system", SYNTHESIS_SYSTEM_PROMPT), ("human", SYNTHESIS_USER_TEMPLATE)]
            )
            chain = prompt | self.llm | StrOutputParser()

            answer = await track_llm_call(
                chain,
                {
                    "topic": state.research_topic,
                    "objectives_block": objectives_block,
                    "claims_block": "\n".join(claims_lines) or "(none)",
                    "unconfirmed_block": "\n".join(unconfirmed_lines) or "(none)",
                },
                tracker=getattr(state, "token_tracker", None),
                node="synthesize",
                model=self.model_name,
            )

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