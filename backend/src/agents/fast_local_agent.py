"""Fast-path agent for simple local-only queries — skips claim extraction
and Groq verification. Answers directly from top reranked local chunks
via the configured Groq model. Falls back to the full pipeline if local
retrieval is empty or too weak to trust."""
import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.state import ResearchState
from src.search_providers.local_rag_provider import LocalRAGProvider
from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

FAST_LOCAL_PROMPT = """Answer the question using ONLY the context below. If the context does not contain the answer, say so plainly — do not guess.

Context:
{context}

Question: {question}

Answer concisely, citing which excerpt(s) you used by their [N] number."""


class FastLocalAgent:
    def __init__(self, llm=None, local_provider=None):
        self.llm = llm or get_llm(
            temperature=0.0,
            model_override="openai/gpt-oss-120b",
            provider_override="groq",
            max_tokens=800,
        )
        self.local_provider = local_provider or LocalRAGProvider()

    async def answer(self, state: ResearchState) -> Dict[str, Any]:
        if "local" not in state.sources_available:
            logger.info("[fast_local] no local sources available — escalating to full pipeline")
            return {"route_decision": "escalate"}

        query = state.research_topic
        results = await self.local_provider.search(
            query,
            max_results=5,
            doc_ids=state.selected_doc_ids or None,
        )

        if not results or len(results[0].content or "") < 50:
            logger.info("[fast_local] empty/weak local results — escalating to full pipeline")
            return {"route_decision": "escalate"}

        context_blocks, citations = [], []
        for i, r in enumerate(results, start=1):
            context_blocks.append(f"[{i}] ({r.source_name}) {r.content}")
            citations.append({
                "index": i,
                "source_name": r.source_name,
                "url": r.url,
                "confidence": "fast_local",
            })

        prompt = ChatPromptTemplate.from_messages([("human", FAST_LOCAL_PROMPT)])
        chain = prompt | self.llm | StrOutputParser()

        try:
            answer_text = await chain.ainvoke({"context": "\n\n".join(context_blocks), "question": query})
        except Exception as e:
            logger.warning(f"[fast_local] synthesis failed, escalating: {e}")
            return {"route_decision": "escalate"}

        logger.info(f"[fast_local] answered directly from {len(results)} local chunk(s)")
        return {
            "final_report": answer_text,
            "citations": citations,
            "current_stage": "complete",
            "route_decision": "done",
        }