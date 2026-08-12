"""Planner Agent — decomposes query into sub-queries + source routing."""
import asyncio
import logging
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.state import ResearchState, ResearchPlan, SearchQuery
from src.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from src.config import config
from src.exceptions import PlanningError
from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, llm=None, max_retries: int = 3):
        self.llm = llm or get_llm(temperature=0.5)
        self.max_retries = max_retries

    async def plan(self, state: ResearchState) -> Dict[str, Any]:
        local_available = "local" in state.sources_available
        local_note = (
            "Local documents ARE available this run — route document-specific queries to 'local'."
            if local_available
            else "No local documents available this run — use 'web' for all queries."
        )

        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            max_queries=config.max_search_queries,
            max_sections=config.max_report_sections,
            local_docs_note=local_note,
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", PLANNER_USER_TEMPLATE)])

        for attempt in range(self.max_retries):
            try:
                chain = prompt | self.llm | JsonOutputParser()
                result = await chain.ainvoke({
                    "topic": state.research_topic,
                    "local_docs_available": local_available,
                })

                if not all(k in result for k in ("topic", "objectives", "search_queries", "report_outline")):
                    raise PlanningError("Invalid plan structure")
                if not result["search_queries"]:
                    raise PlanningError("No search queries generated")

                plan = ResearchPlan(
                    topic=result["topic"],
                    objectives=result["objectives"][:5],
                    search_queries=[
                        SearchQuery(
                            query=sq["query"],
                            purpose=sq["purpose"],
                            source_hint=sq.get("source_hint", "web"),
                        )
                        for sq in result["search_queries"][: config.max_search_queries]
                    ],
                    report_outline=result["report_outline"][: config.max_report_sections],
                )

                return {"plan": plan, "current_stage": "searching", "iterations": state.iterations + 1}

            except Exception as e:
                logger.warning(f"Planning attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return {"error": f"Planning failed: {e}", "iterations": state.iterations + 1}
                await asyncio.sleep(2 ** attempt)

        return {"error": "Planning failed: max retries exceeded"}