"""Planner Agent — decomposes query into sub-queries + source routing."""
import re
import asyncio
import logging
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.state import ResearchState, ResearchPlan, SearchQuery
from src.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from src.config import config, COMPLEXITY_LIMITS
from src.exceptions import PlanningError
from datetime import datetime, timezone
from src.utils.llm_factory import get_llm
from metrics.token_counter import track_llm_call

logger = logging.getLogger(__name__)


SIMPLE_TOPIC_PATTERNS = [
    r"^(who|what|when|where)\s+is\b",
    r"^define\b",
    r"^what does .* mean\??$",
]

COMPARISON_PATTERNS = [
    r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bcomparison\b",
    r"\bdifferences? between\b",
]

def _deterministic_tier_floor(topic: str) -> Optional[str]:
    """Returns a forced complexity tier if the topic clearly matches a
    pattern needing a specific tier, else None (defer to LLM judgment)."""
    normalized = topic.strip().lower()
    word_count = len(normalized.split())
    if word_count <= 8:
        for pattern in SIMPLE_TOPIC_PATTERNS:
            if re.match(pattern, normalized):
                return "simple"
    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, normalized):
            return "complex"
    return None


class PlannerAgent:
    def __init__(self, llm=None, max_retries: int = 3):
        self.llm = llm or get_llm(temperature=0.5)
        self.max_retries = max_retries
        self.model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "unknown")

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
            current_date=datetime.now(timezone.utc).strftime('%B %d, %Y'),
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", PLANNER_USER_TEMPLATE)])

        for attempt in range(self.max_retries):
            try:
                chain = prompt | self.llm | JsonOutputParser()
                result = await track_llm_call(
                    chain,
                    {"topic": state.research_topic, "local_docs_available": local_available},
                    tracker=state.token_tracker,
                    node="plan",
                    model=self.model_name,
                    provider=config.model_provider,
                )

                if not all(k in result for k in ("topic", "objectives", "search_queries", "report_outline")):
                    raise PlanningError("Invalid plan structure")
                if not result["search_queries"]:
                    raise PlanningError("No search queries generated")

                llm_complexity = result.get("complexity", "moderate")
                if llm_complexity not in COMPLEXITY_LIMITS:
                    logger.warning(f"Planner returned unknown complexity={llm_complexity!r}, defaulting to 'moderate'")
                    llm_complexity = "moderate"

                deterministic_floor = _deterministic_tier_floor(state.research_topic)
                complexity = deterministic_floor or llm_complexity
                if deterministic_floor and deterministic_floor != llm_complexity:
                    logger.info(
                        f"[complexity] LLM rated '{llm_complexity}' but topic matches simple-lookup "
                        f"pattern — overriding to 'simple'"
                    )

                tier = COMPLEXITY_LIMITS[complexity]
                query_cap = min(tier["max_queries"], config.max_search_queries)

                logger.info(
                    f"[complexity] topic={state.research_topic!r} tier={complexity} "
                    f"query_cap={query_cap} (LLM proposed {len(result['search_queries'])})"
                )

                plan = ResearchPlan(
                    topic=result["topic"],
                    objectives=result["objectives"][:5],
                    search_queries=[
                        SearchQuery(
                            query=sq["query"],
                            purpose=sq["purpose"],
                            source_hint=sq.get("source_hint", "web"),
                        )
                        for sq in result["search_queries"][:query_cap]
                    ],
                    report_outline=result["report_outline"][: config.max_report_sections],
                    complexity=complexity,
                )

                return {"plan": plan, "current_stage": "searching", "iterations": state.iterations + 1}

            except Exception as e:
                logger.warning(f"Planning attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return {"error": f"Planning failed: {e}", "iterations": state.iterations + 1}
                await asyncio.sleep(2 ** attempt)

        return {"error": "Planning failed: max retries exceeded"}