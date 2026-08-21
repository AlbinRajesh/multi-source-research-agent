"""Router Agent — classifies input as casual chat, a capability/meta
question about the assistant itself, or a real research request.

Capability questions ("what can you do", "are you able to research X")
are matched deterministically first — same pattern as the planner's
complexity-tier floor — because a 3B local model is not reliable enough
on its own to distinguish "are you able to research" (a question about
the assistant) from "research the following" (an actual request). Only
genuinely ambiguous messages fall through to the LLM classifier.
"""
import re
import logging
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from src.state import ResearchState
from src.prompts.router_prompt import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from src.utils.llm_factory import get_llm
from src.config import config

logger = logging.getLogger(__name__)

CAPABILITY_PATTERNS = [
    r"\bwhat (can|do) you do\b",
    r"\bwhat are (your|the) (features|capabilit(y|ies))\b",
    r"\bwhat features (do you have|does this have)\b",
    r"\bare you able to\b",
    r"\bcan you (do|help with|research)\b.{0,30}$",  # short trailing clause, not "can you research X for me and give details on Y"
    r"\bhow do(es)? (you|this) work\b",
    r"\bwho are you\b",
    r"\bwhat is this (tool|app|assistant|agent)\b",
    r"\bwhat (kind of|type of) (questions|topics) can\b",
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|yo|sup)\b",
    r"^(thanks|thank you|thx|ty)\b",
    r"^(bye|goodbye|see ya)\b",
    r"^how are you\b",
]

CAPABILITY_ANSWER = """I'm a research assistant. I search the web across multiple sources, extract factual claims, cross-check each one against the original source and against other sources, and give you back a cited report — every fact is traceable to where it came from and flagged as verified, single-source, or conflicting.

Ask me things like:
- "What is [topic]"
- "Give me details on [topic]"
- "Compare [X] vs [Y]"
- "Latest developments in [topic]"

For anything that isn't a real-world lookup, just chat with me normally."""


def _deterministic_route(message: str) -> Optional[str]:
    normalized = message.strip().lower()
    for pattern in CAPABILITY_PATTERNS:
        if re.search(pattern, normalized):
            return "capability"
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, normalized):
            return "casual"
    return None


class RouterAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm(
            temperature=0.0,
            model_override=config.claim_extraction_model,
            provider_override="ollama",
        )
        self.chat_llm = get_llm(
            temperature=0.6,
            model_override=config.claim_extraction_model,
            provider_override="ollama",
        )

    async def route(self, state: ResearchState) -> Dict[str, Any]:
        topic = state.research_topic
        forced = _deterministic_route(topic)

        if forced == "capability":
            logger.info(f"[router] deterministic match -> capability: {topic!r}")
            return self._casual_result(CAPABILITY_ANSWER)

        if forced == "casual":
            logger.info(f"[router] deterministic match -> casual: {topic!r}")
            reply = await self._casual_reply(topic)
            return self._casual_result(reply)

        # Ambiguous — defer to the local classifier
        try:
            prompt = ChatPromptTemplate.from_messages(
                [("system", ROUTER_SYSTEM_PROMPT), ("human", ROUTER_USER_TEMPLATE)]
            )
            chain = prompt | self.llm
            result = await chain.ainvoke({"message": topic})
            label = result.content.strip().lower()
        except Exception as e:
            logger.warning(f"Router classification failed, defaulting to research: {e}")
            label = "research"

        if "casual" in label:
            reply = await self._casual_reply(topic)
            return self._casual_result(reply)

        return {"is_casual": False, "current_stage": "planning"}

    def _casual_result(self, reply: str) -> Dict[str, Any]:
        return {
            "is_casual": True,
            "final_report": reply,
            "citations": [],
            "current_stage": "complete",
        }

    async def _casual_reply(self, message: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful, professional research assistant for an "
                "enterprise research tool. Reply briefly and naturally.",
            ),
            ("human", "{message}"),
        ])
        chain = prompt | self.chat_llm
        result = await chain.ainvoke({"message": message})
        return result.content.strip()