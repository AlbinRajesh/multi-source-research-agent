"""Router Agent — classifies input as casual chat, a capability/meta
question, a document-summarization request, or a real research request.

Capability and summary intents are matched deterministically first —
same pattern as the planner's complexity-tier floor — because a 3B
local model is not reliable enough on its own to distinguish these
from genuine research requests. Only genuinely ambiguous messages fall
through to the LLM classifier.
"""
import re
import logging
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from src.state import ResearchState
from src.config import config
from src.prompts.router_prompt import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from src.utils.llm_factory import get_llm
from src.processing.output_format import parse_output_format

logger = logging.getLogger(__name__)

CAPABILITY_PATTERNS = [
    r"\bwhat (can|do) you do\b",
    r"\bwhat are (your|the) (features|capabilit(y|ies))\b",
    r"\bwhat features (do you have|does this have)\b",
    r"\bare you able to\b",
    r"\bcan you (do|help with|research)\b.{0,30}$",
    r"\bhow do(es)? (you|this) work\b",
    r"\bwho are you\b",
    r"\bwhat is this (tool|app|assistant|agent)\b",
    r"\bwhat (kind of|type of) (questions|topics) can\b",
    r"\bwhat (kind of |type of )?(things|stuff|help) (can|could) you (do|offer|help with)\b",
    r"\bwhat can you help (me )?with\b",
    r"\bwhat (are you|are you all) capable of\b",
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|yo|sup)\b",
    r"^(thanks|thank you|thx|ty)\b",
    r"^(bye|goodbye|see ya)\b",
    r"^how are you\b",
]

# Summarization intent — checked before capability/greeting since phrases
# like "what is this document about" would otherwise partially match
# capability patterns ("what is this ... about").
SUMMARY_PATTERNS = [
    r"\bsummar(y|ize|ise|isation|ization)\b",
    # Typo-tolerant: "summery", "sumary", "summry", "sumery" etc.
    r"\bsum{1,2}[ae]r[yi]\b",
    r"\btl;?dr\b",
    r"\bwhat (is|'s) (this|the|it) (document|doc|pdf|file|paper|report) about\b",
    r"\bwhat('s| is) (it|this) about\b",
    r"\bgive me (a|the) (summary|overview|gist|rundown)\b",
    r"\bkey (points|takeaways) (of|from) (this|the)\b",
    r"\bcan you (summarize|summarise|sum up)\b",
    # Catch "give me the summery of the pdf" and similar constructs
    r"\bgive me.{0,20}(of|from|about).{0,20}(pdf|document|doc|file|paper|report)\b",
    r"\b(summary|summery|summar[iy]).{0,20}(pdf|document|doc|file|paper|report)\b",
    r"\b(pdf|document|doc|file|paper|report).{0,20}(summary|summery|summar[iy])\b",
]

# Enterprise-grade two-tier matching vocabularies
DOC_NOUN_PATTERN = r"\b(?:documents?|docs?|pdf|files?|papers?|reports?)\b"
ATTACH_TERM_PATTERN = r"\b(?:attach\w*|upload\w*|select\w*)\b"
STRICT_DOC_PHRASE_PATTERN = (
    r"\b(?:this|the|selected|uploaded|attached)\s+"
    r"(?:document|doc|pdf|file|paper|report)\b"
)

CAPABILITY_ANSWER = """I'm a research assistant. I search the web across multiple sources, extract factual claims, cross-check each one against the original source and against other sources, and give you back a cited report — every fact is traceable to where it came from and flagged as verified, single-source, or conflicting.

I can also summarize documents you've uploaded — just ask for a summary, optionally with a length ("summarize this in 3 sentences").

Ask me things like:
- "What is [topic]"
- "Give me details on [topic]"
- "Compare [X] vs [Y]"
- "Summarize this document"
- "Latest developments in [topic]"

For anything that isn't a real-world lookup, just chat with me normally."""

NO_DOCUMENT_FOR_SUMMARY_MSG = (
    "I'd be happy to summarize a document — but I don't see one selected for "
    "this conversation. Please upload or select a document first, then ask "
    "again."
)


def _is_summary_request(message: str) -> bool:
    normalized = message.strip().lower()
    return any(re.search(p, normalized) for p in SUMMARY_PATTERNS)


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
            provider_override=config.router_provider,
            model_override=config.router_model,
            api_key_override=config.router_api_key,
        )
        self.chat_llm = get_llm(
            temperature=0.6,
            model_override=config.router_model,
            provider_override=config.router_provider,
            api_key_override=config.router_api_key,
        )

    async def route(self, state: ResearchState) -> Dict[str, Any]:
        topic = state.research_topic
        output_format = parse_output_format(topic)

        if _is_summary_request(topic):
            normalized_topic = topic.lower()

            has_doc_noun = bool(re.search(DOC_NOUN_PATTERN, normalized_topic))
            has_attach_term = bool(re.search(ATTACH_TERM_PATTERN, normalized_topic))
            strict_phrase = bool(re.search(STRICT_DOC_PHRASE_PATTERN, normalized_topic))

            if state.selected_doc_ids:
                # Loose matching is safe because a document is actually present
                document_reference = has_doc_noun or has_attach_term or strict_phrase
            else:
                # Strict matching required to avoid false positives on web queries
                document_reference = strict_phrase or has_attach_term

            # Fallback for references like "summarize this / it"
            document_reference = document_reference or bool(
                state.selected_doc_ids
                and re.search(r"\b(?:summarize|summarise|summary|overview|tl;?dr)\b.*\b(?:this|it)\b", normalized_topic)
            )

            if document_reference and not state.selected_doc_ids:
                logger.info(f"[router] summary intent, no document selected: {topic!r}")
                result = self._casual_result(NO_DOCUMENT_FOR_SUMMARY_MSG)
                result["output_format"] = output_format
                return result

            if document_reference and state.selected_doc_ids:
                logger.info(f"[router] deterministic match -> summarize: {topic!r}")
                return {
                    "is_casual": False,
                    "route_decision": "summarize",
                    "current_stage": "summarizing",
                    "output_format": output_format,
                }

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
            logger.error(f"[router] classifier call failed, defaulting to research: {e}")
            label = "research"

        if "capability" in label:
            logger.info(f"[router] LLM classifier -> capability: {topic!r}")
            return self._casual_result(CAPABILITY_ANSWER)

        if "casual" in label:
            reply = await self._casual_reply(topic)
            return self._casual_result(reply)

        return {
            "is_casual": False,
            "current_stage": "planning",
            "route_decision": "plan",
            "output_format": output_format,
        }

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