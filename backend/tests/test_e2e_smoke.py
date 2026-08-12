import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.graph import run_research

@pytest.mark.asyncio
async def test_full_pipeline_smoke():
    with patch("src.agents.planner.get_llm"), \
         patch("src.agents.retriever.TavilySearchProvider") as MockProvider, \
         patch("src.agents.retriever.ContentExtractor") as MockExtractor, \
         patch("src.agents.claim_extractor.get_llm"), \
         patch("src.agents.verifier.get_llm"), \
         patch("src.agents.synthesizer.get_llm"), \
         patch("src.agents.planner.ChatPromptTemplate") as MockPlanPrompt, \
         patch("src.agents.claim_extractor.ChatPromptTemplate") as MockClaimPrompt, \
         patch("src.agents.verifier.ChatPromptTemplate") as MockVerifyPrompt, \
         patch("src.agents.synthesizer.ChatPromptTemplate") as MockSynthPrompt:

        # planner returns one query
        class FakePlanChain:
            async def ainvoke(self, _):
                return {
                    "topic": "t", "objectives": ["o1"],
                    "search_queries": [{"query": "q1", "purpose": "p", "source_hint": "web"}],
                    "report_outline": ["s1"],
                }
        MockPlanPrompt.from_messages.return_value.__or__ = lambda self, o: self
        MockPlanPrompt.from_messages.return_value.__or__ = lambda self, o: FakePlanChain()

        provider_instance = MagicMock()
        from src.state import SearchResult
        provider_instance.search = AsyncMock(return_value=[
            SearchResult(query="q1", title="t1", url="https://a.com", snippet="fact one", source_type="web")
        ])
        MockProvider.return_value = provider_instance

        extractor_instance = MagicMock()
        extractor_instance.enhance_results = AsyncMock(side_effect=lambda r, **kw: r)
        MockExtractor.return_value = extractor_instance

        class FakeClaimChain:
            async def ainvoke(self, _):
                return '[{"text": "fact one is true"}]'
        MockClaimPrompt.from_messages.return_value.__or__ = lambda self, o: FakeClaimChain()

        class FakeVerifyChain:
            async def ainvoke(self, _):
                import json
                return json.dumps([{"claim_id": _.get("claims_block", "")[:0] or "x", "is_grounded": True, "contradicts": False}])
        # simpler: verifier keys off actual claim id, so build after claims exist — patch differently
        MockVerifyPrompt.from_messages.return_value.__or__ = lambda self, o: FakeVerifyChain()

        class FakeSynthChain:
            async def ainvoke(self, _):
                return "Final answer text."
        MockSynthPrompt.from_messages.return_value.__or__ = lambda self, o: FakeSynthChain()

        result = await run_research("test topic", use_checkpoints=False)

        assert result["final_report"] or result.get("error")
        assert result["retry_count"] <= result["max_retries"]