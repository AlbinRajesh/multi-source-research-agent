import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from src.agents.verifier import VerificationAgent
from src.state import ResearchState, Claim, SearchResult

@pytest.mark.asyncio
async def test_conflicting_tier_assigned():
    claim = Claim(id="c1", text="X happened in 2020", source_url="https://a.com", source_index=0)
    doc = SearchResult(query="q", title="t", url="https://a.com", snippet="X happened in 2019", source_type="web")
    state = ResearchState(research_topic="t", claims=[claim], search_results=[doc])

    fake_llm_output = json.dumps([
        {"claim_id": "c1", "is_grounded": False, "contradicts": True, "reasoning": "source says 2019 not 2020"}
    ])

    agent = VerificationAgent(llm=MagicMock())
    # patch the chain invocation directly
    agent.verify = VerificationAgent.verify.__get__(agent)

    class FakeChain:
        async def ainvoke(self, _):
            return fake_llm_output

    import src.agents.verifier as verifier_module
    original_prompt_or = None

    # simplest: monkeypatch the chain build inside verify by patching ChatPromptTemplate pipeline
    from unittest.mock import patch
    with patch("src.agents.verifier.ChatPromptTemplate") as MockPrompt:
        mock_chain = FakeChain()
        MockPrompt.from_messages.return_value.__or__ = lambda self, other: MockPrompt.from_messages.return_value
        # easier: directly patch the | operator chain result
        class FakePipeline:
            def __or__(self, other):
                return self
            async def ainvoke(self, inputs):
                return fake_llm_output
        MockPrompt.from_messages.return_value = FakePipeline()

        result = await agent.verify(state)

    assert result["verified_claims"][0].confidence == "conflicting"