import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.retriever import RetrieverAgent
from src.state import ResearchState, SearchResult, ResearchPlan, SearchQuery

def make_result(url, snippet="some content here"):
    return SearchResult(query="q", title="t", url=url, snippet=snippet, source_type="web")

@pytest.mark.asyncio
async def test_dedup_and_scores_accumulate_across_rounds():
    provider = MagicMock()
    extractor = MagicMock()
    scorer = MagicMock()

    # round 1: returns url A
    provider.search = AsyncMock(return_value=[make_result("https://a.com/x", snippet="some content here")])
    extractor.enhance_results = AsyncMock(side_effect=lambda results, **kw: results)
    scorer.filter_results = MagicMock(side_effect=lambda results, min_score: results)
    scorer.score_url = MagicMock(return_value={"score": 80, "factors": [], "level": "high"})

    agent = RetrieverAgent(search_provider=provider, extractor=extractor, scorer=scorer)
    plan = ResearchPlan(topic="t", objectives=[], search_queries=[SearchQuery(query="q1", purpose="p")], report_outline=[])
    state = ResearchState(research_topic="t", plan=plan)

    out1 = await agent.search(state)
    assert len(out1["search_results"]) == 1
    assert len(out1["credibility_scores"]) == 1

    # round 2: same url A again (simulating retry) + new url B with distinct content
    provider.search = AsyncMock(return_value=[
        make_result("https://a.com/x", snippet="some content here"),
        make_result("https://b.com/y", snippet="a completely different topic entirely"),
    ])
    state2 = state.model_copy(update={"search_results": out1["search_results"], "credibility_scores": out1["credibility_scores"]})

    out2 = await agent.search(state2)
    urls = [r.url for r in out2["search_results"]]
    assert urls.count("https://a.com/x") == 1  # deduped, not doubled
    assert "https://b.com/y" in urls
    assert len(out2["credibility_scores"]) == len(out2["search_results"])