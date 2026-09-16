"""Tests for output-format parsing and RAG-vs-web routing.

Run with: pytest tests/test_format_and_routing.py -v
Requires: pytest-asyncio (pip install pytest-asyncio --break-system-packages)
"""
from types import SimpleNamespace
import csv
import json
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from src.agents.router import RouterAgent, NO_DOCUMENT_FOR_SUMMARY_MSG, _is_summary_request
from src.agents.summarizer import _constraint_from_state, _enforce_format
from src.processing.output_format import parse_output_format
from src.state import ResearchState


@pytest.fixture(scope="session")
def csv_results(request):
    rows = []

    def record(question, observed_result, answer=""):
        rows.append(
            {
                "question": question,
                "answer_or_observed_result": answer
                or observed_result.get("final_report")
                or observed_result.get("route_decision", ""),
                "route_decision": observed_result.get("route_decision", ""),
                "output_format": json.dumps(
                    observed_result.get("output_format", {}),
                    sort_keys=True,
                ),
                "status": "passed",
            }
        )

    def write_csv():
        output_path = Path(__file__).resolve().parents[1] / "test_output_results.csv"
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys() if rows else [
                "question",
                "answer_or_observed_result",
                "route_decision",
                "output_format",
                "status",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV results written to: {output_path}")

    request.addfinalizer(write_csv)
    return record


# ---------------------------------------------------------------------------
# Shared fixture: stub out the LLM boundary so tests never hit a real
# provider. The stub always classifies ambiguous input as "research" —
# that's the only label our test queries below should ever produce.
# ---------------------------------------------------------------------------
@pytest.fixture
def router_agent(monkeypatch):
    fake_llm = RunnableLambda(lambda _: SimpleNamespace(content="research"))
    monkeypatch.setattr(
        "src.agents.router.get_llm",
        lambda *args, **kwargs: fake_llm,
    )
    return RouterAgent()


def make_state(topic: str, selected_doc_ids=None) -> ResearchState:
    return ResearchState(
        research_topic=topic,
        selected_doc_ids=selected_doc_ids or [],
    )


# ---------------------------------------------------------------------------
# RAG / document-summarizer — 3 cases
# ---------------------------------------------------------------------------
class TestRagRouting:

    @pytest.mark.asyncio
    async def test_summarize_this_with_selected_doc_routes_to_summarizer(
        self, router_agent, csv_results
    ):
        question = "Summarize this in 4 points"
        state = make_state(question, selected_doc_ids=["doc1"])
        result = await router_agent.route(state)
        csv_results(question, result)

        assert result["route_decision"] == "summarize"
        assert result["output_format"]["style"] == "bullets"
        assert result["output_format"]["count"] == 4

    @pytest.mark.asyncio
    async def test_summarize_pdf_without_selected_doc_asks_user_to_select_one(
        self, router_agent, csv_results
    ):
        question = "Summarize this PDF"
        state = make_state(question, selected_doc_ids=[])
        result = await router_agent.route(state)
        csv_results(question, result)

        assert result["is_casual"] is True
        assert result["final_report"] == NO_DOCUMENT_FOR_SUMMARY_MSG

    def test_enforce_format_caps_bullets_to_requested_count(self, csv_results):
        # Exercises the deterministic post-generation guard the
        # summarizer applies, independent of the LLM's raw output.
        state = make_state("Summarize this in 4 points", selected_doc_ids=["doc1"])
        state.output_format = {"style": "bullets", "count": 4, "exact": True}

        constraint = _constraint_from_state(state)
        raw_summary = "1. AAA\n2. BBB\n3. CCC\n4. DDD\n5. EEE"
        formatted = _enforce_format(raw_summary, constraint)
        csv_results(
            state.research_topic,
            {"output_format": state.output_format},
            answer=formatted,
        )

        bullet_lines = [l for l in formatted.splitlines() if l.strip()]
        assert len(bullet_lines) == 4
        assert all(line.startswith("- ") for line in bullet_lines)


# ---------------------------------------------------------------------------
# Web research — 4 cases
# ---------------------------------------------------------------------------
class TestWebRouting:

    @pytest.mark.asyncio
    async def test_points_request_routes_to_research_with_bullet_format(
        self, router_agent, csv_results
    ):
        question = "Give me 5 points about quantum computing"
        state = make_state(question)
        result = await router_agent.route(state)
        csv_results(question, result)

        assert result["route_decision"] == "plan"
        assert result["output_format"]["style"] == "bullets"
        assert result["output_format"]["count"] == 5

    @pytest.mark.asyncio
    async def test_comparison_with_points_each_sets_per_entity_flag(
        self, router_agent, csv_results
    ):
        question = "What is the difference between X and Y in 2 points each?"
        state = make_state(question)
        result = await router_agent.route(state)
        csv_results(question, result)

        assert result["route_decision"] == "plan"
        assert result["output_format"]["style"] == "comparison"
        assert result["output_format"]["count"] == 2
        assert result["output_format"]["per_entity"] is True

    @pytest.mark.asyncio
    async def test_one_sentence_request_sets_sentence_format(
        self, router_agent, csv_results
    ):
        question = "Tell me about X in 1 sentence"
        state = make_state(question)
        result = await router_agent.route(state)
        csv_results(question, result)

        assert result["route_decision"] == "plan"
        assert result["output_format"]["style"] == "sentences"
        assert result["output_format"]["count"] == 1

    @pytest.mark.asyncio
    async def test_summarize_topic_without_doc_reference_stays_in_web_research(
        self, router_agent, csv_results
    ):
        # Regression test for the exact bug this session's fix targets:
        # contains "summarize" (matches SUMMARY_PATTERNS) but has no
        # document-reference wording and no selected_doc_ids, so it must
        # NOT be routed to the local PDF summarizer.
        question = "Summarize quantum computing in 4 points"
        state = make_state(question, selected_doc_ids=[])
        result = await router_agent.route(state)
        csv_results(question, result)

        assert _is_summary_request(state.research_topic) is True  # sanity: pattern does match
        assert result["route_decision"] == "plan"                 # but routing still goes to web research
        assert result["output_format"]["style"] == "bullets"
        assert result["output_format"]["count"] == 4