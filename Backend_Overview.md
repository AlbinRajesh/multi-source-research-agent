# Backend Overview

This backend is a research pipeline. It takes a user question, plans searches, fetches sources, turns text into checkable claims, verifies those claims, and writes the final answer with citations.

Simple flow:

1. Planner decides what to search.
2. Retriever finds web results and fetches full page text.
3. Credibility filter removes weak sources.
4. Claim extractor turns documents into atomic facts.
5. Verifier checks each fact against source text.
6. If results are weak, it retries search.
7. Synthesizer writes final answer from verified claims only.

## Core files

- [backend/main.py](backend/main.py)  
  Starts the FastAPI app and exposes the research API routes. It also has a streaming route for live progress updates from the workflow.

- [backend/src/config.py](backend/src/config.py)  
  Loads all app settings: model names, API keys, search provider, retry limits, and port.

- [backend/src/state.py](backend/src/state.py)  
  Defines the data model used across the whole workflow: plan, search results, claims, verified claims, report sections, and metrics.

- [backend/src/graph.py](backend/src/graph.py)  
  Builds the LangGraph workflow. This is the main orchestration file: plan -> search -> extract claims -> verify -> retry/check -> synthesize.

- [backend/src/exceptions.py](backend/src/exceptions.py)  
  Custom error classes for planning, search, extraction, verification, LLM, and circuit failures.

## Agent files

- [backend/src/agents/planner.py](backend/src/agents/planner.py)  
  Turns the user topic into a research plan and search queries.

- [backend/src/agents/retriever.py](backend/src/agents/retriever.py)  
  Runs the planned searches, fetches content, removes duplicates, and filters weak sources.

- [backend/src/agents/claim_extractor.py](backend/src/agents/claim_extractor.py)  
  Splits source text into small factual claims that are easy to verify individually.

- [backend/src/agents/verifier.py](backend/src/agents/verifier.py)  
  Checks whether each claim is actually supported by the source. This is the key quality step.

- [backend/src/agents/synthesizer.py](backend/src/agents/synthesizer.py)  
  Builds the final answer using only verified claims and adds citations.

## Search and source handling

- [backend/src/search_providers/base.py](backend/src/search_providers/base.py)  
  Defines the common interface every search provider must follow.

- [backend/src/search_providers/tavily_provider.py](backend/src/search_providers/tavily_provider.py)  
  Calls Tavily to get search results.

- [backend/src/search_providers/searxng_provider.py](backend/src/search_providers/searxng_provider.py)  
  Fallback search provider for SearXNG.

- [backend/src/search_providers/content_extractor.py](backend/src/search_providers/content_extractor.py)  
  Opens each result URL and extracts the real article text from HTML.

## Processing helpers

- [backend/src/processing/dedup.py](backend/src/processing/dedup.py)  
  Removes exact duplicates and near-duplicate sources.

- [backend/src/processing/credibility.py](backend/src/processing/credibility.py)  
  Gives a quick trust score to a source based on domain rules; this is only a pre-filter.

- [backend/src/processing/citations.py](backend/src/processing/citations.py)  
  Formats the final citation list for the answer.

- [backend/src/processing/fetch.py](backend/src/processing/fetch.py)  
  Thin wrapper exposing the content extractor under a processing name.

## Prompt files

These are not logic by themselves; they are the instructions sent to the LLM.

- [backend/src/prompts/planner_prompt.py](backend/src/prompts/planner_prompt.py)  
  Tells the planner how to create search objectives and queries.

- [backend/src/prompts/retriever_prompt.py](backend/src/prompts/retriever_prompt.py)  
  Optional prompt for a more adaptive retriever design.

- [backend/src/prompts/claim_extraction_prompt.py](backend/src/prompts/claim_extraction_prompt.py)  
  Tells the model how to convert text into atomic factual claims.

- [backend/src/prompts/verification_prompt.py](backend/src/prompts/verification_prompt.py)  
  Tells the model to judge whether each claim is truly supported by a source.

- [backend/src/prompts/synthesis_prompt.py](backend/src/prompts/synthesis_prompt.py)  
  Tells the model how to write the final answer using only verified facts and citations.

## Utility files

- [backend/src/utils/http_client.py](backend/src/utils/http_client.py)  
  Shared HTTP client with connection pooling and a circuit breaker for reliability.

- [backend/src/utils/llm_factory.py](backend/src/utils/llm_factory.py)  
  Creates the correct LLM client based on settings like Ollama, OpenAI, Gemini, or Groq.

- [backend/src/utils/cache.py](backend/src/utils/cache.py)  
  Placeholder for future caching logic.

- [backend/src/utils/llm_tracker.py](backend/src/utils/llm_tracker.py)  
  Placeholder for tracking LLM usage and token costs.

## Empty package markers

These are mostly empty and just make Python treat folders as packages.

- [backend/src/__init__.py](backend/src/__init__.py)
- [backend/src/agents/__init__.py](backend/src/agents/__init__.py)
- [backend/src/eval/__init__.py](backend/src/eval/__init__.py)
- [backend/src/processing/__init__.py](backend/src/processing/__init__.py)
- [backend/src/prompts/__init__.py](backend/src/prompts/__init__.py)
- [backend/src/search_providers/__init__.py](backend/src/search_providers/__init__.py)
- [backend/src/utils/__init__.py](backend/src/utils/__init__.py)
- [backend/src/eval/run_eval.py](backend/src/eval/run_eval.py)  
  Empty evaluation script placeholder.

## Tests

- [backend/tests/test_planner.py](backend/tests/test_planner.py)  
  Checks planning logic.

- [backend/tests/test_retriever_accumulation.py](backend/tests/test_retriever_accumulation.py)  
  Checks result accumulation and search behavior.

- [backend/tests/test_dedup.py](backend/tests/test_dedup.py)  
  Checks duplicate removal.

- [backend/tests/test_verifier.py](backend/tests/test_verifier.py)  
  Checks verification logic.

- [backend/tests/test_verifier_conflicting.py](backend/tests/test_verifier_conflicting.py)  
  Checks conflict handling.

- [backend/tests/test_check_retry.py](backend/tests/test_check_retry.py)  
  Checks retry loop logic.

- [backend/tests/test_circuit_isolation.py](backend/tests/test_circuit_isolation.py)  
  Checks circuit breaker behavior.

- [backend/tests/test_e2e_smoke.py](backend/tests/test_e2e_smoke.py)  
  Basic end-to-end smoke test.

- [backend/tests/test_graph_e2e.py](backend/tests/test_graph_e2e.py)  
  End-to-end graph workflow test.

## In one sentence

The backend is a LangGraph research engine: plan -> fetch -> extract facts -> verify them -> write final answer with citations, and retry when evidence is weak.
