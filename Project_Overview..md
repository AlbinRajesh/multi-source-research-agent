# Research & Search Agent — Project Plan

**Type:** Enterprise-grade agentic research assistant
**Orchestration:** LangGraph
**Goal:** Plan searches → retrieve from multiple sources → dedupe → verify evidence → cite → flag uncertainty

---

## 1. High-Level Flow

```
                         ┌─────────────────┐
                         │   User Query     │
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │  Planner Agent   │  decomposes query,
                         │                  │  picks sources/strategy
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Search/Retriever │  fans out sub-queries
                         │      Agent       │  across sources (parallel)
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │  Fetch/Extract   │  (plain code)
                         │   full content   │
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │      Dedup       │  (plain code)
                         │  exact + near-dup │
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │  Credibility     │  (plain code, heuristic)
                         │  Pre-filter      │
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Claim Extraction │  breaks docs into
                         │      Agent       │  atomic checkable claims
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │  Verification    │  checks each claim vs
                         │      Agent       │  source, cross-corroborates
                         └────────┬─────────┘
                                  │
                     ┌────────────┴────────────┐
                     │  low confidence?          │
                     │  retry < max_retries?     │──Yes──▶ back to Search
                     └────────────┬────────────┘
                                  │ No
                                  ▼
                         ┌─────────────────┐
                         │  Synthesizer/    │  builds answer bottom-up
                         │  Writer Agent    │  from verified claims only
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Citation +       │  (plain code formatting)
                         │ Confidence Tags  │
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │  Final Answer    │
                         │  + Sources +     │
                         │  Confidence      │
                         └─────────────────┘
```

---

## 2. Phases Overview

| Phase | Goal | Status Gate to Move On |
|---|---|---|
| **Phase 1** | Core pipeline, web search only | Accuracy proven on eval set |
| **Phase 2** | Add local RAG (your vlmproject retriever) as a second source | Accuracy re-tested with both sources active |
| **Phase 3** | Wrap finished pipeline as an MCP server | End-to-end MCP client test passes |

---

## 3. Phase 1 — Core Research Pipeline (Web Only)

### 3.1 Steps

1. **Define state schema** — the shared data structure flowing through every node
2. **Build Planner Agent** — query decomposition + sub-query generation
3. **Build Search/Retriever Agent** — calls web search API, decides depth
4. **Build Fetch/Extract step** — pull full page content (plain code)
5. **Build Dedup step** — exact + near-duplicate removal (plain code)
6. **Build Credibility Pre-filter** — cheap heuristic scoring (plain code)
7. **Build Claim Extraction Agent** — atomic claims from surviving docs
8. **Build Verification Agent** — groundedness check + cross-source corroboration
9. **Build Synthesizer/Writer Agent** — bottom-up answer from verified claims only
10. **Build Citation + Confidence formatting** (plain code)
11. **Wire conditional retry loop** — re-search if too many low-confidence claims
12. **Add checkpointing** — crash recovery via SQLite
13. **Build eval harness** — golden Q&A test set, measure accuracy before Phase 2

### 3.2 Agents & Tools — What Each Does

| Component | Type | Job | Tool/Tech |
|---|---|---|---|
| Planner Agent | LLM agent | Breaks query into 3–5 sub-queries, decides search depth/strategy | Ollama (qwen3:8b) or hosted LLM |
| Search/Retriever Agent | LLM agent | Decides which queries to run, how many results, whether to dig deeper | Tavily (primary), SearXNG/Brave (fallback) |
| Fetch/Extract | Plain code | Pulls full page content, not just snippets | httpx (async, HTTP/2, connection pooling) + trafilatura/readability |
| Dedup | Plain code | Removes exact duplicates (URL hash) and near-duplicates (embedding similarity) | BAAI embeddings (reuse from vlmproject) or lighter embedding model |
| Credibility Pre-filter | Plain code | Cheap heuristic filter before expensive verification (domain scoring: `.gov`/`.edu` boost, suspicious TLD penalty) | Rule-based scoring function |
| Claim Extraction Agent | LLM agent | Converts paragraphs into atomic, checkable factual claims | Fast/cheap LLM (Ollama local) |
| Verification Agent | LLM agent | Checks each claim against source text; corroborates across sources; assigns confidence tier | Same groundedness-verifier pattern from vlmproject |
| Synthesizer/Writer Agent | LLM agent | Builds final answer using only verified claims, attaches citations | Strongest available LLM (hosted API or largest local model) |
| Citation Formatter | Plain code | Formats citations (APA/MLA/etc.), attaches confidence tags | Templating function |
| Checkpointer | Infra | Saves pipeline state so long research jobs can resume after failure | LangGraph `SqliteSaver` |
| Circuit Breaker | Infra | Fails fast on a dead/slow source instead of hanging | Custom (CLOSED → OPEN → HALF_OPEN state machine) |
| Cache | Infra | Avoids re-searching identical/near-identical queries | File-based, TTL (e.g. 7 days), MD5 topic hash |
| Observability | Infra | Traces every node call across the graph for debugging/audit | LangSmith |
| Eval Harness | Testing | Measures retrieval + verification accuracy against a golden test set | Reuse your `test_retrieval_accuracy.py`-style framework |

### 3.3 File Structure — Phase 1

```
research-agent/
├── src/
│   ├── __init__.py
│   ├── config.py                  # Pydantic settings: model provider, API keys, thresholds
│   ├── state.py                   # ResearchState TypedDict schema
│   ├── graph.py                   # LangGraph StateGraph definition + compile + checkpointing
│   ├── exceptions.py              # Typed exceptions (PlanningError, SearchError, etc.)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py             # Planner Agent node
│   │   ├── retriever.py           # Search/Retriever Agent node
│   │   ├── claim_extractor.py     # Claim Extraction Agent node
│   │   ├── verifier.py            # Verification Agent node
│   │   └── synthesizer.py         # Synthesizer/Writer Agent node
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── planner_prompt.py
│   │   ├── retriever_prompt.py
│   │   ├── claim_extraction_prompt.py
│   │   ├── verification_prompt.py
│   │   └── synthesis_prompt.py
│   │
│   ├── processing/                # plain-code steps, no LLM
│   │   ├── __init__.py
│   │   ├── fetch.py                # full-page fetch + extraction
│   │   ├── dedup.py                 # exact + near-dup removal
│   │   ├── credibility.py           # heuristic domain scoring
│   │   └── citations.py             # citation formatting
│   │
│   ├── search_providers/
│   │   ├── __init__.py
│   │   ├── base.py                  # SearchProvider interface
│   │   ├── tavily_provider.py
│   │   └── searxng_provider.py      # fallback
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── http_client.py           # httpx client, circuit breaker
│   │   ├── cache.py                 # TTL file-based cache
│   │   └── llm_tracker.py           # token/cost tracking
│   │
│   └── eval/
│       ├── __init__.py
│       ├── golden_questions.json    # test question set
│       └── run_eval.py              # accuracy/latency measurement script
│
├── tests/
│   ├── test_planner.py
│   ├── test_dedup.py
│   ├── test_verifier.py
│   └── test_graph_e2e.py
│
├── .cache/
│   ├── research/                    # cached results
│   └── checkpoints/                 # SQLite checkpoints
│
├── .env.example
├── main.py                          # CLI entry point
├── requirements.txt
└── README.md
```

### 3.4 State Schema

```python
class ResearchState(TypedDict):
    query: str
    sub_queries: list[str]
    sources_to_use: list[str]        # e.g. ["web"] in Phase 1
    raw_results: list[dict]
    fetched_docs: list[Document]
    deduped_docs: list[Document]
    claims: list[Claim]              # {text, source_id}
    verified_claims: list[Claim]     # + verdict, confidence tier
    retry_count: int
    final_answer: str
    citations: list[dict]
    total_input_tokens: int
    total_output_tokens: int
```

### 3.5 Working — How It Runs

1. User submits a query.
2. **Planner** breaks it into sub-queries and decides search depth.
3. **Search/Retriever** fans out sub-queries in parallel to the web search provider(s), with a circuit breaker per provider so one dead source doesn't stall the run.
4. **Fetch/Extract** pulls full content for the top results (not just snippets).
5. **Dedup** removes exact duplicate URLs and near-duplicate content (embedding similarity).
6. **Credibility Pre-filter** drops obviously low-quality sources before they reach the expensive LLM stages.
7. **Claim Extraction** turns surviving documents into atomic, checkable claims, each tied to its source.
8. **Verification** checks each claim against its source text, marks it verified / single-source / conflicting, and corroborates across multiple sources where possible.
9. **Conditional check:** if too many claims are low-confidence and retries remain, loop back to Search with a refined plan. Otherwise proceed.
10. **Synthesizer** builds the final answer using only verified claims — bottom-up, not generate-then-verify.
11. **Citation Formatter** attaches sources and confidence tags to the final answer.
12. Result returned to the user, with the full run checkpointed for crash recovery/audit.

### 3.6 Eval Gate (before Phase 2)

Build a golden question set (mirroring your vlmproject 192-question suite approach) covering:
- Simple factual lookups
- Multi-source comparison questions
- Questions with genuinely conflicting sources
- Questions where the correct answer is "insufficient evidence"

Measure: retrieval hit rate, claim-level precision/recall against known-correct verdicts, latency per query, cost per query. Don't proceed to Phase 2 until this is stable.

---

## 4. Phase 2 — Add Local RAG as a Second Source

### 4.1 Steps

1. Define shared `Document` schema (same shape for web and local results)
2. Wrap your existing vlmproject retriever as a `local_rag_search()` adapter returning that shape
3. Extend Planner Agent to choose sources per query: `web`, `local`, or `both`
4. Extend Search node to fan out to whichever sources the plan selected
5. Extend Citation Formatter to branch by `source_type` (web → domain+URL, local → filename+section)
6. Extend Verification Agent's confidence weighting: still verify local claims, but weight "grounded in user's own document" as inherently higher-trust than an unverified web source once it passes the check
7. Re-run the eval harness with local+web active, confirm no regression

### 4.2 Tools & Job

| Component | Job | Tool |
|---|---|---|
| `local_rag_search()` adapter | Wraps vlmproject retriever, returns `Document`-shaped results | Existing hybrid BM25 + Qdrant + reranker pipeline |
| Planner source routing | Decides web vs local vs both per query | LLM planner prompt extension |
| Citation branch | Formats local doc citations differently from web citations | Extended `citations.py` |

### 4.3 File Structure Additions

```
research-agent/
├── src/
│   ├── processing/
│   │   └── document_schema.py       # shared Document TypedDict (web + local)
│   ├── search_providers/
│   │   └── local_rag_provider.py    # wraps vlmproject retriever
```

### 4.4 Working

Planner now also considers whether the query implies "my document" / "uploaded file" intent. If so, it includes `local` in `sources_to_use`. Search node calls both adapters in parallel when both are selected. Everything downstream (dedup, claim extraction, verification, synthesis) is source-agnostic — it just sees `Document` objects, so no redesign is needed, only the two adapter/routing additions above.

---

## 5. Phase 3 — MCP Server Wrapper

### 5.1 Steps

1. Confirm Phase 1 + Phase 2 pipeline is stable and passing eval
2. Write a thin MCP server exposing one tool: `conduct_research(query: str)`
3. Test with an MCP client (Claude Desktop / Claude Code) calling the tool end-to-end
4. Document the tool's input/output contract for other teams/agents to consume

### 5.2 File Structure Addition

```
research-agent/
├── mcp_server/
│   ├── __init__.py
│   └── server.py                    # MCP server, exposes conduct_research tool
```

### 5.3 Example

```python
# mcp_server/server.py
from mcp.server import Server
from src.graph import run_research

server = Server("company-research-agent")

@server.tool()
async def conduct_research(query: str) -> dict:
    """Research a topic using verified, cited sources (web + local documents)."""
    result = await run_research(query)
    return {
        "answer": result["final_answer"],
        "citations": result["citations"],
        "confidence": result["confidence_tiers"],
    }
```

### 5.4 Working

Any MCP-compatible client can now call `conduct_research` without knowing anything about LangGraph, your nodes, or your verification logic — it just sends a query and gets back a cited, confidence-tagged answer. Internal pipeline stays a direct, fast, in-process LangGraph graph; MCP is purely the external integration surface, added last, on top of a proven engine.

---

## 6. Reference Projects (for implementation patterns)

| Project | What to borrow |
|---|---|
| [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) | Retriever abstraction, context dedup approach, scraper failure handling |
| [tarun7r/deep-research-agent](https://github.com/tarun7r/deep-research-agent) | LangGraph node structure, checkpointing, circuit breaker, credibility scoring |
| [LearningCircuit/local-deep-research](https://github.com/LearningCircuit/local-deep-research) | Adaptive source selection strategy, local knowledge base ("bring your own retriever") integration pattern, MCP server pattern |

None of these do claim-level groundedness verification as rigorously as the vlmproject-style verifier — that remains this project's core differentiator.


-----------------------------------------------------------------------------------------------------

Yes — specific files would speed this up a lot, since we can adapt proven patterns instead of writing everything from scratch. I don't need everything, just the pieces that map directly to our Phase 1 nodes. Here's what's actually useful, from which repo:

From tarun7r/deep-research-agent (most directly reusable — same LangGraph structure)
File	Why useful
src/state.py	See their actual ResearchState fields — compare against ours, borrow anything we missed
src/graph.py	Their StateGraph wiring + checkpointing setup — this is close to exactly what we're building
src/agents.py	How they structure agent nodes with dependency injection (their "testable agent architecture" claim)
src/utils/web_utils.py	Circuit breaker implementation + search provider abstraction — don't reinvent this
src/utils/credibility.py	Their domain-scoring logic — we can extend this into our credibility pre-filter step
src/exceptions.py	Their typed exception hierarchy — good structure to mirror
src/prompts/*.py	Their actual planner/searcher/synthesizer prompts — useful as a starting draft even though we'll rewrite for our verification-heavy flow
From assafelovic/gpt-researcher (bigger, more mature — mine for specific patterns only)
File/area	Why useful
Their dedup/context-handling code	You referenced their "context deduplication" fixes — the actual function is worth seeing
Their retriever abstraction (multi_agents/ or actions/retriever.py, depending on version)	How they support multiple pluggable search providers cleanly
Their local document handling (DOC_PATH support)	Directly relevant for our Phase 2 local RAG integration later
From local-deep-research
File/area	Why useful
Their LangGraph adaptive-agent strategy code	This is the piece that got them the 95% SimpleQA score — worth seeing how they let the planner choose search engines dynamically
Their "bring your own retriever" integration (quick_summary(retrievers=...))	Directly informs our Phase 2 local RAG adapter design
What I'll do with them

Not copy-paste — extract the pattern (state shape, node wiring, error handling, credibility scoring formula) and rewrite it to fit our specific 5-agent design with the verification agent as the accuracy backbone, which none of them have. Their license (MIT/Apache-2.0) permits this kind of reuse freely, so no concern there — this is normal, expected open-source practice.

If you can paste in state.py, graph.py, and web_utils.py from tarun7r's repo first — that's the highest-value set to start Phase 1 scaffolding immediately.
-----------------------------------------------------------------------------------------------------
Evaluation plan — mirrors your vlmproject 192-question approach, scaled to this pipeline.

1. Build a golden question set (src/eval/golden_questions.json) — 20-30 to start, categories from your plan doc section 3.6:

json
[
  {"id": "q1", "question": "What is the boiling point of water at sea level?", "category": "simple_factual", "expected_answer_contains": ["100", "212"], "min_citations": 1},
  {"id": "q2", "question": "Compare boiling point elevation effects of salt vs sugar in water", "category": "multi_source_comparison", "min_citations": 2},
  {"id": "q3", "question": "What is the exact boiling point of water on Mars' surface today", "category": "insufficient_evidence", "expect_low_confidence": true}
]

2. Metrics to capture per run (src/eval/run_eval.py):

Retrieval hit rate: did search_results end up non-empty and topically relevant (manual/LLM-judge check)
Claim grounding rate: verified_count / total_claims from your verifier logs — you already log this
Citation accuracy: spot-check N citations — does the cited source actually say what the report claims (this is the one metric that catches hallucination even after your groundedness fixes)
Retry rate: how often retry_count > 0 — high retry rate signals weak initial search/verification
Latency: total wall time, broken down per stage (plan/search/extract/verify/synthesize) — same stage-level profiling pattern you used in vlmproject
Cost: token count × Groq pricing, if using hosted

3. Minimal harness:

python
import asyncio, json, time
from src.graph import run_research

async def run_eval():
    questions = json.load(open("src/eval/golden_questions.json"))
    results = []
    for q in questions:
        start = time.time()
        r = await run_research(q["question"], use_checkpoints=False)
        elapsed = time.time() - start
        verified = sum(1 for v in r.get("verified_claims", []) if v.confidence in ("verified", "single_source"))
        total = len(r.get("verified_claims", [])) or 1
        results.append({
            "id": q["id"], "latency_s": elapsed,
            "grounding_rate": verified / total,
            "citation_count": len(r.get("citations", [])),
            "retry_count": r.get("retry_count", 0),
            "final_report": r.get("final_report"),
        })
    json.dump(results, open("eval_results.json", "w"), indent=2)
    avg_latency = sum(x["latency_s"] for x in results) / len(results)
    avg_grounding = sum(x["grounding_rate"] for x in results) / len(results)
    print(f"Avg latency: {avg_latency:.1f}s | Avg grounding: {avg_grounding:.0%}")

asyncio.run(run_eval())

On latency specifically — since you flagged it: add stage-level timing (same pattern as your test_latency_profile.py in vlmproject) by wrapping each node in graph.py with a timer, storing stage_timings: Dict[str, float] in state. That'll tell you whether latency is search-bound (Tavily+fetch), LLM-bound (Groq calls), or retry-bound (repeated full passes) — you can't fix what you haven't measured per-stage.

On the 429s — the assistant's tip is right; also consider adding a global rate limiter (not just per-agent semaphore) since Planner/Extractor/Verifier/Synthesizer all hit Groq independently and their semaphores don't coordinate with each other.

Want the stage-level timing instrumentation code, or the retry/lower-concurrency fix first?
-----------------------------------------------------------------------------------------------------
Here is a simple, clear breakdown of how your system is working right now and what we did to speed it up.

Part 1: How Your Research Agent Works Right Now
Think of your agent as a team of specialized AI workers operating in a assembly line:

The Planner (plan): Takes your topic ("What is the boiling point of water") and figures out what sub-questions need to be asked.

The Retriever (search): Uses Tavily to search the web, downloads the pages, cleans up the text, and runs them through a Credibility Filter. Crucially, we capped this to only keep the top 8 best sources so we don't overwhelm the system.

The Claim Extractor (extract_claims): Takes those 8 documents and breaks them down into short, independent factual statements (claims).

The Verifier (verify): Checks every single claim against the source text to see if it's actually true and supported. If a claim is fake or unsupported, it gets thrown out.

The Synthesizer (synthesize): Combines all the verified, true facts into a clean, final report and adds neat, deduplicated citation numbers (like [1]).

If too many claims turn out to be weak, the system automatically triggers a Retry Loop to search again—though in your last run, it passed cleanly on the first try!

Part 2: What We Did to Reduce Time and Fix Errors
When you first ran the agent, it was getting stuck in massive delays, taking over 3 minutes (and sometimes failing with errors). We fixed it step-by-step:

1. Stopped Overwhelming Groq (Fixed 429 Rate Limits)
The Problem: Groq's free tier has a limit of 8,000 tokens per minute (TPM). Your agent was trying to send 20 documents all at once, crashing into the limit. Groq would return 429 Too Many Requests, and the system would sit there waiting 20 seconds before retrying.

The Fix:

We capped the max documents sent to extraction from ~20 down to 8 (using credibility scoring).

We lowered the parallel worker limit (max_concurrent) so the agent makes fewer requests at the exact same time.

2. Shrank the Text Size (Reduced Token Spend)
The Problem: We were sending up to 6,000 characters of raw webpage text for every single AI call. That blew past the token limit instantly.

The Fix: We trimmed the document text sent to the AI down to 1,500 characters. This cut the token cost in half, allowing requests to finish much faster without breaking the AI's JSON output.

3. Fixed JSON Parsing & Verifier Crashes (Bug 17 & 19)
The Problem: Sometimes the smaller AI models would output weird text formatting, unescaped newlines, or cut off entirely when hitting rate limits, causing the Python script to crash with JSON errors.

The Fix:

We added strict=False to allow raw newlines in JSON.

We added an empty-string safety check so if a request fails, it fails gracefully instead of crashing the whole pipeline.

4. Fixed Citation Duplication (Bug 18)
The Problem: The same website (like Purdue Chemistry or Wikipedia) was showing up in the citations list 14 separate times.

The Fix: We added a deduplication step by URL so every unique source gets one single citation number (e.g., [1]), matching the numbers used in the text.

5. Added Stage-Level Timings
The Fix: We added a built-in timer (timed_node) that logs exactly how many seconds each phase (search, extract_claims, verify, etc.) takes, which is how we diagnosed that extract_claims was causing the bottleneck in the first place.