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
---------------------------------------------------------------------------------------------------


Per-node timing (this run)
Stage	First pass	Retry pass	Total
plan	1.76s	—	1.76s
search	35.46s	12.18s	47.64s
extract_claims	47.39s	0.00s	47.39s
verify	31.48s	84.06s	115.54s
synthesize	17.50s	—	17.50s
Total			~230s (3.8 min)

-------------------------------------------
some improvements
-------------------------------------------
Per-node timing (this run)
Stage          First pass   Retry pass   Total
plan           1.58s        —            1.58s
search         23.95s       —            23.95s
extract_claims 59.88s       —            59.88s
verify         36.86s       —            36.86s
synthesize     13.45s       —            13.45s
Total                                    ~135.7s (2.3 min)
-------------------------------------------
Stage          First pass   Retry pass   Total
plan           1.21s        —            1.21s
search         5.49s        5.67s        11.16s
extract_claims 15.72s       14.63s       30.35s
verify         2.19s        5.39s        7.58s
refine_search  0.00s        —            0.00s
synthesize     —            —            21.40s
Total                                    ~71.7s (1.2 min)



Plan: $1.58\text{s}$Search: $23.95\text{s}$Extract Claims: $59.88\text{s}$Verify: $36.86\text{s}$Synthesize: $13.45\text{s}$

verify is now your dominant bottleneck at 115s — not extract_claims. The Ollama move worked exactly as intended for extraction speed, but it exposed a bigger problem underneath.

The real bug: over-extraction, not rate limits

Look at the claim counts: qwen2.5:3b extracted 75 claims from 8 docs (~9 claims/doc). Your original Groq extraction on the same kind of query extracted 9-10 claims total from 8 docs (~1.1/doc). That's a ~7x volume increase.

This cascades badly:

verify now has to check 75 claims instead of ~10 → 7x more Groq calls/tokens → this is why verify is now eating 31s then 84s of retries, even though verify itself wasn't touched
Only 39/75 (52%) passed verification on the first pass → triggered a retry
The retry reprocessed all 75 claims again (not just the failed ones), pushing verify to 84s under heavy 429 backoff
Net result: you moved the bottleneck, you didn't remove it — total latency (230s) is actually worse than your original Elon Musk run (~145s from the very first log you shared)

Root cause: qwen2.5:3b-instruct without tight extraction constraints tends to over-decompose text into many small, sometimes redundant or low-information claims. This isn't a model failure — it's a missing constraint in your prompt/pipeline.

Fix — cap and tighten extraction, don't just swap providers:

Add an explicit instruction to CLAIM_EXTRACTION_SYSTEM_PROMPT: "Extract at most 4-6 of the most significant, non-redundant factual claims. Do not extract minor or trivial details."
Add a hard post-filter in claim_extractor.py after parsing: cap claims per doc, e.g. parsed_claims[:6].
This alone should cut your claim volume ~5x, which cuts verify's token load proportionally — likely your single biggest latency win right now, bigger than the Ollama move itself.
Second bug, more serious: your retry loop does nothing useful

Look closely — on retry, search reran the exact same 5 malformed queries ("Elone Muk" biography, etc. — the typo was never corrected), got the same sources, and extract_claims on retry shows 0.00s — meaning zero new documents were processed, because processed_result_indices already covered everything from pass 1.

So the retry:

Spent 12s re-querying Tavily for identical results
Extracted zero new claims
Then re-verified all 75 old claims again, burning 84s and a fresh wave of 429s
For a net gain of 14 percentage points (39→53 out of 75 grounded) — almost entirely from Groq's non-determinism on identical inputs, not from any new evidence

This matches something already flagged in your project's earlier static-analysis phase: the retry loop routes back to search instead of plan, so it never generates refined queries — it just reruns the same failed strategy and hopes for a different result. This is your actual highest-value bug fix, more impactful than either Ollama or extraction limits, because right now every retry is closer to pure waste than genuine research.

Fix: route check_retry's low-confidence branch back to plan, not search, so a new pass can generate different/refined sub-queries (and ideally get a chance to notice and correct "Elone Muk" → "Elon Musk").

Your adaptive sub-query idea — good instinct, here's how to make it real

Yes, "who is elon musk" doesn't need 5 sub-queries fanning out into biography/career/comparison/controversies/2026-news — that's appropriate for a broad research report, wasteful for a simple identity lookup.

Implementation approach — classify query complexity in the planner, don't hardcode it:

python
# rough sketch inside planner.py
def estimate_complexity(topic: str) -> str:
    simple_patterns = [r"^who is\b", r"^what is\b", r"^when (was|did)\b", r"^where is\b"]
    if any(re.match(p, topic.strip().lower()) for p in simple_patterns) and len(topic.split()) <= 6:
        return "simple"
    return "complex"

Then:

simple → 1-2 sub-queries, max_search_results_per_query reduced (e.g. 3 instead of 5), skip the "controversies/comparison/future prospects" objective generation entirely
complex → your current 5-objective flow

Better than a regex heuristic long-term: have the LLM planner call itself output a complexity: "simple" | "moderate" | "complex" field alongside its objectives (one extra field in the same call, zero extra latency), and use that to gate downstream max_search_queries. This scales better than pattern-matching as you add more query types, and it's still one Groq call, not an extra one.

Additional latency/token reduction levers, in priority order
Cap claims per doc (above) — highest ROI, fixes verify's blowup directly
Fix retry → route to plan, not search — stops wasted re-verification cycles
Skip re-verifying already-grounded claims on retry — only re-run verify on the claims that failed, not the full accumulated set. Right now you're paying full verify cost twice for claims that already passed.
Batch verify calls — check if verifier.py already batches multiple claims per source into one call (log shows "Verification failed for source 4", suggesting per-source batching already exists — good). If it's actually per-claim, batching by source is a large win.
Adaptive result count — max_search_results_per_query: 5 is fixed regardless of complexity; tie this to the complexity classifier too.
TokenRateLimiter (your original planned next step) — still valuable for verify/synthesize, which remain on Groq and still hit 429s in this log. This smooths request pacing rather than bursting and immediately eating your 8000 TPM budget.
Edge cases worth handling for an enterprise-grade pipeline
Malformed/misspelled entity names never get corrected — "Elone Muk" persisted through the entire retry cycle. Add an entity-normalization or spell-check step in plan, or have the planner LLM itself flag and correct obvious typos before generating sub-queries.
Retry loop with zero new evidence — should detect "no new documents processed" and either stop retrying (don't waste a cycle) or force query diversification instead of resubmitting identical queries.
Local model unavailable — if Ollama isn't running when claim_extractor calls it, you need a defined fallback (retry on Groq? fail the doc gracefully?) rather than an unhandled connection error taking down the whole run.
VRAM contention — if your embedding model, reranker, and qwen2.5:3b are ever needed concurrently (e.g. if local_rag_enabled Phase 2 comes online), 4GB will be tight. Worth load-testing that combination before Phase 2.
Claim explosion from local model on longer/denser documents — the cap above helps, but also consider truncating input more aggressively for extraction (text[:3000] already exists — maybe bring it down further, extraction doesn't need full-document context, just the most information-dense paragraphs).
Over-triggering retries — with min_claims_verified_ratio: 0.6, a 52% pass rate reasonably triggers retry, but if extraction is capped/tightened per above, expect higher first-pass verification rates, so retries should become rarer and cheaper when they do happen.

My suggested next step, in order: (1) cap claims per doc, (2) fix retry routing to plan, (3) add the complexity-based adaptive sub-query count. Those three together should meaningfully cut both latency and token spend without sacrificing report quality — want me to write the actual code changes for #1 and #2 first, since those are the two clearest bugs?




Claude is AI and can make mistakes. Please double-check responses.
----------------------------------------------------------------------------------------------------
Implementation Plan — Remaining Fixes
1. Retriever-level rate limiting + retry (Tavily)

Problem: Tavily calls have no retry-on-failure and no shared throttle — a single 429/timeout just drops that sub-query's results silently, and concurrent retriever calls aren't coordinated with each other.
Fix: Wrap Tavily calls in a small retry-with-backoff (catch failure, wait, retry, cap at ~3 attempts), plus keep them under the semaphore already added in retriever.py.
Why it helps: Currently a transient Tavily hiccup just silently loses that sub-query's results with no recovery — this makes search resilient instead of fragile, same category of fix as the Groq-side backoff you already benefit from via the SDK.

2. Calibrate min_relevance_score from real data (in progress)

Problem: Two guesses so far — 0.35 (bi-encoder) and 30 (cross-encoder) — both wrong for their respective score scales, one too strict, one impossibly strict.
Fix: Set to 0.0 based on the actual [relevance_dist] you just captured (real facts scored 0.13–8.41, trivia scored -2.83 to -11.41 — clean gap at 0).
Why it helps: This is the one open item directly blocking correct pipeline output right now — everything else is hardening, this is a live bug in your last run (17 → 0 claims survived).

3. Safety floor on relevance filter

Problem: If every claim in a batch scores below threshold (as just happened), the filter can zero out an entire run, forcing a wasted retry cycle or an empty synthesis.
Fix: If the filter would drop below a minimum count (e.g. 3 claims), keep the top-N by score instead of dropping all of them.
Why it helps: Turns "run produces nothing" into "run produces a weaker but real answer" — a graceful degradation instead of a full failure.

4. Consolidate COMPLEXITY_LIMITS into one location

Problem: Currently only defined in graph.py as a stopgap; planner.py may reference its own copy (never confirmed either way).
Fix: Move it to config.py as the single source, import in both graph.py and planner.py.
Why it helps: Prevents the two files silently drifting out of sync if one gets edited later without the other.

Not including, and why — these came from the gpt-researcher review but don't apply to your actual architecture:

Programmatic citation verification — you already have this, more rigorously, via verifier.py's per-claim groundedness + confidence tiers. gpt-researcher was flagging its own gap, not yours.
Durable checkpointing — you already have AsyncSqliteSaver wired in run_research_with_persistence; gpt-researcher doesn't have this at all, so their recommendation doesn't apply to you.
content_extractor.py Tavily batch-limit — that file is yours, not gpt-researcher's (their Tavily extract implementation is architected differently, single-URL calls), so their audit couldn't confirm or deny your specific bug. Still genuinely open on your end — worth checking directly rather than via their codebase.
Groq pricing lookup — gpt-researcher doesn't use Groq, so their cost file has nothing to import. Your placeholder is still open, separate task.

Want me to start with #1 (retry/backoff) or #2 (apply the 0.0 threshold + rerun), since #2 is a one-line change and currently blocking correct output?
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Final Fix Plan — Ordered
Priority 1 — Live bugs blocking correct output

1. Set min_relevance_score to 0.0 (calibrated fix)

Real facts scored 0.13–8.41, trivia scored -2.83 to -11.41 — clean gap at 0
One-line change, currently causing correct answers (e.g. CEO name) to be dropped
Rerun simple-mode CEO query after to confirm fix

2. Safety floor on relevance filter

If filtering would drop below min count (e.g. 3 claims), keep top-N by score instead of zeroing out
Prevents a full run failure when every claim scores low

3. Fix claim_extractor.py list/dict parsing crash

'list' object has no attribute 'get' — still present in your last 3 logs, fix discussed earlier was never applied
python
if isinstance(data, list):
    if data and all(isinstance(item, list) for item in data):
        data = [item for sub in data for item in sub]
    return [item for item in data if isinstance(item, dict) and "text" in item]

4. Fix broken citation URL (localhost:5173 fallback)

Citation [1] in simple-mode run linked to the app's own base URL instead of the real article
Check citation object construction / frontend href mapping for a missing doc.url defaulting silently
Priority 2 — Synthesis/prompt fixes

5. Apply the synthesis_prompt.py fix (already written, confirm it's deployed)

Removes hallucinated ## References section from structured mode
Confirmed working in one test, but re-verify it's actually in the running code given #3 recurred

6. Planner query specificity for topical questions

"Main AI products in 2026" pulled an unrelated shoe-company story
Tighten planner prompt to anchor queries to concrete product/launch terms when topic is product-specific

7. Planner temporal decomposition (not yet implemented)

Evolution/timeline questions ("since 2020") still collapse to recent-only coverage
Add rule: split date ranges into sub-period queries, bump complexity tier for temporal questions
Priority 3 — Resilience/hardening

8. Tavily retry-with-backoff

No retry on Tavily failure currently — a 429/timeout silently drops that sub-query
Wrap in retry (~3 attempts, backoff), keep under existing semaphore in retriever.py

9. Groq TokenRateLimiter integration

429s costing 10–15s per run across verify/synthesize stages
This was already on your roadmap — now has clear evidence it's actively slowing every run

10. Consolidate COMPLEXITY_LIMITS

Currently possibly duplicated between graph.py and planner.py
Move to config.py as single source, import in both
Priority 4 — Frontend

11. Confirm Tailwind is applying to CitationList.jsx

"Single sourceweb" spacing bug still present after the markdown fix (which only touched AnswerDisplay.jsx)
Run the red-border test to confirm Tailwind is actually hitting this file before assuming it's a code bug
Priority 5 — Performance (no correctness impact, do last)

12. Speed up Extract Claims stage

~48s avg, worst bottleneck in every run
Raise max_concurrent above 2, or restrict extraction more aggressively to only newly-fetched docs

13. Reranker double-load fix

Cross-encoder reloading/re-instantiating on retry passes — redundant HF cache roundtrips on a 4GB card
Still open, not yet root-caused
content_extractor.py Tavily batch-limit — worth checking directly since it's your own file, unconfirmed either way
Groq pricing lookup placeholder — separate, low-urgency task

Start with #1–#4 today — those are the ones actively producing wrong or broken answers. Send me retriever.py or the relevance filter code when ready for #1/#2.


---------------------------------------------------------------------------------------------------
Handoff Summary — multi_source_researcher debugging session
Project context

Backend: FastAPI + LangGraph pipeline (Plan → Search → Extract → Verify → Synthesize). Frontend: React/Vite + Tailwind v4. LLM stack: Groq (openai/gpt-oss-20b) for planner/verifier/synthesizer, Ollama (qwen2.5:3b-instruct) for claim extraction. Search via Tavily.

Where we left off

Just finished wiring a custom TokenRateLimiter to fix Groq free-tier TPM throttling (429s costing 10–15s/run). The limiter class and tracker integration are written and reviewed correct — but the actual call-site edits (provider= kwarg) have not yet been confirmed as applied to the live repo. That's the immediate next step.

Fixes locked in during this session (confirm all are actually applied in your repo — several were "written but not yet pasted in" at various points)
claim_extractor.py — _parse_json_array: flatten nested lists before validating, instead of returning raw data unchecked (was causing 'list' object has no attribute 'get' crashes, silently dropping whole documents).
python
if isinstance(data, list):
    if data and all(isinstance(item, list) for item in data):
        data = [item for sub in data for item in sub]
    return [item for item in data if isinstance(item, dict) and "text" in item]
src/utils/relevance.py — raised keep_ratio 0.5→0.65, min_survivors 3→5 (rank-based filter was cutting correct-but-lower-scoring facts, e.g. a named CEO at score -10).
src/search_providers/tavily_provider.py — stop defaulting missing url to "" (was causing broken localhost:5173 citation links). Now skips URL-less results and logs a warning.
src/agents/retriever.py — added retry-with-backoff (3 attempts, 1.5s/3s/6s) around each sub-query in bounded_search; asyncio.gather changed to return_exceptions=True so one failed query doesn't kill the whole search stage.
src/prompts/synthesis_prompt.py — removed the ## References section instruction from SYNTHESIS_SYSTEM_PROMPT (structured mode). LLM was hallucinating a fake references list by echoing claim text, duplicating the real one from citations.py. Confirmed already correctly applied in repo.
src/prompts/planner_prompt.py — two additions drafted, not yet confirmed applied:
Temporal decomposition rule for "since X" / evolution questions (split into sub-period queries)
Query specificity rule for product/entity questions (anchor to concrete terms, not broad category terms)
Complexity tier note: multi-year date range questions should be rated "complex" minimum
metrics/token_rate_limiter.py — new file, sliding-window TPM limiter (groq_token_limiter, 8000 TPM budget × 0.9 safety margin). Written and confirmed correct.
metrics/token_counter.py — track_llm_call() and TokenTrackingCallback updated to take explicit provider: str param instead of sniffing "groq" in model.lower() (which never matched since model name is "openai/gpt-oss-20b" with no "groq" substring). Confirmed correct.
Immediate next steps (in order)
Add provider= kwarg to all 4 track_llm_call(...) call sites — this is what actually activates the rate limiter, nothing works without it:
planner.py: add provider=config.model_provider, (config already imported)
synthesizer.py: add provider=config.model_provider, and add from src.config import config to imports (not currently there)
verifier.py: add provider=config.model_provider, and add from src.config import config to imports (not currently there)
claim_extractor.py: add provider="ollama", (should stay excluded from Groq rate limiting)
Confirm items #1, #2, #3, #4, #6 above are actually pasted into the live repo (several were discussed/written but user hadn't yet confirmed applying them before the conversation moved on).
Once rate limiter is fully wired, rerun the 3 test queries (simple/moderate/complex from earlier) and check: (a) time saved on verify/synthesize stages, (b) CEO-name fact survives relevance filtering now, (c) no more broken localhost:5173 citations.
Still open / not started
Consolidate COMPLEXITY_LIMITS (currently in planner.py, possibly duplicated in graph.py) into config.py as single source
Confirm Tailwind is actually applying to CitationList.jsx (red-border test) — "Single sourceweb" spacing bug still unconfirmed root-caused
Extract Claims stage speed (~48s avg bottleneck) — raise max_concurrent above 2, or restrict extraction more aggressively
Reranker (cross-encoder) reloading/reinstantiating on every retry pass — redundant HF cache roundtrips on 4GB GPU
content_extractor.py Tavily batch chunking — reviewed, confirmed already correct, no action needed
---------------------------------------------------------------------------------------------------
Stage	Range (s)	Avg (s)
plan	1.07–5.98	~2.3
search	3.72–13.42	~7.3
extract_claims	12.65–89.15	~40.5
relevance_filter	0.43–13.48	~3.2
verify	0.97–63.91	~24.5
synthesize	1.73–58.97	~19.5
--------------------------------------------------------------------------------------------------
Total time this run:

Stage	Time
plan	2.08s
search	6.74s
extract_claims	25.85s
relevance_filter	13.13s
verify	1.76s
synthesize	13.60s
Total	~63.2s