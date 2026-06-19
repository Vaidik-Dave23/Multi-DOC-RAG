# DocMind — Multi-Document RAG Platform

A Retrieval-Augmented Generation system for querying and comparing large document sets, built end-to-end as a portfolio project demonstrating production ML engineering: retrieval architecture, local embeddings, an LLM-judged evaluation framework, and empirical chunking-strategy benchmarking on a real 502-page legal corpus.

> **TL;DR:** Naive fixed-size chunking outperformed hierarchical, semantic, and sentence-aware chunking on this corpus — a counter-intuitive, fully-quantified result with a custom RAGAS-style evaluation pipeline that costs 75% fewer LLM calls than a naive implementation. Full findings in [Benchmark Results](#benchmark-results).

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [API reference](#api-reference)
- [Chunking strategies](#chunking-strategies)
- [Evaluation framework](#evaluation-framework)
- [Benchmark results](#benchmark-results)
- [Engineering decisions & tradeoffs](#engineering-decisions--tradeoffs)
- [Debugging log](#debugging-log--bugs-found-via-the-benchmark-itself)
- [Setup & usage](#setup--usage)
- [Project structure](#project-structure)
- [Future work](#future-work)

---

## Overview

DocMind lets a user upload one or more PDFs and then:

- **Query** a single document in natural language (`/query`)
- **Compare** multiple documents against each other on a given dimension (`/compare`)
- **Extract** structured fields from a document
- **Submit feedback** on any generated answer, logged for later review

The interesting engineering problem this project solves isn't "can an LLM answer questions about a PDF" — it's **how do you actually know if your RAG pipeline is good**, and **which of several reasonable design choices (chunking strategy, retrieval depth, model) is empirically better for your corpus**, rather than just assumed to be better.

To answer that, this project includes a custom-built RAGAS-style evaluation harness that:

- Runs a fixed battery of test questions (factual, conceptual, synthesis, and adversarial/hallucination-trap categories) against the live API
- Scores every answer on **faithfulness**, **answer relevancy**, **context precision**, and **context recall**
- Splits scoring between Gemini (for the two metrics that genuinely require reasoning) and a local SentenceTransformer (for the two that don't), cutting LLM calls by ~75%
- Automates the full upload → benchmark → save cycle per chunking strategy, so four chunkers can be compared on identical ground truth with one command each

---

## Architecture

![DocMind system architecture](diagrams/architecture.svg)

**Request flow:**

1. **Upload** — a PDF is read, text extracted (`app/utils/pdf.py`), split into chunks by the active strategy in `app/utils/chunking.py`, embedded locally via `all-MiniLM-L6-v2`, and added to a FAISS `IndexFlatL2` index. Chunk text, embeddings, and metadata (doc_id, filename, chunk_index) are appended to an in-memory list and persisted to `store/metadata.json` + `store/indexes.bin`.
2. **Query** — the question is embedded with the same model, FAISS returns the top-k most similar chunks for the target document, and the chunks plus question are sent to Gemini for generation.
3. **Compare** — same retrieval step, run independently per document, then all retrieved excerpts are combined into a single structured prompt asking Gemini to compare/contrast across documents.
4. **Persistence** — everything (FAISS index, chunk metadata, response log, feedback log) is flat-file JSON/binary on local disk — intentionally simple, no external database, which keeps the project runnable with zero infrastructure setup.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API framework | FastAPI | async-native, automatic OpenAPI docs, Pydantic validation |
| Vector store | FAISS (`IndexFlatL2`) | exact search, no approximation error — appropriate at this corpus scale |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | fast, runs locally on CPU, no API cost or rate limits for the embedding step |
| Generation | Gemini API (`google-genai` SDK), `gemini-2.5-flash` | strong cost/quality tradeoff; **note:** originally built on `gemini-2.0-flash`, which Google deprecated mid-project — see [Debugging Log](#debugging-log--bugs-found-via-the-benchmark-itself) |
| Persistence | flat JSON + FAISS binary | zero infra, trivially portable, sufficient at this scale |
| Evaluation | custom RAGAS-style harness (no `ragas` library) | see [Evaluation Framework](#evaluation-framework) for why |

---

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | `POST` | Upload a PDF, chunk it with the currently active strategy, embed, index |
| `/query` | `POST` | `{"question": str}` → answer grounded in the single most relevant document's chunks |
| `/compare` | `POST` | `{"query": str, "doc_ids": [str]}` → cross-document comparison |
| `/documents` | `GET` | List all uploaded documents with `doc_id`, `filename`, `chunk_count` |
| `/documents/{doc_id}` | `DELETE` | Remove a document and all its chunks |
| `/feedback` | `POST` | Log a thumbs up/down + comment against a previous response |
| `/debug` | `GET` | Raw FAISS vector count + metadata sample, for sanity-checking the index |

---

## Chunking strategies

Four interchangeable strategies live in `app/utils/chunking.py`. The active strategy is selected by which function `upload.py` calls — a deliberate simplicity tradeoff (see [Engineering Decisions](#engineering-decisions--tradeoffs)):

| Strategy | Approach |
|---|---|
| **Naive** | Fixed-size character/token windows with simple overlap |
| **Hierarchical** | Splits by document structure (sections → paragraphs) before falling back to size limits |
| **Semantic** | Groups sentences by embedding similarity, splitting where topic shifts |
| **Sentence** | Splits strictly on sentence boundaries, grouped to a target size |

---

## Evaluation framework

![RAGAS evaluation pipeline](diagrams/evaluation_pipeline.svg)

### Why not just use the `ragas` library

The official `ragas` package pins `langchain-core<0.3`, while any current `langchain-google-genai` integration requires `langchain-core>=1.3`. These cannot coexist in the same virtual environment — every installation attempt produced a different half-broken combination (`ModuleNotFoundError: langchain_core.pydantic_v1`, `ChatVertexAI` import failures, etc.). Rather than pin to increasingly ancient transitive dependencies, the four RAGAS metrics were **reimplemented directly against the `google-genai` SDK** the app already uses — zero new dependency surface, full control over prompts and parsing.

### The four metrics, and where each one runs

| Metric | What it measures | Where it runs | Why |
|---|---|---|---|
| **Faithfulness** | Are the answer's claims actually supported by the retrieved context? | Gemini | Requires real claim-by-claim reasoning against source text — not reducible to embedding similarity |
| **Context Recall** | Does the retrieved context contain enough information to derive the ground-truth answer? | Gemini | Same — judging "is this information present" is a reasoning task |
| **Answer Relevancy** | Does the answer actually address the question asked? | Local (SentenceTransformer cosine similarity, question ↔ answer) | Directional alignment between question and answer is a reasonable embedding-similarity proxy, and removing it from the Gemini budget halves the per-query call count |
| **Context Precision** | What proportion of retrieved chunks are relevant to the question? | Local (cosine similarity, question ↔ each chunk) — batched single Gemini call available via `--use-gemini-precision` as a fallback | Same rationale; default is local |

### Cost optimization journey

The benchmark evolved through several rounds of real bottlenecks hit in practice, each one fixed in turn:

1. **v1 → v2**: dropped the `ragas` library entirely (dependency conflict, above), reimplemented metrics as direct Gemini prompts.
2. **Free-tier rate limits**: Gemini's free tier caps at 5 req/min and 20 req/day on certain models. Added exponential backoff with retry-delay parsing from the API's own error response, plus (temporarily, during free-tier testing) fixed inter-request delays.
3. **Batched context precision**: scoring 5 retrieved chunks one Gemini call each → one call returning a JSON map of all 5 scores at once.
4. **Hash-based JSON cache**: every metric call is keyed on `sha256(question, answer, contexts, metric)`. Rerunning the same benchmark — which happens constantly during iteration — costs nothing on a cache hit.
5. **SentenceTransformer substitution**: moved `answer_relevancy` and `context_precision` off Gemini entirely (table above). **Net result: 8 Gemini calls per query sample → 2.** Across the full 14-question suite, **109 calls → ~28**.
6. **Paid tier**: once the project moved off the free tier, all artificial rate-limit delays were removed; only genuine transient-error retries remain.
7. **Two-stage pipeline** (`--two-stage`): an optional mode where Stage 1 scores every test case with the free local metrics only, ranks them, and Stage 2 spends Gemini calls only on the top-N survivors — useful when iterating on a chunker and wanting a cheap first-pass signal before committing API budget to the full reasoning-based metrics.

### Automated chunker comparison

```bash
# Edit upload.py to call the desired chunk function, restart the server, then:
python ragas_benchmark_v4.py --chunker naive        --auto-upload
python ragas_benchmark_v4.py --chunker hierarchical --auto-upload
python ragas_benchmark_v4.py --chunker semantic      --auto-upload
python ragas_benchmark_v4.py --chunker sentence      --auto-upload

# Then, the payoff:
python ragas_benchmark_v4.py --compare-chunkers
```

`--auto-upload` wipes whatever's currently indexed and re-uploads every PDF in `Test_Data/`, so each chunker is benchmarked against an identical, clean corpus. Each run's results land in their own namespaced folder (`benchmark_results/<chunker>/`) with their own score cache, so reruns within a chunker get cache hits without polluting the others.

---

## Benchmark results

**Test corpus:** *Kesavananda Bharati v. State of Kerala* (1973) — a 502-page, ~422,000-word Indian Supreme Court constitutional law judgment, split into 10 thematic PDF segments (case overview, eight individual judges' opinions, a curated "basic structure doctrine" extract, and a majority-vs-minority comparison document).

**Test set:** 11 `/query` questions (factual, conceptual, synthesis, and 2 deliberate hallucination-trap edge cases) + 3 `/compare` questions (cross-document comparisons of opposing judicial views).

### /query — aggregate scores by chunker

| Chunker | Latency (ms) | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---:|---:|---:|---:|---:|
| **Naive** | 8,425 | 0.893 | 0.795 | **0.625** | **0.590** |
| Hierarchical | 8,992 | 0.925 | 0.777 | 0.170 | 0.046 |
| Semantic | 16,402 | **1.000** | 0.789 | 0.161 | 0.182 |
| Sentence | 14,422 | 0.970 | **0.815** | 0.275 | 0.273 |

### /compare — aggregate scores by chunker

| Chunker | Latency (ms) | Faithfulness | Answer Relevancy | Context Precision |
|---|---:|---:|---:|---:|
| **Naive** | 52,376 | **0.878** | 0.742 | **0.643** |
| Hierarchical | 58,254 | 0.773 | 0.755 | 0.599 |
| Semantic | 46,808 | 0.562 | 0.780 | 0.644 |
| Sentence | 50,703 | 0.285 | **0.793** | 0.634 |

### The headline finding

**Naive chunking won on the metrics that matter most for a RAG system: context precision and context recall, on both endpoints, by a wide margin.** This was not the expected outcome — semantic and hierarchical chunking are usually assumed to retrieve more relevant context, and the data says otherwise on this corpus.

The qualitative evidence makes the *why* concrete. On the same edge-case question about the Kerala Land Reforms Act:

- **Naive** correctly retrieved the actual statutory passages and answered that the judgment doesn't prescribe punishments, citing specific paragraph numbers.
- **Hierarchical** retrieved a single unrelated fragment about John Locke and natural-rights theory.
- **Semantic** retrieved excerpts about property acquisition under Article 31(2) and an unrelated case citation.
- **Sentence** retrieved excerpts about equality and non-discrimination — also unrelated.

**Interpretation:** for a dense, long-paragraph legal judgment where meaning depends on surrounding sentences staying together, fixed-size windowing with overlap preserved more usable local context than chunkers that split on structural or semantic boundaries — those boundaries, on this specific document, didn't align well with where the legally relevant information actually lived. This is a genuinely useful, generalizable lesson: **chunking strategy quality is corpus-dependent, and "smarter" splitting logic is not a free win** — it has to be validated empirically against the documents it will actually run on, not assumed from general intuition.

### A caveat on faithfulness, and what it does and doesn't mean

Semantic chunking's perfect 1.000 faithfulness on `/query` looks like the best result in the table — but faithfulness only measures *"is this claim supported by the retrieved context,"* not *"is this claim true."* In the qualitative review, Hierarchical's answer to a factual count question stated "eleven judges" against a ground truth of 13, and still scored a high faithfulness number, because that's a hallucination-vs-context-recall failure, not a faithfulness one. The right way to read this benchmark is faithfulness alongside context recall, never in isolation — a chunker can be "faithful to garbage" if its retrieval handed the model the wrong material in the first place.

Sentence chunking's compare-faithfulness of 0.285 is the most damning single number in the report: it indicates the model was working from thin, disconnected sentence-level fragments and filling the gaps with outside knowledge not grounded in what was actually retrieved — a genuine hallucination signal, not a measurement artifact.

---

## Engineering decisions & tradeoffs

A running list of deliberate choices made during this project, kept here because they're exactly the kind of decisions an interviewer asks about.

**FAISS deletion strategy.** `DELETE /documents/{doc_id}` removes chunks from `store.metadata` but does not rebuild the FAISS index, leaving orphaned vectors in place. This trades a small amount of wasted memory and marginally slower search for avoiding a full reindex on every delete — a reasonable choice at this scale, explicitly **not** the right choice if document churn were high-frequency in production, where a periodic compaction job would be needed instead.

**Module-scoping pattern for shared state.** Early versions imported `metadata` and `index` directly from `store.py` into route files (`from app.store import metadata`), which created local copies that silently diverged from the live module state. Fixed by switching every route to `import app.store as store` and referencing `store.metadata` / `store.index` throughout, so all routes share one live reference.

**Chunking strategy is a code-level switch, not a request parameter.** `upload.py` calls one hardcoded chunk function; changing strategy means editing that line and restarting the server. This was a deliberate scope decision to keep `/upload`'s request contract simple during the comparison phase — the natural next step (a `?chunker=` query param routing to all four functions) is in [Future Work](#future-work).

**Re-embedding cost in `/compare`.** Originally, `compare.py` called `embedding_model.encode()` fresh on every retrieved document's chunks, on every single comparison request — even though those same chunks were already embedded once at upload time and that vector was simply discarded. Fixed by caching each chunk's embedding vector in `store.metadata` at upload time and reading it back directly in `compare.py`, eliminating the redundant encode pass. This is the single highest-leverage performance fix in the project — see the debugging log below for the actual before/after numbers.

**Gemini free-tier quota as a first-class design constraint.** Both DocMind itself and the benchmark harness had to be built around Gemini's free-tier rate and daily quota limits (5 req/min, 20 req/day on certain models) — not an edge case to handle defensively, but a constraint that directly shaped the evaluation architecture's emphasis on minimizing total API calls.

---

## Debugging log — bugs found via the benchmark itself

The evaluation harness didn't just score the system — it surfaced real bugs that would have gone unnoticed without an automated, repeatable test suite. Documented here because finding bugs *through* an evaluation framework is itself a notable engineering outcome, not just a benchmark result.

### Bug 1 — `/compare` doc_id mismatch (silent retrieval failure)

**Symptom:** every `/compare` answer from the model was some variant of *"please provide the document excerpts, none were given."*

**Root cause:** the benchmark client was sending document **filenames** as `doc_ids`, but `compare.py` matches against the **UUID** `doc_id` key actually stored in `store.metadata`. The mismatch produced an empty `doc_contexts` list every time, silently — no error, just an empty prompt that Gemini correctly flagged as missing input.

**Fix:** corrected the benchmark client to pull the real `doc_id` field from `/documents`' response instead of `filename`.

**Lesson:** a route can be "working" (200 OK, valid response shape) while doing zero of the actual retrieval work it's supposed to do. Field-name mismatches between two systems are invisible without an end-to-end test that checks *content*, not just status codes.

### Bug 2 — `/compare` never returned its retrieved contexts

**Symptom:** `context_precision` scored exactly `0.0` across every chunker, every question, even after Bug 1 was fixed.

**Root cause:** `compare.py` built `doc_contexts` (the retrieved chunk text per document) and used it to construct the Gemini prompt — but never included it in the JSON response. The evaluation harness had nothing to score against.

**Fix:** added `"contexts": doc_contexts` to `compare.py`'s return payload.

**Lesson:** an endpoint's *internal* correctness and its *observability* are separate concerns — `/compare` was retrieving the right chunks all along once Bug 1 was fixed, but there was no way to verify that from outside the function until the response actually exposed what it had retrieved.

### Bug 3 — `gemini-2.0-flash` deprecation, mid-project

**Symptom:** every `/query` and `/compare` call started failing with `404 NOT_FOUND: This model models/gemini-2.0-flash is no longer available`.

**Root cause:** Google retired the `gemini-2.0-flash` family. The model string was hardcoded in `query.py`/`compare.py` and also referenced in the benchmark scripts.

**Fix:** migrated both the application and the benchmark harness to `gemini-2.5-flash`.

**Lesson:** model deprecation cycles are a real, recurring operational concern for any system built on a hosted LLM API — pinning model strings to an environment variable (`GEMINI_MODEL` in `.env`) rather than hardcoding them is the correct long-term pattern, even though this project's `.env` value still required a manual update when the model was retired.

### Bug 4 — redundant re-embedding made `/compare` take 165 seconds

**Symptom:** `/compare` requests were timing out even at a 120-second client timeout; raised to 300s, they completed but took up to ~165,000ms each.

**Root cause:** `compare.py` called `embedding_model.encode(chunk_texts)` fresh, for every chunk of every document, on every single request — completely redundant, since those same chunks were already embedded once at upload time, with the resulting vectors simply discarded.

**Fix:** cache each chunk's embedding vector in `store.metadata` at upload time (`embed_and_store()` now stores `.tolist()` of the FAISS-bound vector alongside the chunk text); `compare.py` reads `c["embedding"]` directly instead of re-encoding.

**Lesson:** "compute once, reuse" is the most basic optimization in computing, and it's exactly the kind of redundant work that's easy to introduce when a retrieval helper function is written once for one endpoint (`/query`, which embeds the query but reuses indexed chunk vectors via FAISS) and then copy-adapted for a second endpoint (`/compare`) without re-examining what's actually being recomputed unnecessarily.

---

## Setup & usage

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`.env`:
```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash
```

```bash
uvicorn main:app --reload
```

### Benchmark harness

```bash
pip install google-genai requests pandas sentence-transformers

# Drop your test PDFs in Backend/Test_Data/, then:
python ragas_benchmark_v4.py --chunker naive --auto-upload --api-key YOUR_KEY

# After running all four chunkers:
python ragas_benchmark_v4.py --compare-chunkers
```

Useful flags:

| Flag | Effect |
|---|---|
| `--chunker LABEL` | namespaces output under `benchmark_results/<label>/` |
| `--auto-upload` | wipes existing docs, uploads everything in `--test-data-dir` |
| `--no-wipe` | with `--auto-upload`, skip the wipe step |
| `--two-stage --top-n N` | cheap local pre-filter before spending Gemini calls on survivors |
| `--use-gemini-precision` | batched-Gemini context precision instead of the local default |
| `--skip-query` / `--skip-compare` | run only one half of the suite |
| `--no-cache` / `--clear-cache` | disable or reset the score cache |
| `--compare-chunkers` | skip benchmarking, print the cross-chunker comparison table |

---

## Project structure

```
backend/
├── app/
│   ├── routes/
│   │   ├── upload.py        # chunking strategy selected here
│   │   ├── query.py
│   │   ├── compare.py
│   │   ├── extract.py
│   │   └── feedback.py
│   ├── utils/
│   │   ├── chunking.py      # naive · hierarchical · semantic · sentence
│   │   ├── embedding.py     # SentenceTransformer + FAISS glue
│   │   └── pdf.py
│   └── store.py             # FAISS index + metadata, JSON persistence
├── store/                   # persisted on disk: indexes.bin, metadata.json, logs
├── Test_Data/               # benchmark corpus PDFs (drop-in, gitignored)
├── benchmark_results/
│   ├── naive/
│   ├── hierarchical/
│   ├── semantic/
│   ├── sentence/
│   └── chunker_comparison.csv
├── ragas_benchmark_v4.py    # evaluation harness
├── main.py
└── .env
```

---

## Future work

- **`?chunker=` query parameter on `/upload`** — eliminate the one remaining manual step (editing `upload.py` + restarting the server) in the chunker comparison workflow, enabling a true single-command sweep across all four strategies.
- **Investigate the Q05 retrieval gap** — every chunker failed to retrieve the actual Basic Structure Doctrine definition for a direct question about it, despite that being the central holding of the case. Worth checking whether this is a genuine corpus-segmentation issue (the relevant judgment section not landing in the uploaded test PDFs as expected) versus a retrieval weakness shared across all four chunkers.
- **Hybrid chunking** — given naive chunking's win on this corpus, a worthwhile follow-up is a hybrid strategy (e.g. semantic boundary detection with a naive-style minimum window size) to test whether it's possible to capture semantic chunking's structural awareness without losing naive's local-context preservation.
- **Deployment to Render.**
- **Embedding cache backfill** for any documents uploaded before the embedding-caching fix, so `/compare` doesn't require a full re-upload of historical data.
