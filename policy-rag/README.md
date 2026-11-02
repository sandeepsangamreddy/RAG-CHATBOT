# Activ One MAX Policy RAG

A Python retrieval-augmented generation (RAG) application for asking questions
about the Aditya Birla Activ One MAX health insurance policy wording. It
includes PDF ingestion, local hybrid search, optional cross-encoder reranking,
Claude-based grounded answers, a Streamlit chat interface, and a golden-set
evaluation harness with optional Ragas metrics.

This is a policy-wording assistant, not a coverage decision, legal opinion, or
substitute for a member's Policy Schedule.

## What is in this repository

- `data/Active_One_Max_policy_wording-160925.pdf.pdf` — supplied policy PDF.
- `data/activ_one_max_golden_dataset.json` — 100 evaluation cases, with 20
  examples each for direct lookup, multi-hop reasoning, edge cases,
  conversational questions, and out-of-scope questions.
- `src/` — configuration, PDF ingestion and indexing, retrieval, generation,
  prompts, and command-line question answering.
- `app/streamlit_app.py` — Streamlit chat UI.
- `eval/` — golden-set runner, custom metrics and Claude judge, Ragas adapter,
  and report generation.
- `tests/` — tests for ingestion, retrieval, refusal behavior, and Ragas
  evaluation handling.
- `requirements.txt`, `.env.example`, and `Makefile` — Python dependencies,
  example configuration, and convenience commands.

The generated `index/` and evaluation output are intentionally not committed.
You build the index from the PDF on each new machine/deployment. The supplied
PDF and golden dataset are tracked in this repository; confirm you have
permission to publish or host the policy document before making the repository
or application public.

## Technology stack

| Area | Technology | Role |
|---|---|---|
| Language/runtime | Python (developed locally with Python 3.14.2; project setup targets Python 3.11+) | Application, ingestion, retrieval, evaluation |
| PDF processing | PyMuPDF (`pymupdf`) | Extracts page blocks and tables and uses layout coordinates |
| Chunking | Custom Python logic | Groups text by policy section/definition, preserves page metadata, handles tables and selected two-column pages |
| Embeddings | Sentence Transformers, `BAAI/bge-small-en-v1.5`, PyTorch | Creates local dense vectors for policy passages and queries |
| Vector store | ChromaDB persistent local collection | Stores and searches dense vectors using cosine distance |
| Numerical compatibility | NumPy | Array/numeric dependency used by the ML/vector stack |
| Keyword search | `rank-bm25` / BM25Okapi | Sparse retrieval with domain aliases; serialized locally with Python pickle |
| Rank fusion | Custom reciprocal-rank fusion (RRF) | Combines dense and BM25 rankings without adding unlike raw score scales |
| Reranker | Sentence Transformers cross-encoder, `BAAI/bge-reranker-base` | Reranks the fused candidate set; optional/configurable |
| Answer model | Anthropic Python SDK and Claude Messages API | Generates policy-grounded answers; rewrites follow-ups when chat history exists |
| Chat UI | Streamlit | Local interactive browser application |
| Evaluation | Custom Python harness and 100-item JSON golden set | Measures retrieval and answer quality; writes resumable results |
| LLM evaluation | Ragas 0.4.x, `langchain-community`, Anthropic adapter; optional custom Claude judge | Scores context precision, context recall, and faithfulness; makes additional API calls |
| Evaluation/report helpers | pandas, Rich, tqdm, RapidFuzz | CSV/report output, progress display, fuzzy matching of retrieved passages to golden snippets |
| Configuration | `python-dotenv`, environment variables | Loads local `.env` settings and API key |
| Tests | pytest | Runs automated tests |

Ragas is used only by the evaluation command, not during ordinary chat. The
embedding and reranker models run locally after their weights are downloaded.
Claude answer generation, query rewriting for follow-up turns, the optional
custom judge, and Ragas LLM metrics require network access and can incur
Anthropic API cost. Questions and retrieved policy passages are sent to
Anthropic when those API calls are made.

## Architecture and request flow

```text
Policy PDF
  |
  v
PyMuPDF page blocks/tables
  -> glyph and punctuation cleanup
  -> section-, definition-, table-, page-, and column-aware chunks
        |                                           |
        v                                           v
SentenceTransformer embeddings                 BM25Okapi corpus
        |                                           |
        v                                           v
Persistent ChromaDB dense search             Sparse keyword search
        \                                           /
         -> reciprocal-rank fusion (RRF), metadata boosts
            -> optional BGE cross-encoder reranking
            -> top-k passages and retrieval-confidence heuristic
            -> refusal gate when evidence is weak
            -> Claude answer grounded in retrieved passages
            -> citations checked against retrieved section/page metadata
            -> Streamlit UI or CLI

Golden dataset -> evaluation runner -> retrieval metrics / optional Claude judge
                                      -> optional Ragas scores
                                      -> JSONL, CSV, JSON, Markdown reports
```

### Ingestion and chunking

The PDF ingestion pipeline:

1. Reads page text blocks and table structures with PyMuPDF.
2. Uses geometry to separate clause and glossary/simplified columns on pages
   10–27; it avoids merging the two columns into one passage.
3. Cleans known PDF glyph defects and repeated page text.
4. Identifies section headings, definitions, and annexures and emits chunks
   with source filename, section, and page range metadata.
5. Chunks long text at sentence boundaries. The configured maximum of 450 is
   a regex-based word-like count, **not an exact model-token count**. Long
   clauses may carry approximately 15% whole-sentence overlap. Tables and
   definitions have specialized handling.
6. Writes inspectable `index/chunks.jsonl`, a BM25 pickle, and a persistent
   ChromaDB collection.

Table extraction uses PyMuPDF table detection and formats rows as searchable
text with column labels. Extraction falls back to page text blocks where
table-detection results are unavailable. Scanned pages are not OCR-processed.

### Retrieval and generation

- The query is embedded using the BGE query prefix. Alias expansion supports
  terms such as “piles”/“hemorrhoids” and “C-section”/“caesarean” for lexical
  search.
- Dense and sparse results are fused by RRF (`rrf_k=60` by default). Table and
  explicitly requested section matches can receive boosts; glossary and
  simplified passages are down-weighted.
- Up to 10 fused candidates are passed to the optional reranker; the default
  response context contains up to 6 passages (`TOP_K`).
- A lightweight retrieval-confidence heuristic can refuse before generation.
  The answer prompt separately instructs Claude to use only retrieved context,
  cite sources, respect optional-cover conditions, and surface conflicting
  policy statements instead of silently resolving them.
- Citation text generated by Claude is parsed and retained only when the cited
  section and page correspond to a retrieved chunk. This is a consistency
  check, not proof that each factual claim is supported.
- Follow-up questions with conversation history are rewritten into standalone
  queries by Claude before retrieval. With no history, the original question
  is used directly.

## Setup

Run commands from the `policy-rag` directory. Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY` to your own key. Do not commit `.env`,
put a real key in `.env.example`, or share a key in logs/screenshots. The app
loads `.env` from the project directory and also reads environment variables.

The project has been run in a local Python 3.14.2 virtual environment.
`requirements.txt` declares lower bounds rather than a fully pinned lockfile;
recreating the environment can therefore resolve newer dependency versions.

## Build the policy index

The PDF currently in `data/` has a double `.pdf.pdf` suffix. The ingestion
default is configured to match that filename:

```powershell
.\.venv\Scripts\python.exe -m src.ingest --rebuild
```

Or specify the input explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.ingest --pdf "data\Active_One_Max_policy_wording-160925.pdf.pdf" --rebuild
```

Ingestion reports the page/chunk counts and prints deterministic sample
chunks. Inspect `index/chunks.jsonl`, especially the two-column sections and
tables, after changing the PDF or extraction logic. Rebuild whenever the PDF,
chunking, embedding model, or index format changes. The generated index is
local and git-ignored; a fresh clone must build it before asking questions.

## Run the application

Start the browser UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.
Keep the terminal process running while using the app; press `Ctrl+C` there to
stop it.

Ask one question from the terminal:

```powershell
.\.venv\Scripts\python.exe -m src.rag --ask "What is the free look period?"
```

The app requires a valid Anthropic API key for answer generation. The Streamlit
sidebar shows the configured Claude model, retrieval count, reranker state,
and refusal threshold.

## Configuration

These settings are read from environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Required for Claude answer generation and LLM-based evaluation |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Model used for answers and follow-up query rewriting |
| `JUDGE_MODEL` | `CLAUDE_MODEL` value | Model used by the custom judge and Ragas |
| `RERANK` | `false` in code; `true` in `.env.example` | Enable the local cross-encoder reranker |
| `TOP_K` | `6` | Number of passages returned to generation |
| `REFUSAL_THRESHOLD` | `0.15` | Heuristic pre-generation refusal threshold |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence Transformer embedding model |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker model |

The `.env.example` file is a template, not the live `.env`. The first run may
download model weights from Hugging Face and needs sufficient disk space,
memory, and network access. Set `RERANK=false` to reduce local model load and
startup cost; restart Streamlit after changing settings.

Other retrieval/evaluation parameters (candidate counts, RRF constant, chunk
limits, retry settings, and evaluation worker count) are defined in
`src/config.py` rather than exposed as environment variables.

## Golden-set evaluation

The supplied JSON dataset is used by `eval.run_eval`. It has 100 unique items
with IDs, question, reference answer, verbatim context, source reference, and
category. Run a small API-backed evaluation:

```powershell
.\.venv\Scripts\python.exe -m eval.run_eval --limit 5 --rerank --no-judge
```

`--no-judge` skips the separate custom Claude judge, but answers are still
generated by Claude. Add `--ragas` to make additional Ragas calls for
context precision, context recall, and faithfulness on in-scope examples.
Out-of-scope examples are excluded from Ragas scoring.

Example for all 100 questions with the custom judge, reranker, and Ragas:

```powershell
.\.venv\Scripts\python.exe -m eval.run_eval --rerank --ragas
```

This can make many API calls, take significant time, and incur cost. Use
`--limit` to try a small run first. `--no-judge` is useful when only retrieval
and/or Ragas results are wanted; it does not disable answer generation.

Available evaluation options:

| Option | Effect |
|---|---|
| `--limit N` | Evaluate the first N cases after category filtering |
| `--category "..."` | Restrict evaluation to an exact category name |
| `--no-judge` | Skip the custom answer judge |
| `--ragas` | Add Ragas context precision, context recall, and faithfulness |
| `--rerank` / `--no-rerank` | Override the configured reranker for this run |
| `--out PATH` | Select the output directory (default `eval/out`) |
| `--force` | Ignore prior rows and replace the raw JSONL with this run's selected cases |

The runner normally resumes from `eval/out/raw_results.jsonl`, skipping an
item only when its saved judge, Ragas, and reranker flags match the current
run. Use `--force` to rerun. When using `--force` with a category or limit, the
raw-results file is replaced with only those selected cases.

### Evaluation metrics and outputs

- **Retrieval metrics:** page hit@3/@6, section hit@3/@6, mean reciprocal rank
  (MRR) for expected pages, and custom verbatim-context snippet recall.
- **Citation metric:** proportion of parsed cited pages matching expected
  source pages.
- **Refusal metrics:** refusal precision/recall/F1, false-refusal rate, and
  out-of-scope hallucination rate.
- **Custom Claude judge:** correctness, completeness, faithfulness (each 1–5),
  and a hallucination flag with a short reason.
- **Ragas:** LLM-based context precision, context recall, and answer
  faithfulness. These are separate from the similarly named custom metrics.
- **Pass rate:** out-of-scope cases pass when refused; in-scope pass requires
  judge correctness and faithfulness scores of at least 4 and no refusal.

The output directory contains:

- `raw_results.jsonl` — resumable per-item records, including answers,
  retrieved chunks, citations, timing, and scores.
- `results.csv` — flattened per-item results.
- `aggregate_metrics.json` — overall and per-category aggregate metrics.
- `report.md` — readable summary and failure analysis.

Treat LLM-judge and Ragas values as model-assisted estimates, not ground truth.
The evaluation runner produces answers and judges via Anthropic; protect the
output because it may contain user prompts, policy passages, and generated
answers.

## Tests

From `policy-rag/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Some retrieval tests require the built policy index and may be skipped if it
is missing. Tests do not constitute legal validation or a guarantee of
production readiness.

## Deployment

The app can be deployed as a Streamlit application, but a fresh host needs
Python dependencies, the policy PDF, the generated index, model weights, and
an Anthropic API key. The current code reads the key from the process
environment or local `.env`; never commit a real key. The generated `index/`
is ignored by Git and must be created or supplied in the deployment
environment. The index is local filesystem storage, so a host with ephemeral
storage may need an index-build step or persistent volume.

For Streamlit Community Cloud, select `app/streamlit_app.py` as the app entry
point and make dependencies available where the build can install them (the
current `requirements.txt` is in this project directory). Configure the key
using the host's secret mechanism and ensure it is available as
`ANTHROPIC_API_KEY` to this code. Provision/build the index before the app
tries to answer. The current repository does not include deployment automation
or a startup index-build hook; confirm model downloads, memory, disk, network,
and build-time limits on the chosen host.

The chat UI currently has no user authentication or rate limiting. Do not
expose it publicly without adding access controls and controlling API usage.
The app sends questions and retrieved passages to Anthropic for generation.
Only deploy the policy document where you are permitted to host it.

## Design choices and alternatives

- **ChromaDB plus BM25:** persistent local vector search and a simple lexical
  index suit this single-document prototype without a separate database
  service. The project has no relational database. PostgreSQL/pgvector,
  Elasticsearch/OpenSearch, or a managed vector store could be better for
  multi-user, multi-document, filtering, operational, or larger-scale needs.
- **Hybrid retrieval:** dense search handles semantic paraphrases; BM25 helps
  exact policy terms and identifiers. RRF combines ranks without requiring
  calibration of the two systems' raw score scales.
- **BGE cross-encoder:** adds query-passage interaction to reranking a small
  fused candidate set. It uses local compute and model storage; compare it to
  alternative rerankers on the golden set before treating it as the best
  option.
- **Claude API:** avoids hosting a generation model locally and provides one
  provider for answers and LLM-assisted evaluation. It requires network access,
  costs money, and sends request context to Anthropic. A local or different
  hosted model is an alternative, subject to quality, performance, and privacy
  evaluation.
- **Streamlit:** provides a compact Python UI for a demo. A larger production
  system may benefit from a separately deployed API, authentication, and
  dedicated frontend.

## Known limitations and safety

- PDF extraction depends on selectable text and table geometry. Scanned
  documents are not OCR-processed. Review extracted chunks after re-ingestion.
- Chunk limits use word-like regex counts, not a model tokenizer.
- Retrieval confidence is a heuristic over dense similarity and BM25 overlap,
  not a calibrated probability. Reranking is not a substitute for evaluation.
- Citation parsing checks metadata correspondence; it cannot guarantee that
  citations entail every sentence.
- The LLM judge and Ragas depend on model behavior and can be inconsistent.
  Keep a human-reviewed golden set and inspect errors by category.
- The assistant only sees the policy wording. Personal benefits may depend on
  a Policy Schedule the app does not load.
- An incorrect, incomplete, or outdated PDF/index can cause unsupported or
  misleading answers. Do not use this prototype as an authoritative claims or
  coverage decision system.
