# Activ One MAX Policy RAG

A single-document retrieval-augmented assistant and evaluation harness for the
Aditya Birla Active One MAX policy wording. The policy PDF and 100-item golden dataset are inputs; put them in `data/` using the filenames below. They are not
generated or included by this project.

## Requirements and setup

- Python 3.11 or later
- An Anthropic API key for answering and optional judge/Ragas evaluation
- The supplied 56-page policy PDF and golden JSON dataset

From this directory in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set ANTHROPIC_API_KEY to your own key.
```

Place the inputs at:

```text
data/Active_One_Max_policy_wording-160925.pdf
data/activ_one_max_golden_dataset.json
```

Do not commit `.env`, policy indexes, or the evaluation output. They are ignored
by Git.

## Run

```powershell
# Parse, chunk, and index the policy; prints five deterministic sample chunks.
.\.venv\Scripts\python.exe -m src.ingest --pdf data/Active_One_Max_policy_wording-160925.pdf --rebuild

# Browser chat UI
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py

# One question in the terminal
.\.venv\Scripts\python.exe -m src.rag --ask "What is the free look period?"

# Run 10 items with the reranker and Ragas metrics enabled.
.\.venv\Scripts\python.exe -m eval.run_eval --limit 10 --rerank --ragas --no-judge

# Full 100-item evaluation with reranking and Ragas; --no-judge avoids
# the separate legacy judge call (Ragas still uses Claude).
.\.venv\Scripts\python.exe -m eval.run_eval --rerank --ragas --no-judge

# Tests
.\.venv\Scripts\python.exe -m pytest -q
```

The evaluation resumes by skipping IDs already in `eval/out/raw_results.jsonl`.
Use `--force` to rerun selected cases and replace the raw output. Additional
options are `--category "Out-of-Scope / Unanswerable"`, `--rerank`,
`--no-rerank`, `--ragas`, `--no-judge`, and `--out eval/out`. Ragas adds
Claude-scored context precision, context recall, and answer faithfulness for
in-scope cases. It requires the Anthropic API key and makes additional API
calls. `--no-judge` skips the separate custom Claude judge but does not skip
Ragas. Resume records are reused only when judge, Ragas, and reranker settings
match; use `--force` to replace the output.

The reranker is also available in the chat UI. Set `RERANK=true` in `.env` and
restart Streamlit. The first run may download `BAAI/bge-reranker-base`; use
`RERANK=false` or the eval `--no-rerank` option to disable it.

The Makefile offers `install`, `ingest`, `chat`, `ask`, `eval`, and `test`
targets. For example, `make ask Q="What is the free look period?"` or
`make eval EVAL_ARGS="--limit 10 --rerank --ragas --no-judge"`.

## Architecture

```text
Policy PDF
   |
   v
PyMuPDF blocks/tables -> glyph cleanup -> page/section/column-aware chunks
   |                                             |
   +-> SentenceTransformer -> Chroma             +-> BM25 pickle
                                    \               /
                                     RRF + boosts + optional BGE reranker
                                                  |
                                                  v
                            retrieval confidence / refusal gate
                                                  |
                                                  v
                                 Claude grounded answer + citations
                                      |                     |
                                  Streamlit UI          Eval harness
                                                             |
                                                       optional Ragas
```

## Design decisions

- **Two-column pages:** pages 10-27 are read using block coordinates and split
  into independent left clause and right glossary/simplification streams. The
  two streams are never concatenated into clause text. Simplified passages are
  down-weighted during retrieval; the answer prompt prioritizes original
  clauses because they prevail over simplified wording.
- **Hybrid retrieval:** local BGE embeddings and Chroma supply dense candidates;
  alias-aware BM25 supplies sparse candidates. Reciprocal-rank fusion uses
  `k=60`, then applies metadata boosts and optionally the BGE cross-encoder.
- **Refusal behavior:** retrieval confidence and lexical overlap control a
  pre-generation refusal gate. The generator is also instructed to refuse when
  retrieved passages do not substantiate an answer.
- **Conflicting clauses:** the answer prompt explicitly requires presenting
  both cited statements when the retrieved source passages conflict, rather
  than resolving policy wording conflicts silently.
- **Table extraction:** PyMuPDF table detection is preferred; block text remains
  the fallback. Tables are emitted as header/row text with page and section
  context so retrieval can match specific entries.

## Known limitations

- Extraction quality depends on the supplied PDF's selectable text and table
  geometry. OCR/scanned pages are not OCR-processed. Validate the generated
  `index/chunks.jsonl`, especially the two-column sections and Annexure IV,
  before relying on answers.
- Retrieval confidence is a lightweight calibration over embedding similarity
  and BM25 overlap, not a statistically calibrated probability.
- The LLM judge is model-dependent; its scores are estimates, not legal or
  actuarial validation. Evaluation results require the supplied golden set.
- This assistant reads policy wording only. A user's Policy Schedule may add,
  remove, or parameterize benefits; check it for personal cover details.
- Model weights are downloaded locally on first use; the optional reranker adds
  additional storage and startup time.
