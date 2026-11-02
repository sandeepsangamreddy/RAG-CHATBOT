# Activ One MAX Policy RAG

A single-document retrieval-augmented assistant and evaluation harness for the
Aditya Birla Activ One MAX policy wording. The policy PDF and 100-item golden
dataset are inputs; put them in `data/` using the filenames below. They are not
generated or included by this project.

## Requirements and setup

- Python 3.11 or later
- An Anthropic API key for answering and judge evaluation
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

# Run the 10-item smoke evaluation first; disable judge scoring if desired.
.\.venv\Scripts\python.exe -m eval.run_eval --limit 10

# Full 100-item evaluation. Add --no-judge to skip judge scoring.
.\.venv\Scripts\python.exe -m eval.run_eval

# Tests
.\.venv\Scripts\python.exe -m pytest -q
```

The evaluation resumes by skipping IDs already in `eval/out/raw_results.jsonl`.
Use `--force` to rerun selected cases and replace the raw output. Additional
options are `--category "Out-of-Scope / Unanswerable"`, `--rerank`,
`--no-rerank`, `--no-judge`, and `--out eval/out`. A run with `--no-judge`
still needs the API key for answer generation; it only skips the separate
LLM-as-judge calls.

The Makefile offers `install`, `ingest`, `chat`, `ask`, `eval`, and `test`
targets. For example, `make ask Q="What is the free look period?"`.

## Architecture

```text
Policy PDF
   |
   v
PyMuPDF blocks/tables -> glyph cleanup -> page/section/column-aware chunks
   |                                             |
   +-> SentenceTransformer -> Chroma             +-> BM25 pickle
                                    \               /
                                     RRF + boosts + optional reranker
                                                  |
                                                  v
                            retrieval confidence / refusal gate
                                                  |
                                                  v
                                 Claude grounded answer + citations
                                      |                     |
                                  Streamlit UI          Eval harness
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
