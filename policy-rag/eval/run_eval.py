"""Run resumable retrieval and answer-quality evaluation on the golden dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from tqdm import tqdm

from eval.metrics import item_metrics, judge_answer
from eval.report import write_report
from src.config import settings
from src.rag import RAGPipeline

LOGGER = logging.getLogger(__name__)
REQUIRED_FIELDS = {
    "id", "category", "user_query", "ground_truth_response", "verbatim_context", "source_reference"
}


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("Golden dataset must be a JSON array.")
    for index, item in enumerate(dataset):
        if not isinstance(item, dict) or not REQUIRED_FIELDS.issubset(item):
            raise ValueError(f"Golden dataset item {index} is missing one or more required fields.")
    return dataset


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return existing
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                existing[str(row["id"])] = row
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid raw-results line {line_number} in {path}: {exc}") from exc
    return existing


def _evaluate_one(
    item: dict[str, Any],
    pipeline: RAGPipeline,
    client: Any | None,
    do_judge: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = pipeline.ask(item["user_query"], history=None)
    latency = time.perf_counter() - started
    judge = None
    if do_judge:
        if client is None:
            raise ValueError("An Anthropic client is required for judge evaluation.")
        judge = judge_answer(
            item["user_query"], item["ground_truth_response"], result["answer"],
            result["retrieved"], client, model=pipeline.settings.judge_model,
        )
    metrics = item_metrics(item, result, judge)
    retrieved_pages: list[int] = sorted({
        page
        for chunk in result["retrieved"]
        for page in range(int(chunk["page_start"]), int(chunk["page_end"]) + 1)
    })
    row: dict[str, Any] = {
        "id": item["id"],
        "category": item["category"],
        "query": item["user_query"],
        "ground_truth_response": item["ground_truth_response"],
        "source_reference": item["source_reference"],
        "answer": result["answer"],
        "citations": result["citations"],
        "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in result["retrieved"]],
        "retrieved_pages": retrieved_pages,
        "retrieved_sections": [chunk["section_id"] for chunk in result["retrieved"]],
        "retrieved": result["retrieved"],
        "refused": result["refused"],
        "rewritten_query": result["rewritten_query"],
        "latency": latency,
        **metrics,
    }
    return row


def run(
    *,
    limit: int | None = None,
    category: str | None = None,
    do_judge: bool = True,
    rerank: bool | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Run selected evaluation cases concurrently and write resumable outputs."""
    data = _load_dataset(settings.dataset_path)
    if category:
        data = [item for item in data if item["category"] == category]
    if limit is not None:
        data = data[:limit]
    if not data:
        raise ValueError("No golden dataset items match the selected category/limit.")

    destination = output_dir or settings.project_root / "eval" / "out"
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "raw_results.jsonl"
    previous = {} if force else _read_existing(raw_path)
    selected = [item for item in data if str(item["id"]) not in previous]
    effective_settings = replace(settings, rerank=settings.rerank if rerank is None else rerank)
    pipeline = RAGPipeline(effective_settings)
    client: Any | None = None
    if do_judge:
        if not effective_settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing; set it in .env or run with --no-judge.")
        client = Anthropic(api_key=effective_settings.anthropic_api_key)

    mode = "w" if force else "a"
    with raw_path.open(mode, encoding="utf-8") as raw_file:
        with ThreadPoolExecutor(max_workers=effective_settings.eval_workers) as pool:
            futures = {
                pool.submit(_evaluate_one, item, pipeline, client, do_judge): item for item in selected
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                item = futures[future]
                row = future.result()
                previous[str(item["id"])] = row
                raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_file.flush()

    ordered = [previous[str(item["id"])] for item in data if str(item["id"]) in previous]
    if not ordered:
        raise RuntimeError("Evaluation completed without producing or loading any results.")
    write_report(ordered, destination)
    return ordered


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for golden-dataset evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--no-judge", action="store_true")
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank", action="store_true")
    rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.set_defaults(rerank=None)
    parser.add_argument("--out", type=Path, default=settings.project_root / "eval" / "out")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(
            limit=args.limit, category=args.category, do_judge=not args.no_judge,
            rerank=args.rerank, output_dir=args.out, force=args.force,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
