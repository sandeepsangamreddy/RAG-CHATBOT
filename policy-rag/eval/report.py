"""Persist evaluation tables and render a concise Markdown/terminal report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from eval.metrics import aggregate_metrics

METRIC_KEYS = (
    "page_hit_at_3", "page_hit_at_6", "section_hit_at_3", "section_hit_at_6",
    "mrr_page", "context_recall", "correctness", "completeness", "faithfulness",
    "judge_hallucination_rate", "refusal_precision", "refusal_recall", "refusal_f1",
    "false_refusal_rate", "out_of_scope_hallucination_rate", "citation_accuracy",
    "mean_latency", "pass_rate", "ragas_context_precision", "ragas_context_recall",
    "ragas_faithfulness",
)


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Write item CSV, Markdown report, and print a rich summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_metrics(rows)
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        judge = row.get("judge") or {}
        ragas = row.get("ragas") or {}
        flat_rows.append({
            "id": row.get("id"), "category": row.get("category"), "query": row.get("query"),
            "answer": row.get("answer"), "refused": row.get("refused"),
            "retrieved_pages": ", ".join(str(p) for p in row.get("retrieved_pages", [])),
            "expected_pages": ", ".join(str(p) for p in row.get("expected_pages", [])),
            "expected_sections": ", ".join(row.get("expected_sections", [])),
            "page_hit_at_3": row.get("page_hit_at_3"), "page_hit_at_6": row.get("page_hit_at_6"),
            "section_hit_at_3": row.get("section_hit_at_3"),
            "section_hit_at_6": row.get("section_hit_at_6"),
            "mrr_page": row.get("mrr_page"), "context_recall": row.get("context_recall"),
            "citation_accuracy": row.get("citation_accuracy"), "latency": row.get("latency"),
            "correctness": judge.get("correctness"), "completeness": judge.get("completeness"),
            "faithfulness": judge.get("faithfulness"), "hallucination": judge.get("hallucination"),
            "judge_reason": judge.get("reason"), "pass": row.get("pass"),
            "ragas_context_precision": ragas.get("context_precision"),
            "ragas_context_recall": ragas.get("context_recall"),
            "ragas_faithfulness": ragas.get("faithfulness"),
        })
    pd.DataFrame(flat_rows).to_csv(output_dir / "results.csv", index=False)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["# Activ One MAX RAG Evaluation", "", "## Overall metrics", ""]
    overall = aggregate["overall"]
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in METRIC_KEYS:
        lines.append(f"| {key.replace('_', ' ')} | {_format(overall.get(key))} |")
    lines.extend(["", "## Per-category metrics", ""])
    lines.append("| Category | N | Page hit@6 | Section hit@6 | Context recall | Correctness | Judge faithfulness | Ragas precision | Ragas recall | Ragas faithfulness | Refusal F1 | False refusal | OOS hallucination | Pass rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for category, values in aggregate["categories"].items():
        lines.append(
            f"| {category} | {values['count']} | {_format(values['page_hit_at_6'])} | "
            f"{_format(values['section_hit_at_6'])} | {_format(values['context_recall'])} | "
            f"{_format(values['correctness'])} | {_format(values['faithfulness'])} | "
            f"{_format(values['ragas_context_precision'])} | {_format(values['ragas_context_recall'])} | "
            f"{_format(values['ragas_faithfulness'])} | "
            f"{_format(values['refusal_f1'])} | {_format(values['false_refusal_rate'])} | "
            f"{_format(values['out_of_scope_hallucination_rate'])} | {_format(values['pass_rate'])} |"
        )

    failures = [row for row in rows if row.get("pass") is False]
    failures.sort(key=lambda row: (
        row.get("judge", {}).get("correctness", 1),
        row.get("judge", {}).get("faithfulness", 1),
        row.get("page_hit_at_6", False),
    ))
    lines.extend(["", "## Top 10 failures", ""])
    if not failures:
        lines.append("No failing items with an evaluated pass/fail outcome.")
    for row in failures[:10]:
        lines.extend([
            f"### {row.get('id')} — {row.get('category')}",
            f"- Query: {row.get('query')}",
            f"- Expected: {row.get('ground_truth_response')}",
            f"- Actual: {row.get('answer')}",
            f"- Retrieved pages: {row.get('retrieved_pages', [])}",
            f"- Judge: {row.get('judge', {}).get('reason', 'not run')}",
            "",
        ])
    wrongly_answered = [
        row for row in rows if row.get("out_of_scope") and not row.get("refused")
    ]
    lines.extend(["## Out-of-scope items wrongly answered", ""])
    if not wrongly_answered:
        lines.append("None.")
    else:
        for row in wrongly_answered:
            lines.append(f"- **{row.get('id')}**: {row.get('query')} — {row.get('answer')}")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = Table(title="RAG evaluation summary")
    table.add_column("Category")
    table.add_column("N", justify="right")
    table.add_column("Page@6", justify="right")
    table.add_column("Context recall", justify="right")
    table.add_column("Refusal F1", justify="right")
    table.add_column("False refusal", justify="right")
    table.add_column("Ragas faithfulness", justify="right")
    table.add_column("Pass rate", justify="right")
    for category, values in [("Overall", overall), *aggregate["categories"].items()]:
        table.add_row(
            category, str(values["count"]), _format(values["page_hit_at_6"]),
            _format(values["context_recall"]), _format(values["refusal_f1"]),
            _format(values["false_refusal_rate"]), _format(values["ragas_faithfulness"]),
            _format(values["pass_rate"]),
        )
    Console().print(table)
    return aggregate
