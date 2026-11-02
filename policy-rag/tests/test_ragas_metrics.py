"""Unit tests for optional Ragas evaluation result handling."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from eval.metrics import aggregate_metrics
from eval.ragas_metrics import RagasEvaluator
from src.config import settings


class FakeMetric:
    """Capture Ragas-style score inputs and return a deterministic score."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.inputs: dict[str, object] = {}

    def score(self, **kwargs: object) -> SimpleNamespace:
        self.inputs = kwargs
        return SimpleNamespace(value=self.value)


def test_ragas_anthropic_metrics_initialize_without_sampling_parameters() -> None:
    evaluator = RagasEvaluator(replace(settings, anthropic_api_key="test-key"))
    metrics = evaluator._metrics()

    assert [type(metric).__name__ for metric in metrics] == [
        "ContextPrecision",
        "ContextRecall",
        "Faithfulness",
    ]


def test_ragas_scores_use_references_contexts_and_answer() -> None:
    evaluator = RagasEvaluator.__new__(RagasEvaluator)
    precision = FakeMetric(0.8)
    recall = FakeMetric(0.7)
    faithfulness = FakeMetric(0.9)
    evaluator._metrics = lambda: (precision, recall, faithfulness)

    scores = evaluator.score(
        question="What is the waiting period?",
        reference="The waiting period is 30 days.",
        response="It is 30 days.",
        contexts=["The initial waiting period is 30 days."],
    )

    assert scores == {
        "context_precision": 0.8,
        "context_recall": 0.7,
        "faithfulness": 0.9,
    }
    assert precision.inputs["reference"] == "The waiting period is 30 days."
    assert recall.inputs["retrieved_contexts"] == ["The initial waiting period is 30 days."]
    assert faithfulness.inputs["response"] == "It is 30 days."


def test_ragas_aggregates_ignore_items_without_ragas_scores() -> None:
    aggregate = aggregate_metrics([
        {
            "category": "Direct Factual / Lookup",
            "out_of_scope": False,
            "refused": False,
            "ragas": {
                "context_precision": 0.8,
                "context_recall": 0.6,
                "faithfulness": 1.0,
            },
        },
        {
            "category": "Out-of-Scope / Unanswerable",
            "out_of_scope": True,
            "refused": True,
            "ragas": None,
        },
    ])

    overall = aggregate["overall"]
    assert overall["ragas_context_precision"] == 0.8
    assert overall["ragas_context_recall"] == 0.6
    assert overall["ragas_faithfulness"] == 1.0
    assert overall["ragas_scored_items"] == 1
