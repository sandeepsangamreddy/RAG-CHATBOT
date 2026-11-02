"""Optional Ragas metrics for retrieval precision/recall and response faithfulness."""

from __future__ import annotations

import threading
from typing import Any

from anthropic import Anthropic

from src.config import settings


class RagasEvaluator:
    """Score in-scope RAG results with Ragas using the configured Anthropic model."""

    def __init__(self, settings_obj: Any = settings) -> None:
        self.settings = settings_obj
        if not settings_obj.anthropic_api_key:
            raise ValueError("Ragas evaluation requires ANTHROPIC_API_KEY in .env.")
        self._thread_local = threading.local()

    def _metrics(self) -> tuple[Any, Any, Any]:
        """Create per-thread Ragas metrics and their Anthropic evaluator."""
        cached = getattr(self._thread_local, "metrics", None)
        if cached is not None:
            return cached

        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

        anthropic_client = Anthropic(api_key=self.settings.anthropic_api_key)
        evaluator_llm = llm_factory(
            self.settings.judge_model,
            provider="anthropic",
            client=anthropic_client,
            temperature=None,
            top_p=None,
        )
        cached = (
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm),
        )
        self._thread_local.metrics = cached
        return cached

    def score(
        self,
        *,
        question: str,
        reference: str,
        response: str,
        contexts: list[str],
    ) -> dict[str, float]:
        """Return Ragas context precision, context recall, and faithfulness scores."""
        if not contexts:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
            }
        precision_metric, recall_metric, faithfulness_metric = self._metrics()
        common = {"user_input": question, "retrieved_contexts": contexts}
        precision = precision_metric.score(**common, reference=reference)
        recall = recall_metric.score(**common, reference=reference)
        faithfulness = faithfulness_metric.score(**common, response=response)
        return {
            "context_precision": float(precision.value),
            "context_recall": float(recall.value),
            "faithfulness": float(faithfulness.value),
        }
