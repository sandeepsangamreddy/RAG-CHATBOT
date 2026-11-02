"""Pipeline orchestration and a command-line question interface."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Any

from src.config import settings
from src.generator import AnswerGenerator
from src.retriever import HybridRetriever

LOGGER = logging.getLogger(__name__)


class RAGPipeline:
    """Lazily load the retrieval/generation components and answer user questions."""

    def __init__(self, settings_obj: Any = settings) -> None:
        self.settings = settings_obj
        self._retriever: HybridRetriever | None = None
        self._generator: AnswerGenerator | None = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._retriever is not None and self._generator is not None:
            return
        with self._load_lock:
            if not self.settings.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is missing. Copy .env.example to .env and set "
                    "ANTHROPIC_API_KEY before asking questions."
                )
            if self._retriever is None:
                self._retriever = HybridRetriever(self.settings)
            if self._generator is None:
                self._generator = AnswerGenerator(self.settings)

    def ask(self, query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Retrieve grounded passages and generate one answer."""
        if not query.strip():
            raise ValueError("Question must not be empty.")
        self._load()
        if self._retriever is None or self._generator is None:
            raise RuntimeError("The RAG pipeline failed to initialize its components.")
        rewritten_query = self._generator.rewrite_query(query, history)
        chunks = self._retriever.retrieve(rewritten_query, k=self.settings.top_k)
        return self._generator.generate(
            query=query,
            chunks=chunks,
            history=history,
            retrieval_confidence=self._retriever.last_confidence,
            rewritten_query=rewritten_query,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for one question."""
    parser = argparse.ArgumentParser(description="Ask the Activ One MAX policy RAG chatbot.")
    parser.add_argument("--ask", required=True, help="Question about the policy wording")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = RAGPipeline().ask(args.ask)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
        for citation in result["citations"]:
            print(f"- Section {citation['section_id']}, p.{citation['page']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
