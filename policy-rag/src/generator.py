"""Claude-backed generation with source citations and a retrieval refusal gate."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from anthropic import APIStatusError, Anthropic, RateLimitError

from src.config import settings
from src.ingest import Chunk
from src.prompts import ANSWER_TEMPLATE, REWRITE_PROMPT, SYSTEM_PROMPT

LOGGER = logging.getLogger(__name__)
REFUSAL = "I'm sorry, but that information is not available in this document."
CITATION_RE = re.compile(r"\[(?:Section\s+)?([^,\]]+),\s*p\.?\s*(\d+)\]", re.IGNORECASE)


class AnswerGenerator:
    """Generate concise document-grounded answers using the Anthropic SDK."""

    def __init__(self, settings_obj: Any = settings, client: Any | None = None) -> None:
        self.settings = settings_obj
        if client is not None:
            self.client = client
        else:
            if not settings_obj.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is missing. Add it to policy-rag/.env before asking questions.")
            self.client = Anthropic(api_key=settings_obj.anthropic_api_key)

    @staticmethod
    def _history_text(history: list[dict[str, str]] | None) -> str:
        if not history:
            return "(no previous turns)"
        turns = history[-8:]
        return "\n".join(
            f"{turn.get('role', 'user').title()}: {turn.get('content', '')}" for turn in turns
        )

    def _message(self, *, model: str, max_tokens: int, system: str, content: str) -> str:
        """Call Claude with bounded retries for rate-limit and overload responses."""
        for attempt in range(self.settings.retry_attempts):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=self.settings.temperature,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                )
                return "".join(
                    block.text for block in response.content if getattr(block, "type", None) == "text"
                ).strip()
            except RateLimitError:
                if attempt == self.settings.retry_attempts - 1:
                    raise
                delay = self.settings.retry_base_delay_seconds * (2 ** attempt)
                LOGGER.warning("Claude rate limited; retrying in %s seconds.", delay)
                time.sleep(delay)
            except APIStatusError as exc:
                if exc.status_code not in {429, 500, 502, 503, 529} or attempt == self.settings.retry_attempts - 1:
                    raise
                delay = self.settings.retry_base_delay_seconds * (2 ** attempt)
                LOGGER.warning("Claude returned HTTP %s; retrying in %s seconds.", exc.status_code, delay)
                time.sleep(delay)
        raise RuntimeError("Claude request exhausted its retry attempts.")

    def rewrite_query(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        """Resolve a follow-up into a standalone retrieval question when history exists."""
        if not history:
            return query.strip()
        prompt = REWRITE_PROMPT.format(history=self._history_text(history), query=query)
        rewritten = self._message(
            model=self.settings.claude_model,
            max_tokens=self.settings.rewrite_max_tokens,
            system="Rewrite the user's question as requested. Return only the question.",
            content=prompt,
        )
        return rewritten.strip() or query.strip()

    @staticmethod
    def _context(chunks: list[Chunk]) -> str:
        ordered = sorted(
            enumerate(chunks, start=1),
            key=lambda item: (item[1].chunk_type in {"glossary", "simplified"}, item[0]),
        )
        return "\n\n".join(
            f"[{index}] (Section {chunk.section_id}, p.{chunk.page_start}"
            f"{f'-{chunk.page_end}' if chunk.page_end != chunk.page_start else ''}, "
            f"type={chunk.chunk_type})\n{chunk.original_text}"
            for index, chunk in ordered
        )

    @staticmethod
    def _citations(answer: str, chunks: list[Chunk]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for section_id, page_text in CITATION_RE.findall(answer):
            page = int(page_text)
            section = section_id.strip()
            matching = next(
                (chunk for chunk in chunks
                 if chunk.section_id.casefold() == section.casefold()
                 and chunk.page_start <= page <= chunk.page_end),
                None,
            )
            citation = (section, page)
            if matching and citation not in seen:
                citations.append({
                    "section_id": matching.section_id,
                    "page": page,
                    "snippet": matching.original_text[:300],
                })
                seen.add(citation)
        return citations

    def generate(
        self,
        query: str,
        chunks: list[Chunk],
        history: list[dict[str, str]] | None = None,
        retrieval_confidence: float = 1.0,
        rewritten_query: str | None = None,
    ) -> dict[str, Any]:
        """Return the answer, citations, retrieved metadata, and refusal status."""
        standalone_query = rewritten_query or self.rewrite_query(query, history)
        retrieved = [
            {
                "chunk_id": chunk.chunk_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "chunk_type": chunk.chunk_type,
                "score": chunk.score,
                "original_text": chunk.original_text,
                "bm25_overlap": chunk.bm25_overlap,
            }
            for chunk in chunks
        ]
        if retrieval_confidence < self.settings.refusal_threshold and not any(
            chunk.bm25_overlap for chunk in chunks
        ):
            return {
                "answer": REFUSAL, "citations": [], "retrieved": retrieved,
                "refused": True, "rewritten_query": standalone_query,
            }
        history_text = self._history_text(history)
        content = ANSWER_TEMPLATE.format(
            query=standalone_query,
            history=history_text,
            context=self._context(chunks) if chunks else "(no relevant context passages)",
        )
        answer = self._message(
            model=self.settings.claude_model,
            max_tokens=self.settings.max_tokens,
            system=SYSTEM_PROMPT,
            content=content,
        )
        if not answer:
            raise RuntimeError("Claude returned an empty answer.")
        return {
            "answer": answer,
            "citations": self._citations(answer, chunks),
            "retrieved": retrieved,
            "refused": answer.lstrip().startswith(REFUSAL),
            "rewritten_query": standalone_query,
        }
