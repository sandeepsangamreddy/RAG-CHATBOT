"""Hybrid dense/BM25 retrieval with reciprocal-rank fusion and optional reranking."""

from __future__ import annotations

import json
import logging
import pickle
import re
import threading
from dataclasses import dataclass, replace
from typing import Any

from src.config import settings
from src.ingest import ALIASES, STOPWORDS, Chunk, normalize_for_index, tokenize

LOGGER = logging.getLogger(__name__)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class _Candidate:
    """Mutable retrieval state for one policy chunk."""

    chunk: Chunk
    rrf_score: float
    dense_similarity: float = 0.0
    bm25_overlap: bool = False


def expand_query(query: str) -> str:
    """Expand known colloquial policy terms without modifying the dense query."""
    normalized = normalize_for_index(query).lower()
    expansions: list[str] = []
    for canonical, aliases in ALIASES.items():
        if canonical in normalized or any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            expansions.append(canonical)
            expansions.extend(aliases)
    if "bike accident" in normalized:
        expansions.extend(("accident", "injury"))
    if "c section" in normalized or "c-section" in normalized:
        expansions.extend(("caesarean", "cesarean"))
    return " ".join([query, *dict.fromkeys(expansions)])


class HybridRetriever:
    """Retrieve policy chunks from persistent Chroma and BM25 indexes."""

    def __init__(self, settings_obj: Any = settings) -> None:
        self.settings = settings_obj
        self._lock = threading.RLock()
        self._confidence_state = threading.local()
        if not settings_obj.chunks_path.is_file() or not settings_obj.bm25_path.is_file():
            raise FileNotFoundError("Policy indexes are missing; run `python -m src.ingest --rebuild` first.")
        with settings_obj.chunks_path.open(encoding="utf-8") as source:
            self.chunks = [Chunk(**json.loads(line)) for line in source if line.strip()]
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        with settings_obj.bm25_path.open("rb") as source:
            sparse_state = pickle.load(source)
        self.bm25 = sparse_state["bm25"]
        self.bm25_ids: list[str] = sparse_state["chunk_ids"]
        self.corpus: list[list[str]] = sparse_state["corpus"]
        if not self.chunks or len(self.bm25_ids) != len(self.chunks):
            raise ValueError("The policy indexes are empty or inconsistent; rebuild the index.")

        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(settings_obj.chroma_dir))
        self.collection = client.get_collection("activ_one_max_policy")
        self.embedder = SentenceTransformer(settings_obj.embed_model)
        self.reranker = None
        if settings_obj.rerank:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(settings_obj.rerank_model)
        self.last_confidence = 0.0

    @property
    def last_confidence(self) -> float:
        """Return the retrieval confidence belonging to the current calling thread."""
        return float(getattr(self._confidence_state, "value", 0.0))

    @last_confidence.setter
    def last_confidence(self, value: float) -> None:
        self._confidence_state.value = value

    @staticmethod
    def _section_boost(query: str, section_id: str) -> bool:
        requested = re.findall(r"\b([A-E]\.\d+(?:\.\d+)*)\b", query, flags=re.IGNORECASE)
        return any(section_id.upper().startswith(value.upper()) for value in requested)

    def retrieve(self, query: str, k: int = 6) -> list[Chunk]:
        """Return the top-k chunks and expose calibrated confidence on the instance."""
        if not query.strip():
            self.last_confidence = 0.0
            return []
        with self._lock:
            dense_query = self.embedder.encode(
                [QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False
            )[0].tolist()
        expanded = expand_query(query)
        query_tokens = tokenize(expanded)
        scores = self.bm25.get_scores(query_tokens)
        sparse_order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
        sparse_order = [idx for idx in sparse_order if float(scores[idx]) > 0][:self.settings.bm25_candidates]
        collection_count = self.collection.count()
        if not collection_count:
            raise ValueError("The Chroma collection is empty; rebuild the policy index.")
        dense_count = min(self.settings.dense_candidates, collection_count)
        with self._lock:
            dense_result = self.collection.query(
                query_embeddings=[dense_query],
                n_results=max(1, dense_count),
                include=["distances", "metadatas"],
            )
        result_ids = dense_result.get("ids")
        result_distances = dense_result.get("distances")
        if not result_ids or not result_distances:
            raise RuntimeError("Chroma returned no candidate IDs or distances for the query.")
        dense_ids = result_ids[0]
        dense_distances = result_distances[0]
        candidates: dict[str, _Candidate] = {}
        for rank, (chunk_id, distance) in enumerate(zip(dense_ids, dense_distances), start=1):
            chunk = self.by_id.get(chunk_id)
            if chunk is None:
                continue
            similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
            candidates[chunk_id] = _Candidate(
                replace(chunk), 1 / (self.settings.rrf_k + rank), similarity
            )
        for rank, idx in enumerate(sparse_order, start=1):
            chunk_id = self.bm25_ids[idx]
            chunk = self.by_id.get(chunk_id)
            if chunk is None:
                continue
            candidate = candidates.setdefault(chunk_id, _Candidate(replace(chunk), 0.0))
            candidate.rrf_score += 1 / (self.settings.rrf_k + rank)
            doc_tokens = set(self.corpus[idx])
            candidate.bm25_overlap = bool(
                set(query_tokens).intersection(doc_tokens) - STOPWORDS
            )

        query_lower = query.lower()
        for candidate in candidates.values():
            chunk = candidate.chunk
            score = candidate.rrf_score
            if any(term in query_lower for term in ("annexure", "table", "benefit table")):
                if chunk.chunk_type in {"table", "annexure"}:
                    score *= self.settings.table_retrieval_boost
            if self._section_boost(query, chunk.section_id):
                score *= self.settings.section_retrieval_boost
            if chunk.chunk_type in {"glossary", "simplified"}:
                score *= self.settings.glossary_retrieval_weight
            candidate.rrf_score = score

        selected = sorted(candidates.values(), key=lambda c: (-c.rrf_score, c.chunk.chunk_id))[
            :self.settings.rrf_candidates
        ]
        if self.reranker is not None and selected:
            pairs = [(query, c.chunk.normalized_text) for c in selected]
            with self._lock:
                rerank_scores = self.reranker.predict(pairs)
            ranked = sorted(
                ((float(score), candidate) for score, candidate in zip(rerank_scores, selected)),
                key=lambda pair: (-pair[0], pair[1].chunk.chunk_id),
            )
            selected = [candidate for _, candidate in ranked]
            for score, candidate in ranked:
                candidate.rrf_score = score
        best_similarity = max((candidate.dense_similarity for candidate in selected), default=-1.0)
        lexical_signal = any(candidate.bm25_overlap for candidate in selected)
        dense_confidence = max(
            0.0,
            min(1.0, (best_similarity - self.settings.dense_confidence_floor)
                / self.settings.dense_confidence_scale),
        )
        self.last_confidence = max(
            dense_confidence, self.settings.lexical_confidence if lexical_signal else 0.0
        )
        results: list[Chunk] = []
        for candidate in selected[:k]:
            candidate.chunk.score = float(candidate.rrf_score)
            candidate.chunk.bm25_overlap = candidate.bm25_overlap
            results.append(candidate.chunk)
        return results
