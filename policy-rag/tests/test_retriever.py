"""Retrieval smoke tests against known policy topics and page ranges."""

from __future__ import annotations

import pytest

from src.config import settings
from src.retriever import HybridRetriever, expand_query

EXPECTED_PAGES = [
    ("waiting period for cataract", {10, 11}),
    ("grace period", set(range(15, 28))),
    ("compassionate visit limit", {40}),
    ("free look period", set(range(15, 28))),
    ("super reload first claim", set(range(1, 10))),
    ("piles surgery", {10, 11}),
    ("pre hospitalization days", set(range(1, 10))),
    ("ombudsman Kerala", set(range(15, 28))),
]


def test_slang_expansion_adds_canonical_terms() -> None:
    assert "hemorrhoids" in expand_query("piles surgery").lower()
    assert "diabetes" in expand_query("sugar").lower()
    assert "caesarean" in expand_query("c section").lower()


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    if not settings.chunks_path.is_file() or not settings.bm25_path.is_file():
        pytest.skip("Policy index has not been built.")
    try:
        return HybridRetriever()
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"Retrieval dependencies/index unavailable: {exc}")
    raise RuntimeError("Retriever fixture did not initialize.")


@pytest.mark.parametrize(("query", "expected_pages"), EXPECTED_PAGES)
def test_query_retrieves_expected_page(
    retriever: HybridRetriever, query: str, expected_pages: set[int]
) -> None:
    chunks = retriever.retrieve(query, k=6)
    retrieved_pages = {
        page for chunk in chunks for page in range(chunk.page_start, chunk.page_end + 1)
    }
    assert expected_pages.intersection(retrieved_pages), (
        f"{query!r} did not retrieve any expected page; got {sorted(retrieved_pages)}"
    )
