"""Integration checks for extraction and section-aware policy chunks."""

from __future__ import annotations

import re

import pytest

from src.config import settings
from src.ingest import (
    Chunk,
    _chunk_records,
    clean_glyphs,
    extract_records,
    normalize_for_index,
)


def test_known_glyphs_and_index_punctuation_are_normalized() -> None:
    cleaned = clean_glyphs("a\x8fer Aer le\x8f ﬁ ﬂ")
    assert cleaned == "after After left fi fl"
    assert normalize_for_index("“left”—‘right’") == '"left"-\'right\''


def test_section_and_column_records_remain_distinct() -> None:
    chunks = _chunk_records([
        {"page": 10, "kind": "LEFT", "text": "D.1.2 Specific diseases"},
        {"page": 10, "kind": "LEFT", "text": "The clause column states this."},
        {"page": 10, "kind": "GLOSSARY", "text": "Glossary definition text."},
    ], "synthetic.pdf")
    clause = next(chunk for chunk in chunks if chunk.chunk_type == "clause")
    glossary = next(chunk for chunk in chunks if chunk.chunk_type == "glossary")
    assert clause.section_id == "D.1.2"
    assert "Glossary definition" not in clause.original_text
    assert glossary.section_id == "D.1.2"


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    if not settings.pdf_path.is_file():
        pytest.skip(f"Policy PDF is not present: {settings.pdf_path}")
    _, records = extract_records(settings.pdf_path)
    return _chunk_records(records, settings.pdf_path.name)


def test_indexed_text_has_no_bad_glyphs(chunks: list[Chunk]) -> None:
    text = "\n".join(chunk.embedding_text for chunk in chunks)
    assert not any(character in text for character in ("ﬁ", "ﬂ", "ﬀ", "ﬃ", "ﬄ", "\x8f"))


def test_after_is_repaired(chunks: list[Chunk]) -> None:
    text = "\n".join(chunk.normalized_text for chunk in chunks)
    assert re.search(r"\bafter\b", text, flags=re.IGNORECASE)
    assert not re.search(r"\baer\b", text, flags=re.IGNORECASE)


def test_specific_disease_table_rows_are_preserved(chunks: list[Chunk]) -> None:
    assert any("Cataract" in chunk.original_text and "Cataract Surgery" in chunk.original_text
               for chunk in chunks)


def test_repeated_product_footer_is_removed(chunks: list[Chunk]) -> None:
    text = "\n".join(chunk.embedding_text for chunk in chunks)
    assert "Product UIN" not in text


def test_annexure_iv_has_condition_chunks_with_teleconsult(chunks: list[Chunk]) -> None:
    chronic = [
        chunk for chunk in chunks
        if chunk.section_id == "Annexure IV" and "Teleconsult" in chunk.original_text
    ]
    assert len(chronic) >= 40


def test_required_chunk_metadata_is_present(chunks: list[Chunk]) -> None:
    assert chunks
    assert all(
        chunk.page_start > 0 and chunk.section_id and chunk.chunk_type
        for chunk in chunks
    )


def test_glossary_column_is_not_mixed_into_clause_chunks(chunks: list[Chunk]) -> None:
    glossary_texts = [chunk.original_text for chunk in chunks if chunk.chunk_type == "glossary"]
    clause_texts = [chunk.original_text for chunk in chunks if chunk.chunk_type == "clause"]
    assert glossary_texts
    for glossary in glossary_texts:
        normalized = re.sub(r"\s+", " ", glossary).strip()
        assert not any(normalized in re.sub(r"\s+", " ", clause).strip() for clause in clause_texts)
