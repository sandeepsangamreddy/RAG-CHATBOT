"""Extract, clean, chunk, and index the policy PDF."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import pickle
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pymupdf as fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings

LOGGER = logging.getLogger(__name__)
LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
QUOTE_DASH_MAP = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‐": "-", "‑": "-"}
)
HEADING_RE = re.compile(
    r"^\s*((?:Section\s+[A-E]\.?)|(?:[A-E]\.\d+(?:\.\d+){0,4})|"
    r"(?:Annexure\s+(?:I|II|III|IV)\b)|"
    r"(?:\d{1,3}\.\s+[A-Z][A-Za-z][A-Za-z /,'()&-]{2,90}))\s*[:.\-]?\s*(.*)$",
    re.IGNORECASE,
)
SECTION_ID_RE = re.compile(r"\b([A-E]\.\d+(?:\.\d+)*)\b")
DEFINITION_RE = re.compile(r"^\s*(\d{1,3})\.\s+([A-Z][^.\n]{1,90})\.\s*(.*)$")
DEFINITION_MEANS_RE = re.compile(
    r"^\s*(\d{1,3})\.\s+(.+?)\s+(?:means|shall mean|is defined as)\s+(.+)$",
    re.IGNORECASE,
)
DEFINITION_HEADER_RE = re.compile(r"^\s*(\d{1,3})\.\s+([A-Z][A-Za-z0-9 /,'()&-]{2,90})\.?\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*", re.IGNORECASE)

ALIASES: dict[str, tuple[str, ...]] = {
    "hemorrhoids": ("piles",),
    "diabetes": ("sugar",),
    "hypertension": ("bp", "high blood pressure"),
    "caesarean": ("c-section", "c section", "cesarean"),
    "emergency": ("er",),
    "waiting period": ("wait period", "waiting time", "wait time"),
    "ambulance": ("ambulance cover",),
    "ayush": ("ayurveda", "yoga", "unani", "siddha", "homeopathy"),
    "accident": ("bike accident", "road accident", "injury"),
}
ALIAS_TO_CANONICAL = {
    alias.lower(): canonical for canonical, aliases in ALIASES.items() for alias in aliases
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "do", "for", "from", "how", "i",
    "in", "is", "it", "me", "of", "on", "or", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "will", "with", "you", "your",
}


@dataclass
class Chunk:
    """A source-linked text unit written to chunks.jsonl and both indexes."""

    chunk_id: str
    page_start: int
    page_end: int
    section_id: str
    section_title: str
    chunk_type: str
    source_file: str
    original_text: str
    normalized_text: str
    embedding_text: str
    alias: list[str]
    score: float = 0.0
    bm25_overlap: bool = False


def clean_glyphs(text: str) -> str:
    """Repair known PDF glyph defects while preserving quote and dash characters."""
    value = text.translate(LIGATURES).replace("\x8f", "ft")
    return re.sub(r"\bAer\b", "After", re.sub(r"\baer\b", "after", value))


def normalize_for_index(text: str) -> str:
    """Normalize punctuation in a copy used for embeddings and lexical matching."""
    return clean_glyphs(text).translate(QUOTE_DASH_MAP)


def tokenize(text: str, aliases: Iterable[str] = ()) -> list[str]:
    """Tokenize text and append alias tokens used by the BM25 index."""
    terms = [term.lower() for term in TOKEN_RE.findall(normalize_for_index(text))]
    alias_terms = [term.lower() for alias in aliases for term in TOKEN_RE.findall(alias)]
    return terms + alias_terms


def _block_text(block: tuple[Any, ...]) -> str:
    return str(block[4]).strip() if len(block) > 4 else ""


def _line_key(line: str) -> str:
    return re.sub(r"\s+", " ", clean_glyphs(line)).strip().casefold()


def _section_for_heading(line: str, current_id: str, current_title: str) -> tuple[str, str, str] | None:
    match = HEADING_RE.match(line)
    if not match:
        return None
    heading, rest = match.group(1).strip(), match.group(2).strip()
    sec_match = SECTION_ID_RE.search(heading)
    annexure_match = re.search(r"Annexure\s+(I|II|III|IV)", heading, re.IGNORECASE)
    if sec_match:
        section_id = sec_match.group(1).upper()
    elif annexure_match:
        section_id = f"Annexure {annexure_match.group(1).upper()}"
    elif heading.lower().startswith("section"):
        letter = re.search(r"[A-E]", heading, re.IGNORECASE)
        section_id = letter.group(0).upper() if letter else current_id
    else:
        definition = DEFINITION_RE.match(line)
        if definition:
            section_id = f"Definition {definition.group(1)}"
        else:
            return None
    title = rest.rstrip(".: ") if rest else current_title
    if not title and section_id == current_id:
        title = current_title
    return section_id, title, rest


def _definition_parts(text: str, section_id: str) -> tuple[str, str, str] | None:
    """Recognize numbered definitions, including wrapped headings and inline 'means' text."""
    is_definition_section = section_id.upper() == "A" or section_id.lower().startswith("definition")
    means = DEFINITION_MEANS_RE.match(text)
    if means:
        return means.group(1), means.group(2).strip(), means.group(3).strip()
    definition = DEFINITION_RE.match(text)
    if definition and (is_definition_section or re.search(r"\bmeans\b", definition.group(3), re.I)):
        return definition.group(1), definition.group(2).strip(), definition.group(3).strip()
    header = DEFINITION_HEADER_RE.match(text)
    if header and is_definition_section:
        return header.group(1), header.group(2).strip(), ""
    return None


def _aliases_for(text: str) -> list[str]:
    normalized = normalize_for_index(text).lower()
    found: set[str] = set()
    for canonical, aliases in ALIASES.items():
        if canonical in normalized or any(alias in normalized for alias in aliases):
            found.add(canonical)
            found.update(aliases)
    return sorted(found)


def _chunk_id(page_start: int, page_end: int, section_id: str, chunk_type: str, text: str) -> str:
    digest = hashlib.sha1(f"{page_start}:{page_end}:{section_id}:{chunk_type}:{text}".encode()).hexdigest()[:12]
    return f"p{page_start}-{page_end}-{digest}"


def _make_chunk(
    text: str,
    page_start: int,
    page_end: int,
    section_id: str,
    section_title: str,
    chunk_type: str,
    source_file: str,
) -> Chunk:
    original = clean_glyphs(text).strip()
    normalized = normalize_for_index(original)
    header = f"Section {section_id} {section_title} (Page {page_start}):"
    return Chunk(
        chunk_id=_chunk_id(page_start, page_end, section_id, chunk_type, original),
        page_start=page_start,
        page_end=page_end,
        section_id=section_id,
        section_title=section_title,
        chunk_type=chunk_type,
        source_file=source_file,
        original_text=original,
        normalized_text=normalized,
        embedding_text=f"{header}\n{normalized}",
        alias=_aliases_for(normalized),
    )


def _table_text(table: Any) -> list[str]:
    extracted = table.extract()
    if not extracted:
        return []
    rows = [[re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row] for row in extracted]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    header = rows[0]
    rendered = [" | ".join(cell for cell in header if cell)]
    for row in rows[1:]:
        values = [f"{header[i] if i < len(header) and header[i] else f'Column {i + 1}'}: {cell}"
                  for i, cell in enumerate(row) if cell]
        if values:
            rendered.append(" | ".join(values))
    return rendered


def _table_records(
    page: fitz.Page, page_number: int
) -> tuple[list[tuple[float, str]], list[tuple[float, float, float, float]]]:
    records: list[tuple[float, str]] = []
    boxes: list[tuple[float, float, float, float]] = []
    try:
        found = page.find_tables()
    except (AttributeError, RuntimeError) as exc:
        LOGGER.debug("Table detection unavailable on page %s: %s", page_number, exc)
        return records, boxes
    if found is None:
        return records, boxes
    for table in found.tables:
        try:
            rendered = _table_text(table)
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning("Could not extract a table on page %s: %s", page_number, exc)
            continue
        if rendered:
            groups = [rendered]
            if 48 <= page_number <= 56:
                combo_re = re.compile(
                    r"\b([A-Z][A-Za-z/' -]{1,35}\s*\+\s*[A-Z][A-Za-z/' -]{1,35})\b"
                )
                header = rendered[0]
                detected: list[list[str]] = []
                current: list[str] = []
                for row in rendered[1:]:
                    if combo_re.search(row):
                        if current:
                            detected.append(current)
                        current = [header, row]
                    elif current:
                        current.append(row)
                    else:
                        current = [header, row]
                if current:
                    detected.append(current)
                if len(detected) > 1:
                    groups = detected
            for group_index, group in enumerate(groups):
                if group:
                    records.append((float(table.bbox[1]) + group_index * 0.01, "\n".join(group)))
            x0, y0, x1, y1 = (float(value) for value in table.bbox)
            boxes.append((x0, y0, x1, y1))
    return records, boxes


def _inside_box(block: tuple[Any, ...], boxes: list[tuple[float, float, float, float]]) -> bool:
    if len(block) < 4:
        return False
    x0, y0 = float(block[0]), float(block[1])
    return any(x0 >= bx0 - 1 and x0 <= bx2 + 1 and y0 >= by0 - 1 and y0 <= by3 + 1
               for bx0, by0, bx2, by3 in boxes)


def extract_records(pdf_path: Path) -> tuple[int, list[dict[str, Any]]]:
    """Extract unique page text and table rows in reading/column order."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Policy PDF not found: {pdf_path}")
    document = fitz.open(pdf_path)
    raw_pages: list[list[tuple[str, str]]] = []
    raw_keys: list[list[str]] = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        blocks = sorted(page.get_text("blocks"), key=lambda b: (round(float(b[1]), 1), float(b[0])))
        tables, boxes = _table_records(page, page_index + 1)
        use_columns = 10 <= page_index + 1 <= 27
        midpoint = float(page.rect.width) / 2
        left: list[tuple[float, str]] = []
        right: list[tuple[float, str]] = []
        normal: list[tuple[float, str]] = []
        for block in blocks:
            if not isinstance(block, tuple):
                continue
            text = _block_text(block)
            if not text or _inside_box(block, boxes):
                continue
            x, y = float(block[0]), float(block[1])
            for line_number, line in enumerate(text.splitlines()):
                line = line.strip()
                if not line:
                    continue
                line_y = y + line_number * 0.1
                if use_columns:
                    (left if x < midpoint else right).append((line_y, line))
                else:
                    normal.append((line_y, line))
        page_records: list[tuple[str, str]] = []
        if use_columns:
            column_type = "glossary" if page_index + 1 <= 14 else "simplified"
            events = [(y, 0, "LEFT", value) for y, value in left]
            events.extend((y, 0, "TABLE", "TABLE " + row) for y, row in tables)
            events.extend((y, 1, column_type.upper(), value) for y, value in right)
            for _, _, kind, value in sorted(events, key=lambda event: (event[0], event[1])):
                page_records.append((kind, value))
        else:
            events = [(y, 0, value) for y, value in normal]
            events.extend((y, 0, "TABLE " + row) for y, row in tables)
            page_records = [
                (f"{y:.3f}", value)
                for y, _, value in sorted(events, key=lambda event: event[0])
            ]
        raw_pages.append(page_records)
        raw_keys.append([_line_key(text) for _, text in page_records if text.strip()])
    document.close()

    frequencies: collections.Counter[str] = collections.Counter()
    for keys in raw_keys:
        frequencies.update(set(keys))
    repeated = {line for line, count in frequencies.items() if count > 0.30 * len(raw_pages)}
    for known in ("Product Name: Activ One, Product UIN: ADIHLIP24097V012324", "Activ One MAX"):
        repeated.add(_line_key(known))
    output: list[dict[str, Any]] = []
    for page_number, records in enumerate(raw_pages, start=1):
        for source_kind, text in records:
            if _line_key(text) in repeated:
                continue
            output.append({"page": page_number, "kind": source_kind, "text": clean_glyphs(text)})
    return len(raw_pages), output


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", re.sub(r"\s+", " ", text).strip())
    return [part.strip() for part in parts if part.strip()]


def _overlap_tail(sentences: list[str], target_words: int) -> list[str]:
    """Return whole trailing sentences approximating the configured overlap."""
    tail: list[str] = []
    count = 0
    for sentence in reversed(sentences):
        tail.insert(0, sentence)
        count += len(TOKEN_RE.findall(sentence))
        if count >= target_words:
            break
    return tail


def _chunk_records(records: list[dict[str, Any]], source_file: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    section_id, section_title = "General", "Policy wording"
    buffers: dict[tuple[str, str, str], list[str]] = {}
    buffer_words: dict[tuple[str, str, str], int] = {}
    buffer_pages: dict[tuple[str, str, str], tuple[int, int]] = {}
    pending_definition: dict[str, Any] | None = None

    def flush_definition() -> None:
        nonlocal pending_definition
        if pending_definition is None:
            return
        number = pending_definition["number"]
        term = pending_definition["term"]
        text = f"{number}. {term}. {' '.join(pending_definition['body'])}".strip()
        chunks.append(_make_chunk(
            text, pending_definition["page_start"], pending_definition["page_end"],
            f"Definition {number}", term, "definition", source_file,
        ))
        pending_definition = None

    def flush(key: tuple[str, str, str]) -> None:
        text = " ".join(buffers.pop(key, [])).strip()
        buffer_words.pop(key, None)
        pages = buffer_pages.pop(key, None)
        if text:
            sid, title, kind = key
            page_start, page_end = pages or (1, 1)
            chunks.append(_make_chunk(text, page_start, page_end, sid, title, kind, source_file))

    for record in records:
        page = int(record["page"])
        kind = str(record["kind"])
        text = str(record["text"]).strip()
        if not text:
            continue
        if kind.startswith("TABLE ") or text.startswith("TABLE "):
            flush_definition()
            flush_all = list(buffers)
            for key in flush_all:
                flush(key)
            table_body = text.removeprefix("TABLE ").strip()
            section_guess = SECTION_ID_RE.search(table_body)
            if section_guess:
                section_id = section_guess.group(1)
            if 10 <= page <= 11:
                section_id, section_title, chunk_type = "D.1.2", "Specific Disease Waiting Period", "table"
            elif 42 <= page <= 43:
                section_id, section_title, chunk_type = "Annexure I", "Non-medical Expenses", "annexure"
            elif 44 <= page <= 45:
                section_id, section_title, chunk_type = "Annexure II", "Annexure II", "annexure"
            elif 46 <= page <= 47:
                section_id, section_title, chunk_type = "Annexure III", "Product Benefit Table", "annexure"
            elif 48 <= page <= 56:
                section_id, section_title, chunk_type = "Annexure IV", "Chronic Management Program", "annexure"
            else:
                chunk_type = "table"
            rows = [row for row in table_body.splitlines() if row.strip()]
            table_parts: list[str] = [table_body]
            if rows and page < 48:
                header = rows[0]
                table_parts = []
                group: list[str] = []
                group_words = len(TOKEN_RE.findall(header))
                for row in rows[1:]:
                    row_words = len(TOKEN_RE.findall(row))
                    if group and group_words + row_words > settings.chunk_max_tokens:
                        table_parts.append("\n".join([header, *group]))
                        group = []
                        group_words = len(TOKEN_RE.findall(header))
                    group.append(row)
                    group_words += row_words
                if group:
                    table_parts.append("\n".join([header, *group]))
                if not table_parts:
                    table_parts = [table_body]
            for table_part in table_parts:
                chunks.append(_make_chunk(
                    table_part, page, page, section_id, section_title, chunk_type, source_file
                ))
            continue
        chunk_type = (
            kind.lower() if kind in {"GLOSSARY", "SIMPLIFIED"}
            else "annexure" if section_id.startswith("Annexure")
            else "clause"
        )
        definition = _definition_parts(text, section_id)
        heading = _section_for_heading(text, section_id, section_title)
        if pending_definition is not None and (definition or heading):
            flush_definition()
        if definition:
            for key in list(buffers):
                flush(key)
            pending_definition = {
                "number": definition[0],
                "term": definition[1],
                "body": [definition[2]] if definition[2] else [],
                "page_start": page,
                "page_end": page,
            }
            section_id = f"Definition {definition[0]}"
            section_title = definition[1]
            continue
        if pending_definition is not None:
            pending_definition["body"].append(text)
            pending_definition["page_end"] = page
            continue
        if heading:
            for key in list(buffers):
                flush(key)
            section_id, new_title, remainder = heading
            if new_title:
                section_title = new_title
            text = remainder
            if not text:
                continue
        key = (section_id, section_title, chunk_type)
        if chunk_type in {"glossary", "simplified"}:
            for sentence in _split_sentences(text):
                count = len(TOKEN_RE.findall(sentence))
                existing = buffer_words.get(key, 0)
                if existing and existing + count > settings.chunk_max_tokens:
                    flush(key)
                    existing = 0
                buffers.setdefault(key, []).append(sentence)
                buffer_words[key] = existing + count
                start_page, end_page = buffer_pages.get(key, (page, page))
                buffer_pages[key] = (min(start_page, page), max(end_page, page))
            continue
        for sentence in _split_sentences(text):
            count = len(TOKEN_RE.findall(sentence))
            existing = buffer_words.get(key, 0)
            if existing and existing + count > settings.chunk_max_tokens:
                previous = list(buffers.get(key, []))
                previous_pages = buffer_pages.get(key, (page, page))
                overlap = _overlap_tail(
                    previous, max(1, round(settings.chunk_max_tokens * settings.chunk_overlap))
                )
                flush(key)
                existing = 0
                overlap_words = sum(len(TOKEN_RE.findall(part)) for part in overlap)
                if count <= settings.chunk_max_tokens and overlap_words + count <= settings.chunk_max_tokens:
                    buffers[key] = overlap
                    buffer_words[key] = overlap_words
                    buffer_pages[key] = (previous_pages[1], page)
                    existing = overlap_words
            buffers.setdefault(key, []).append(sentence)
            buffer_words[key] = existing + count
            start_page, end_page = buffer_pages.get(key, (page, page))
            buffer_pages[key] = (min(start_page, page), max(end_page, page))
    for key in list(buffers):
        flush(key)
    flush_definition()
    id_counts: collections.Counter[str] = collections.Counter()
    for chunk in chunks:
        id_counts[chunk.chunk_id] += 1
        if id_counts[chunk.chunk_id] > 1:
            chunk.chunk_id = f"{chunk.chunk_id}-{id_counts[chunk.chunk_id]}"
    return chunks


def build_indexes(chunks: list[Chunk], settings_obj: Any = settings) -> None:
    """Persist JSONL inspection data, the Chroma dense index, and BM25 state."""
    settings_obj.index_dir.mkdir(parents=True, exist_ok=True)
    with settings_obj.chunks_path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    corpus = [tokenize(chunk.normalized_text, chunk.alias) for chunk in chunks]
    with settings_obj.bm25_path.open("wb") as output:
        pickle.dump({"bm25": BM25Okapi(corpus), "chunk_ids": [c.chunk_id for c in chunks],
                     "corpus": corpus}, output)

    import chromadb

    client = chromadb.PersistentClient(path=str(settings_obj.chroma_dir))
    collection_name = "activ_one_max_policy"
    try:
        client.delete_collection(collection_name)
    except Exception as exc:
        if "does not exist" not in str(exc).lower():
            raise
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    model = SentenceTransformer(settings_obj.embed_model)
    batch_size = settings_obj.embedding_batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        embeddings = model.encode([c.embedding_text for c in batch], normalize_embeddings=True,
                                  show_progress_bar=False).tolist()
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.embedding_text for c in batch],
            metadatas=[{
                "page_start": c.page_start, "page_end": c.page_end,
                "section_id": c.section_id, "section_title": c.section_title,
                "chunk_type": c.chunk_type, "source_file": c.source_file,
                "alias": json.dumps(c.alias), "original_text": c.original_text,
                "normalized_text": c.normalized_text,
            } for c in batch],
        )


def ingest(pdf_path: Path, rebuild: bool = False) -> tuple[int, list[Chunk]]:
    """Run PDF extraction and indexing, returning the page and chunk totals."""
    if not rebuild and settings.chunks_path.exists() and settings.bm25_path.exists():
        raise FileExistsError("An index already exists; pass --rebuild to replace it.")
    pages, records = extract_records(pdf_path)
    chunks = _chunk_records(records, pdf_path.name)
    if not chunks:
        raise ValueError(f"No usable text was extracted from {pdf_path}")
    build_indexes(chunks)
    return pages, chunks


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for PDF ingestion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=settings.pdf_path)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        pages, chunks = ingest(args.pdf, rebuild=args.rebuild)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    counts = collections.Counter(chunk.chunk_type for chunk in chunks)
    print(f"Pages: {pages}\nChunks: {len(chunks)}")
    for chunk_type, count in sorted(counts.items()):
        print(f"  {chunk_type}: {count}")
    print("\nRandom sample chunks:")
    for chunk in random.Random(settings.random_seed).sample(chunks, min(5, len(chunks))):
        print(f"\n[{chunk.section_id}, p.{chunk.page_start}, {chunk.chunk_type}] {chunk.original_text[:500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
