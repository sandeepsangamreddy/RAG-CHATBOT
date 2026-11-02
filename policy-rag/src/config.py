"""Application settings loaded from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Typed settings shared by ingestion, retrieval, generation, and evaluation."""

    project_root: Path = PROJECT_ROOT
    pdf_path: Path = PROJECT_ROOT / "data" / "Active_One_Max_policy_wording-160925.pdf.pdf"
    dataset_path: Path = PROJECT_ROOT / "data" / "activ_one_max_golden_dataset.json"
    index_dir: Path = PROJECT_ROOT / "index"
    chroma_dir: Path = PROJECT_ROOT / "index" / "chroma"
    chunks_path: Path = PROJECT_ROOT / "index" / "chunks.jsonl"
    bm25_path: Path = PROJECT_ROOT / "index" / "bm25.pkl"
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    judge_model: str = os.getenv("JUDGE_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"))
    rerank: bool = _env_bool("RERANK", False)
    top_k: int = int(os.getenv("TOP_K", "6"))
    refusal_threshold: float = float(os.getenv("REFUSAL_THRESHOLD", "0.15"))
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    max_tokens: int = 1000
    rewrite_max_tokens: int = 120
    judge_max_tokens: int = 300
    retry_attempts: int = 3
    retry_base_delay_seconds: int = 1
    dense_candidates: int = 20
    bm25_candidates: int = 20
    rrf_k: int = 60
    rrf_candidates: int = 10
    dense_confidence_floor: float = 0.35
    dense_confidence_scale: float = 0.55
    lexical_confidence: float = 0.16
    table_retrieval_boost: float = 1.2
    section_retrieval_boost: float = 1.25
    glossary_retrieval_weight: float = 0.7
    embedding_batch_size: int = 64
    chunk_min_tokens: int = 250
    chunk_max_tokens: int = 450
    chunk_overlap: float = 0.15
    eval_workers: int = 4
    random_seed: int = 17


settings = Settings()
