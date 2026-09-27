"""Runtime configuration, read from environment variables (and `.env` if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("CLEARANCE_DATA_DIR", "data")))

    # LLM
    llm_provider: str = field(default_factory=lambda: _env("CLEARANCE_LLM_PROVIDER", "anthropic"))
    small_model: str = field(default_factory=lambda: _env("CLEARANCE_SMALL_MODEL", "claude-haiku-4-5"))
    large_model: str = field(default_factory=lambda: _env("CLEARANCE_LARGE_MODEL", "claude-opus-5"))
    judge_model: str = field(default_factory=lambda: _env("CLEARANCE_JUDGE_MODEL", "claude-sonnet-5"))
    large_effort: str = field(default_factory=lambda: _env("CLEARANCE_LARGE_EFFORT", "medium"))
    router_threshold: float = field(
        default_factory=lambda: float(_env("CLEARANCE_ROUTER_THRESHOLD", "0.5"))
    )

    # Embeddings
    embedder: str = field(default_factory=lambda: _env("CLEARANCE_EMBEDDER", "local"))
    embed_model: str = field(
        default_factory=lambda: _env("CLEARANCE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )
    reranker_model: str = field(
        default_factory=lambda: _env(
            "CLEARANCE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # Retrieval
    retrieval_mode: str = field(default_factory=lambda: _env("CLEARANCE_RETRIEVAL_MODE", "hybrid"))
    use_reranker: bool = field(default_factory=lambda: _bool("CLEARANCE_USE_RERANKER", False))
    use_graph: bool = field(default_factory=lambda: _bool("CLEARANCE_USE_GRAPH", True))
    top_k: int = field(default_factory=lambda: int(_env("CLEARANCE_TOP_K", "8")))
    candidate_k: int = 50

    # Cache
    cache_mode: str = field(default_factory=lambda: _env("CLEARANCE_CACHE_MODE", "acl_aware"))
    cache_threshold: float = field(
        default_factory=lambda: float(_env("CLEARANCE_CACHE_THRESHOLD", "0.92"))
    )
    # "dependencies": ACL-check only the chunks the answer depends on (shares far more often).
    # "context": ACL-check every chunk the model was shown (strictest, rarely shares).
    cache_provenance: str = field(
        default_factory=lambda: _env("CLEARANCE_CACHE_PROVENANCE", "dependencies")
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(_env("CLEARANCE_CACHE_TTL_SECONDS", "86400"))
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clearance.db"

    @property
    def vectors_path(self) -> Path:
        return self.data_dir / "vectors.npz"

    def with_overrides(self, **kwargs) -> Settings:
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


def get_settings() -> Settings:
    return Settings()
