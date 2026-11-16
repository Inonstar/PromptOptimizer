"""On-disk response cache keyed by ``(model, effort, prompt hash)``.

Backed by SQLite (stdlib, no extra dependency) rather than a bespoke
file-per-key scheme, for the same reason ``observability/local.py`` uses
SQLite: atomic writes and safe concurrent access from multiple worker
threads without a lock file of our own.

The cache stores :class:`~promptopt.llm.base.CompletionResult` payloads,
the shared shape every :meth:`~promptopt.llm.base.LLMClient.complete` call
returns. Structured/search results are not cached here in M0 -- callers
that need to skip a repeated structured call can serialize
:class:`~promptopt.llm.base.StructuredResult` themselves; revisit if that
turns out to be a hot path.

:seealso: ``lib_structure.md`` -> "Package layout" (``llm/cache.py``)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from promptopt.llm.base import CompletionResult
from promptopt.llm.routing import Effort


def make_cache_key(
    *,
    model: str,
    effort: Effort | None,
    system: str | None,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    """Derive a stable cache key from the parameters that affect the response.

    :param model: The model identifier.
    :type model: str
    :param effort: The requested effort level, if any.
    :type effort: Effort | None
    :param system: System instructions, if any.
    :type system: str | None
    :param messages: The conversation, as plain ``{"role": ..., "content": ...}`` dicts.
    :type messages: list[dict[str, str]]
    :param max_tokens: The requested output token ceiling.
    :type max_tokens: int
    :returns: A hex-encoded SHA-256 digest identifying this request.
    :rtype: str
    """
    payload = json.dumps(
        {"model": model, "effort": effort, "system": system, "messages": messages, "max_tokens": max_tokens},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """A SQLite-backed cache of :class:`CompletionResult` payloads.

    :param path: File the SQLite database is stored at. Parent directories
        are created if missing.
    :type path: str | Path
    """

    def __init__(self, path: str | Path) -> None:
        """Open (creating if needed) the cache database at ``path``.

        :param path: File the SQLite database is stored at.
        :type path: str | Path
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "  key TEXT PRIMARY KEY,"
                "  value TEXT NOT NULL,"
                "  created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))"
                ")"
            )
            self._conn.commit()

    def get(self, key: str) -> CompletionResult | None:
        """Look up a cached completion by key.

        :param key: The cache key, typically from :func:`make_cache_key`.
        :type key: str
        :returns: The cached result, or ``None`` on a cache miss.
        :rtype: CompletionResult | None
        """
        with self._lock:
            row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return CompletionResult.model_validate_json(row[0])

    def set(self, key: str, result: CompletionResult) -> None:
        """Store a completion under a key, overwriting any existing entry.

        :param key: The cache key, typically from :func:`make_cache_key`.
        :type key: str
        :param result: The completion to store.
        :type result: CompletionResult
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO responses (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, result.model_dump_json()),
            )
            self._conn.commit()

    def clear(self) -> None:
        """Remove every entry from the cache."""
        with self._lock:
            self._conn.execute("DELETE FROM responses")
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ResponseCache":
        """Support use as a context manager.

        :returns: This cache instance.
        :rtype: ResponseCache
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the cache on context-manager exit.

        :param exc_info: Standard ``__exit__`` exception triple, unused.
        :type exc_info: object
        """
        self.close()
