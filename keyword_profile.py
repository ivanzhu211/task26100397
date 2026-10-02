"""Audience keyword profile library backed by SQLite.

build(records, path) consumes (author_id, tokens) records and persists a
per-author term-frequency index at ``path``. summarize(path, members) reads
that index and returns the top-20 positive TF-IDF keywords for a cohort.
"""

from __future__ import annotations

import math
import os
import sqlite3
from typing import Iterable, Iterator, Sequence, Tuple

Record = Tuple[str, Sequence[str]]
ScoredToken = Tuple[str, float]

_SCHEMA = """
CREATE TABLE authors (
    author_id TEXT PRIMARY KEY
);
CREATE TABLE counts (
    author_id TEXT NOT NULL,
    token TEXT NOT NULL,
    c INTEGER NOT NULL,
    PRIMARY KEY (author_id, token)
);
"""


def build(records: Iterable[Record], path: str) -> None:
    """Consume ``records`` and build a persistent index at ``path``."""
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        for author_id, tokens in records:
            conn.execute(
                "INSERT OR IGNORE INTO authors (author_id) VALUES (?)",
                (author_id,),
            )
            per_record: dict[str, int] = {}
            for token in tokens:
                per_record[token] = per_record.get(token, 0) + 1
            conn.executemany(
                "INSERT INTO counts (author_id, token, c) VALUES (?, ?, ?) "
                "ON CONFLICT (author_id, token) DO UPDATE SET c = c + excluded.c",
                [(author_id, token, c) for token, c in per_record.items()],
            )
        conn.commit()
    finally:
        conn.close()


def summarize(path: str, members: Iterable[str]) -> list[ScoredToken]:
    """Return up to 20 top-scoring (token, score) pairs for ``members``."""
    conn = sqlite3.connect(path)
    try:
        n_authors = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        known = {row[0] for row in conn.execute("SELECT author_id FROM authors")}
        cohort = list(dict.fromkeys(m for m in members if m in known))
        if not cohort:
            return []
        idf = {
            token: 1.0 + math.log((n_authors + 1) / (df + 1))
            for token, df in conn.execute(
                "SELECT token, COUNT(*) FROM counts GROUP BY token"
            )
        }
        sums: dict[str, float] = {}
        for member in cohort:
            weights = {
                token: (1.0 + math.log(c)) * idf[token]
                for token, c in conn.execute(
                    "SELECT token, c FROM counts WHERE author_id = ?", (member,)
                )
            }
            norm = math.sqrt(sum(w * w for w in weights.values()))
            if norm == 0.0:
                continue
            for token, weight in weights.items():
                sums[token] = sums.get(token, 0.0) + weight / norm
        size = len(cohort)
        scored = [
            (token, total / size) for token, total in sums.items() if total > 0.0
        ]
        scored.sort(key=lambda item: (-round(item[1], 12), item[0]))
        return scored[:20]
    finally:
        conn.close()
