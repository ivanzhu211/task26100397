"""受众关键词画像库：为作者分群生成 TF-IDF 关键词摘要。

持久化使用 SQLite。build 消费 (author_id, tokens) 记录迭代器建立索引；
summarize 消费作者标识迭代器，返回分群得分最高的 20 个正分词元及得分。
"""

from __future__ import annotations

import math
import os
import sqlite3
from typing import Dict, Iterable, List, Sequence, Tuple

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
    """消费记录迭代器，在 path 指定的位置建立持久化索引。

    同一作者可跨多条记录出现；空词元记录也用于声明作者存在。
    """
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.executescript(_SCHEMA)
            for author_id, tokens in records:
                key = str(author_id)
                conn.execute(
                    "INSERT OR IGNORE INTO authors (author_id) VALUES (?)", (key,)
                )
                for token in tokens:
                    conn.execute(
                        "INSERT INTO counts (author_id, token, c) VALUES (?, ?, 1) "
                        "ON CONFLICT (author_id, token) DO UPDATE SET c = c + 1",
                        (key, str(token)),
                    )
    finally:
        conn.close()


def summarize(path: str, members: Iterable[str]) -> List[ScoredToken]:
    """返回分群得分最高的 20 个正分词元及得分，不足 20 个则全部返回。

    成员重复只计一次；不存在的作者忽略；空分群返回空列表。
    """
    conn = sqlite3.connect(path)
    try:
        total_authors = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        df: Dict[str, int] = {
            token: n
            for token, n in conn.execute(
                "SELECT token, COUNT(DISTINCT author_id) FROM counts GROUP BY token"
            )
        }
        idf = {
            token: 1.0 + math.log((total_authors + 1) / (n + 1))
            for token, n in df.items()
        }

        seen = set()
        unique_members = []
        for member in members:
            key = str(member)
            if key not in seen:
                seen.add(key)
                unique_members.append(key)

        sums: Dict[str, float] = {}
        group_size = 0
        for key in unique_members:
            rows = conn.execute(
                "SELECT token, c FROM counts WHERE author_id = ?", (key,)
            ).fetchall()
            if not rows and not conn.execute(
                "SELECT 1 FROM authors WHERE author_id = ?", (key,)
            ).fetchone():
                continue  # 不存在的作者忽略
            group_size += 1
            weights = {
                token: (1.0 + math.log(c)) * idf[token] for token, c in rows
            }
            norm = math.sqrt(sum(w * w for w in weights.values()))
            if norm == 0.0:
                continue  # 空词作者：零向量，但仍计入分母
            for token, w in weights.items():
                sums[token] = sums.get(token, 0.0) + w / norm

        if group_size == 0:
            return []
        scores = [
            (token, round(total / group_size, 12))
            for token, total in sums.items()
        ]
        scores = [item for item in scores if item[1] > 0]
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:20]
    finally:
        conn.close()
