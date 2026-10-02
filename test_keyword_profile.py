"""keyword_profile 的自动化测试。"""

import math
import os
import random
import tempfile
import unittest
from collections import Counter

from keyword_profile import build, summarize


def reference_summarize(records, members):
    """独立的纯内存参考实现，用于对照。"""
    counts = {}
    for author_id, tokens in records:
        counts.setdefault(str(author_id), Counter()).update(map(str, tokens))
    total_authors = len(counts)
    df = Counter()
    for counter in counts.values():
        for token in counter:
            df[token] += 1
    idf = {t: 1.0 + math.log((total_authors + 1) / (n + 1)) for t, n in df.items()}

    seen, unique = set(), []
    for m in members:
        k = str(m)
        if k not in seen:
            seen.add(k)
            unique.append(k)
    unique = [k for k in unique if k in counts]
    if not unique:
        return []

    sums = {}
    for k in unique:
        weights = {t: (1.0 + math.log(c)) * idf[t] for t, c in counts[k].items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0.0:
            continue
        for t, w in weights.items():
            sums[t] = sums.get(t, 0.0) + w / norm
    scores = [(t, round(s / len(unique), 12)) for t, s in sums.items()]
    scores = [item for item in scores if item[1] > 0]
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:20]


BASE_RECORDS = [
    ("alice", ["apple", "banana", "apple"]),
    ("bob", ["banana", "cherry"]),
    ("alice", ["cherry", "apple"]),
    ("carol", []),  # 空词作者
    ("bob", ["apple", "apple", "apple", "banana"]),  # 高产作者
    ("dave", ["durian"]),
    ("alice", []),
]


class KeywordProfileTest(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        os.remove(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_interleaved_records_and_duplicate_members(self):
        build(iter(BASE_RECORDS), self.path)
        members = ["alice", "bob", "alice", "carol", "ghost", "bob"]
        expected = reference_summarize(BASE_RECORDS, ["alice", "bob", "carol"])
        result = summarize(self.path, iter(members))
        self.assertEqual(result, expected)
        self.assertTrue(all(score > 0 for _, score in result))
        self.assertLessEqual(len(result), 20)
        # 排序：分值降序，同分按词元升序
        self.assertEqual(
            result, sorted(result, key=lambda item: (-item[1], item[0]))
        )

    def test_empty_token_author_counts_in_denominator(self):
        build(iter(BASE_RECORDS), self.path)
        with_carol = summarize(self.path, ["alice", "carol"])
        without_carol = summarize(self.path, ["alice"])
        self.assertTrue(with_carol)
        for (t1, s1), (t2, s2) in zip(with_carol, without_carol):
            self.assertEqual(t1, t2)
            self.assertLessEqual(abs(s1 * 2 - s2), 2e-12)  # 容忍 round(...,12) 的尾差

    def test_reorder_and_split_invariance(self):
        build(iter(BASE_RECORDS), self.path)
        baseline = summarize(self.path, ["alice", "bob", "carol"])

        # 按作者聚合成总词频后，随机拆分、交错、重排
        rng = random.Random(42)
        per_author = {}
        for author_id, tokens in BASE_RECORDS:
            per_author.setdefault(author_id, []).extend(tokens)
        shuffled = []
        for author_id, tokens in per_author.items():
            pool = list(tokens)
            rng.shuffle(pool)
            while pool:
                take = rng.randint(1, len(pool))
                shuffled.append((author_id, pool[:take]))
                pool = pool[take:]
            shuffled.append((author_id, []))
        rng.shuffle(shuffled)

        build(iter(shuffled), self.path)
        self.assertEqual(summarize(self.path, ["alice", "bob", "carol"]), baseline)

    def test_outsider_author_changes_global_idf(self):
        build(iter(BASE_RECORDS), self.path)
        before = summarize(self.path, ["alice", "bob"])

        extra = BASE_RECORDS + [("erin", ["apple", "zebra"])]
        build(iter(extra), self.path)
        after = summarize(self.path, ["alice", "bob"])

        self.assertEqual(after, reference_summarize(extra, ["alice", "bob"]))
        self.assertNotEqual(before, after)
        shared = {t: s for t, s in before} 
        changed = [t for t, s in after if t in shared and shared[t] != s]
        self.assertTrue(changed, "全局 IDF 变化应影响分群得分")

    def test_persistence_and_reopen(self):
        build(iter(BASE_RECORDS), self.path)
        first = summarize(self.path, ["alice", "bob"])
        # 每次 summarize 都重新打开同一文件，模拟索引关闭后重开
        second = summarize(self.path, ["alice", "bob"])
        self.assertEqual(first, second)
        self.assertTrue(first)

    def test_empty_group_and_unknown_authors(self):
        build(iter(BASE_RECORDS), self.path)
        self.assertEqual(summarize(self.path, []), [])
        self.assertEqual(summarize(self.path, ["ghost", "nobody"]), [])

    def test_empty_token_author_only_group(self):
        build(iter(BASE_RECORDS), self.path)
        self.assertEqual(summarize(self.path, ["carol"]), [])


if __name__ == "__main__":
    unittest.main()
