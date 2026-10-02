"""Tests for keyword_profile: computation, invariance, and global IDF."""

import math
import os
import tempfile
import unittest

from keyword_profile import build, summarize


def reference_profile(records, members):
    """Independent in-memory implementation of the specified scoring."""
    counts = {}
    for author_id, tokens in records:
        counts.setdefault(author_id, {})
        for token in tokens:
            counts[author_id][token] = counts[author_id].get(token, 0) + 1
    n_authors = len(counts)
    cohort = list(dict.fromkeys(m for m in members if m in counts))
    if not cohort:
        return []
    df = {}
    for token_counts in counts.values():
        for token in token_counts:
            df[token] = df.get(token, 0) + 1
    idf = {t: 1.0 + math.log((n_authors + 1) / (d + 1)) for t, d in df.items()}
    sums = {}
    for member in cohort:
        weights = {
            t: (1.0 + math.log(c)) * idf[t] for t, c in counts[member].items()
        }
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0.0:
            continue
        for token, weight in weights.items():
            sums[token] = sums.get(token, 0.0) + weight / norm
    size = len(cohort)
    scored = [(t, s / size) for t, s in sums.items() if s > 0.0]
    scored.sort(key=lambda item: (-round(item[1], 12), item[0]))
    return scored[:20]


class KeywordProfileTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "index.sqlite")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_interleaved_computation(self):
        records = [
            ("alice", ["apple", "banana", "apple"]),
            ("prolific", ["apple"] * 10 + ["cherry"] * 3),
            ("bob", ["banana", "date"]),
            ("alice", ["cherry"]),
            ("prolific", ["apple"] * 5 + ["date"]),
            ("lurker", []),
            ("bob", ["apple", "apple"]),
            ("prolific", ["elderberry"] * 20),
        ]
        members = ["alice", "bob", "alice", "prolific", "lurker", "ghost", "bob"]
        build(iter(records), self.path)
        expected = reference_profile(records, members)
        actual = summarize(self.path, iter(members))
        self.assertEqual([t for t, _ in expected], [t for t, _ in actual])
        for (_, exp_score), (_, act_score) in zip(expected, actual):
            self.assertAlmostEqual(exp_score, act_score, places=12)
        # Reopening the persisted index yields the same profile.
        self.assertEqual(actual, summarize(self.path, members))
        # Empty cohort and cohort of unknown authors both return empty.
        self.assertEqual([], summarize(self.path, []))
        self.assertEqual([], summarize(self.path, ["ghost"]))

    def test_split_and_shuffle_invariance(self):
        base = [
            ("alice", ["apple", "banana", "apple"]),
            ("prolific", ["apple"] * 10 + ["cherry"] * 3),
            ("bob", ["banana", "date"]),
            ("alice", ["cherry"]),
            ("prolific", ["apple"] * 5 + ["date"]),
            ("lurker", []),
            ("bob", ["apple", "apple"]),
            ("prolific", ["elderberry"] * 20),
        ]
        # Same per-author totals, but records reshuffled and split differently.
        variant = [
            ("prolific", ["elderberry"] * 7),
            ("bob", ["apple"]),
            ("lurker", []),
            ("prolific", ["apple"] * 8),
            ("alice", ["apple", "banana"]),
            ("prolific", ["cherry"] * 3 + ["elderberry"] * 13),
            ("bob", ["banana", "date", "apple"]),
            ("prolific", ["apple"] * 7 + ["date"]),
            ("alice", ["apple", "cherry"]),
        ]
        members = ["alice", "bob", "prolific", "lurker"]
        build(iter(base), self.path)
        variant_path = os.path.join(self.tmpdir.name, "variant.sqlite")
        build(iter(variant), variant_path)
        self.assertEqual(
            summarize(self.path, members), summarize(variant_path, members)
        )
        # Sanity check: the variant really is a different record layout.
        self.assertNotEqual(len(base), len(variant))

    def test_global_idf_includes_non_members(self):
        records = [
            ("alice", ["apple", "banana", "apple"]),
            ("bob", ["banana", "date"]),
            ("alice", ["cherry"]),
            ("bob", ["apple", "apple"]),
        ]
        members = ["alice", "bob"]
        build(iter(records), self.path)
        before = summarize(self.path, members)
        self.assertEqual(
            [t for t, _ in before],
            [t for t, _ in reference_profile(records, members)],
        )
        # Add an author outside the cohort; global N and df change.
        extended = records + [("carol", ["apple", "apple", "apple"])]
        build(iter(extended), self.path)
        after = summarize(self.path, members)
        expected_after = reference_profile(extended, members)
        self.assertEqual(
            [t for t, _ in expected_after], [t for t, _ in after]
        )
        for (_, exp_score), (_, act_score) in zip(expected_after, after):
            self.assertAlmostEqual(exp_score, act_score, places=12)
        # The non-member reshapes global IDF, so cohort scores must move.
        self.assertEqual([t for t, _ in before], [t for t, _ in after])
        for (_, before_score), (_, after_score) in zip(before, after):
            self.assertNotAlmostEqual(before_score, after_score, places=12)


if __name__ == "__main__":
    unittest.main()
