import os
import tempfile
import unittest

from bitllm_hive.metrics import AdaptivePolicy, MetricsTracker, PolicyConfig, Sample
from bitllm_hive.specialists import SpecialistRegistry


def _sample(name, score, latency=0.1):
    return Sample(specialist=name, backend="mock", task="t", score=score, latency_s=latency, healed=False, code_len=10)


class TestMetricsTracker(unittest.TestCase):
    def test_stats_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.jsonl")
            tr = MetricsTracker(path)
            tr.record(_sample("coder", 1.0))
            tr.record(_sample("coder", 0.0))
            tr.mark_win("coder")
            st = tr.stats("coder")
            self.assertEqual(st.runs, 2)
            self.assertEqual(st.wins, 1)
            self.assertAlmostEqual(st.success_rate, 0.5)

            reloaded = MetricsTracker(path)
            self.assertEqual(reloaded.stats("coder").runs, 2)
            self.assertEqual(reloaded.stats("coder").wins, 1)

    def test_leaderboard_orders_by_ewma(self):
        tr = MetricsTracker(None)
        for _ in range(3):
            tr.record(_sample("good", 1.0))
            tr.record(_sample("bad", 0.0))
        names = [n for n, _ in tr.leaderboard()]
        self.assertEqual(names, ["good", "bad"])
        self.assertIn("good", tr.report())


class TestAdaptivePolicy(unittest.TestCase):
    def setUp(self):
        self.registry = SpecialistRegistry.default()
        self.tracker = MetricsTracker(None)
        self.policy = AdaptivePolicy(self.registry, self.tracker, PolicyConfig(min_runs=2))

    def test_exploit_lowers_temperature_on_success(self):
        start = self.registry.get("coder").temperature
        for _ in range(6):
            self.tracker.record(_sample("coder", 1.0))
        self.assertLess(self.registry.get("coder").temperature, start)
        self.assertTrue(any("exploit" in a for a in self.policy.adjustments))

    def test_explore_then_bench_on_failure(self):
        start = self.registry.get("tester").temperature
        for _ in range(3):
            self.tracker.record(_sample("tester", 0.0))
        self.assertGreater(self.registry.get("tester").temperature, start)
        for _ in range(6):
            self.tracker.record(_sample("tester", 0.0))
        self.assertIn("tester", self.policy.benched)
        team = self.policy.pick_team(["coder", "tester"])
        self.assertEqual(team, ["coder"])

    def test_pick_team_caps_and_ranks(self):
        for _ in range(3):
            self.tracker.record(_sample("optimizer", 1.0))
            self.tracker.record(_sample("coder", 0.5))
        self.policy.config.max_team = 1
        self.assertEqual(self.policy.pick_team(["coder", "optimizer"]), ["optimizer"])


if __name__ == "__main__":
    unittest.main()
