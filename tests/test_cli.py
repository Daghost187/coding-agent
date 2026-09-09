import contextlib
import io
import json
import os
import tempfile
import unittest

from bitllm_hive.backends import BackendRegistry
from bitllm_hive.cli import main
from bitllm_hive.specialists import SpecialistRegistry


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.metrics = os.path.join(self.tmp.name, "m.jsonl")
        self.exp = os.path.join(self.tmp.name, "e.jsonl")
        self.base = ["--metrics", self.metrics, "--experience", self.exp]

    def tearDown(self):
        self.tmp.cleanup()
        BackendRegistry.reset()

    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([*self.base, *argv])
        return code, out.getvalue(), err.getvalue()

    def test_solve_json_and_rounds(self):
        code, out, err = self._run(
            "solve", "Write a python function add(a, b)", "-t", "assert add(1, 2) == 3",
            "--backend", "mock", "--rounds", "4", "--json",
        )
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertIn("def add", data["code"])
        self.assertEqual(data["score"], 1)
        self.assertTrue(any("exploit" in a for a in data["adjustments"]))
        self.assertIn("[result]", err)
        self.assertIn("[adjust]", err)

    def test_solve_nonzero_exit_on_failure(self):
        code, out, _ = self._run(
            "solve", "Write a python function add(a, b)", "-t", "assert add(1, 2) == 99",
            "--backend", "mock", "-q",
        )
        self.assertEqual(code, 1)

    def test_stats_after_solve(self):
        self._run("solve", "Write add", "-t", "assert add(1, 1) == 2", "-b", "mock", "-q", "-s", "coder")
        code, out, _ = self._run("stats", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["coder"]["runs"], 1)
        self.assertEqual(json.loads(out)["coder"]["wins"], 1)

    def test_backends_never_default_to_hosted(self):
        code, out, _ = self._run("backends")
        self.assertEqual(code, 0)
        self.assertNotIn("default: openai", out)
        self.assertIn("(hosted, opt-in only)", out)

    def test_plugin_loads_specialist_and_backend(self):
        code, out, _ = self._run("--plugin", "tests.sample_plugin", "specialists", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["shouter"]["backend"], "shout")

        code, out, _ = self._run(
            "--plugin", "tests.sample_plugin", "solve", "add", "-t", "assert add(2, 2) == 4",
            "-s", "shouter", "--json",
        )
        self.assertEqual(code, 0)
        self.assertIn("SHOUT", json.loads(out)["code"])

    def test_reset(self):
        self._run("solve", "add", "-t", "assert add(1, 1) == 2", "-b", "mock", "-q", "-s", "coder")
        self.assertTrue(os.path.exists(self.metrics))
        self._run("reset")
        self.assertFalse(os.path.exists(self.metrics))


class TestOrchestratorResilience(unittest.TestCase):
    def test_failing_specialist_does_not_crash_squad(self):
        from bitllm_hive.orchestrator import Orchestrator
        from bitllm_hive.self_improve import ExperienceBank
        from bitllm_hive.metrics import MetricsTracker
        from bitllm_hive.specialists import Specialist

        registry = SpecialistRegistry.default()
        registry.set_backend("mock")
        registry.register("broken", Specialist(role="broken", system_prompt="x", backend_name="openai"))
        orch = Orchestrator(registry=registry, bank=ExperienceBank("/dev/null"), tracker=MetricsTracker(None))
        result = orch.solve("Write add(a, b)", "assert add(1, 1) == 2", specialists=["broken", "coder"])
        self.assertEqual(result.winner_id, "coder")
        errors = {r["expert_id"]: r["error"] for r in result.scoreboard}
        self.assertIsNotNone(errors["broken"])
        self.assertIsNone(errors["coder"])


if __name__ == "__main__":
    unittest.main()
