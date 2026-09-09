import os
import tempfile
import unittest

from bitllm_hive.orchestrator import Orchestrator
from bitllm_hive.self_improve import ExperienceBank


class TestOrchestrator(unittest.TestCase):
    def test_solve_add_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = ExperienceBank(path=os.path.join(tmp, "exp.jsonl"))
            orchestrator = Orchestrator(bank=bank, parallel=False)
            result = orchestrator.solve(
                task="Write a python function add(a, b) that returns the sum.",
                tests="assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
            )
            self.assertIn("def add", result.code)
            self.assertEqual(result.score, 1)
            self.assertGreater(len(bank.entries()), 0)

    def test_solve_with_specialist_override(self):
        orchestrator = Orchestrator(parallel=False)
        result = orchestrator.solve(
            task="Write a python function add(a, b) that returns the sum.",
            tests="assert add(0, 0) == 0\n",
            specialists=["coder"],
        )
        self.assertIn("def add", result.code)
        self.assertEqual(result.specialists_used, ["coder"])


if __name__ == "__main__":
    unittest.main()
