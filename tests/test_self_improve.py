import os
import tempfile
import unittest

from bitllm_hive.self_improve import ExperienceBank, MetaLearner, Outcome


class TestSelfImprove(unittest.TestCase):
    def test_experience_bank_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "exp.jsonl")
            bank = ExperienceBank(path=path)
            bank.record(Outcome(task="add two numbers", specialists=["coder"], winner="mock", score=1.0))
            self.assertEqual(len(bank.entries()), 1)
            bank2 = ExperienceBank(path=path)
            self.assertEqual(len(bank2.entries()), 1)

    def test_meta_learner_suggest_team(self):
        bank = ExperienceBank(path="/dev/null")
        bank.record(Outcome(task="add two numbers", specialists=["coder", "tester"], winner="mock", score=1.0))
        bank.record(Outcome(task="multiply two numbers", specialists=["coder"], winner="mock", score=0.0))
        learner = MetaLearner(bank)
        team = learner.suggest_team("sum of two numbers")
        self.assertIn("coder", team)


if __name__ == "__main__":
    unittest.main()
