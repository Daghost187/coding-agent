import unittest

from bitllm_hive.code_healer import CodeHealer
from bitllm_hive.fusion import FusionEngine


class TestFusion(unittest.TestCase):
    def test_fusion_picks_best(self):
        tests = "assert add(2, 3) == 5\n"
        candidates = [
            {"expert_id": 0, "code": "def add(a, b):\n    return a + b\n"},
            {"expert_id": 1, "code": "def add(a, b):\n    return a - b\n"},
        ]
        engine = FusionEngine(tests=tests, healer=CodeHealer())
        result = engine.fuse(candidates, return_details=True)
        self.assertEqual(result["winner_id"], 0)
        self.assertEqual(result["score"], 1)


if __name__ == "__main__":
    unittest.main()
