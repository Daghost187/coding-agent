import unittest

from bitllm_hive.specialists import Specialist, SpecialistRegistry, Role, build_specialist_from_config


class TestSpecialists(unittest.TestCase):
    def test_default_registry_populated(self):
        registry = SpecialistRegistry.default()
        self.assertIn("coder", registry.names())
        self.assertIn("tester", registry.names())

    def test_specialist_generate_uses_backend(self):
        specialist = Specialist(role="coder", system_prompt="You are a coder.")
        code = specialist.generate("Write a python function add(a, b).")
        self.assertIn("def add", code)

    def test_build_from_config(self):
        cfg = {
            "role": "optimizer",
            "backend": "mock",
            "temperature": 0.1,
        }
        specialist = build_specialist_from_config("optimizer", cfg)
        self.assertEqual(specialist.role, "optimizer")
        self.assertEqual(specialist.temperature, 0.1)

    def test_role_enum_values(self):
        self.assertEqual(Role.CODER.value, "coder")


if __name__ == "__main__":
    unittest.main()
