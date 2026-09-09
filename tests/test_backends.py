import unittest

from bitllm_hive.backends import BackendRegistry, MockBackend, OpenAIBackend, OllamaBackend, LLMBackend


class TestBackends(unittest.TestCase):
    def test_registry_has_defaults(self):
        names = BackendRegistry.available()
        self.assertIn("mock", names)
        self.assertIn("openai", names)
        self.assertIn("ollama", names)
        self.assertIn("transformers", names)

    def test_mock_backend_returns_code(self):
        backend = MockBackend()
        code = backend.generate("Write a python function add(a, b).")
        self.assertIn("def add", code)

    def test_openai_backend_requires_key(self):
        backend = OpenAIBackend(api_key=None)
        with self.assertRaises(RuntimeError):
            backend.generate("hello")

    def test_ollama_backend_fields(self):
        backend = OllamaBackend(model="codellama")
        self.assertEqual(backend.model, "codellama")
        self.assertTrue(issubclass(OllamaBackend, LLMBackend))


if __name__ == "__main__":
    unittest.main()
