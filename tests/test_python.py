import tempfile
import unittest
from pathlib import Path

from onebit_llm import (
    BinaryTensor,
    DirectoryWeightProvider,
    GenerationOptions,
    InferenceBackend,
    ModelConfig,
    ModelManifest,
    OneBitModel,
    ParameterSpec,
    TensorShape,
    Tokenizer,
    WeightProvider,
)


class StaticTokenizer(Tokenizer):
    def encode(self, text: str) -> list[int]:
        return [len(text)]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


class UnusedWeightProvider(WeightProvider):
    def load(self, parameter: ParameterSpec) -> BinaryTensor:
        raise AssertionError(f"unexpected weight load: {parameter.name}")


class StaticBackend(InferenceBackend):
    def generate(
        self,
        manifest: ModelManifest,
        weights: WeightProvider,
        prompt_token_ids: list[int] | tuple[int, ...],
        options: GenerationOptions,
    ) -> list[int]:
        self.last_parameter_count = manifest.parameter_count
        self.last_prompt = tuple(prompt_token_ids)
        self.last_limit = options.max_new_tokens
        return [7, 8]


class ModelConfigTests(unittest.TestCase):
    def test_default_manifest_is_nominally_two_billion_parameters(self) -> None:
        manifest = ModelManifest.build()

        self.assertEqual(manifest.parameter_count, 2_000_027_648)
        self.assertEqual(manifest.packed_weight_bytes, 250_003_456)
        self.assertEqual(len(manifest.parameters), 282)
        self.assertEqual(manifest.config.head_width, 128)
        self.assertEqual(manifest.config.key_value_width, 1_024)

    def test_invalid_head_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(model_width=2_047)


class BinaryTensorTests(unittest.TestCase):
    def test_binary_signs_are_lsb_first(self) -> None:
        tensor = BinaryTensor(TensorShape(2, 4), bytes((0b1010_0101,)))

        self.assertEqual(
            [tensor.sign(row, column) for row in range(2) for column in range(4)],
            [1, -1, 1, -1, -1, 1, -1, 1],
        )

    def test_directory_provider_validates_exact_size(self) -> None:
        parameter = ParameterSpec("test.weight", TensorShape(2, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.weight.bin"
            path.write_bytes(bytes((0xFF,)))
            tensor = DirectoryWeightProvider(directory).load(parameter)

        self.assertEqual(tensor.packed_data, bytes((0xFF,)))

    def test_directory_provider_rejects_wrong_size(self) -> None:
        parameter = ParameterSpec("test.weight", TensorShape(2, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.weight.bin"
            path.write_bytes(b"")
            with self.assertRaises(ValueError):
                DirectoryWeightProvider(directory).load(parameter)


class RuntimeTests(unittest.TestCase):
    def test_backend_and_tokenizer_are_replaceable(self) -> None:
        model = OneBitModel()
        backend = StaticBackend()

        result = model.complete(
            "code",
            StaticTokenizer(),
            UnusedWeightProvider(),
            backend,
            GenerationOptions(max_new_tokens=12),
        )

        self.assertEqual(result, "7,8")
        self.assertEqual(backend.last_prompt, (4,))
        self.assertEqual(backend.last_limit, 12)
        self.assertEqual(backend.last_parameter_count, 2_000_027_648)


if __name__ == "__main__":
    unittest.main()
