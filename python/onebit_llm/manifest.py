"""Deterministic parameter manifest for the model architecture."""

from dataclasses import dataclass

from .config import ModelConfig


@dataclass(frozen=True, slots=True)
class TensorShape:
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("tensor dimensions must be positive")

    @property
    def element_count(self) -> int:
        return self.rows * self.columns

    @property
    def packed_byte_count(self) -> int:
        return (self.element_count + 7) // 8


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    shape: TensorShape

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must not be empty")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    config: ModelConfig
    parameters: tuple[ParameterSpec, ...]

    @classmethod
    def build(cls, config: ModelConfig | None = None) -> "ModelManifest":
        resolved = config or ModelConfig()
        width = resolved.model_width
        kv_width = resolved.key_value_width
        ff_width = resolved.feed_forward_width

        parameters: list[ParameterSpec] = [
            ParameterSpec(
                "token_embedding.weight",
                TensorShape(resolved.vocabulary_size, width),
            )
        ]
        for layer_index in range(resolved.layer_count):
            prefix = f"layers.{layer_index:02d}"
            parameters.extend(
                (
                    ParameterSpec(
                        f"{prefix}.attention.query.weight",
                        TensorShape(width, width),
                    ),
                    ParameterSpec(
                        f"{prefix}.attention.key.weight",
                        TensorShape(kv_width, width),
                    ),
                    ParameterSpec(
                        f"{prefix}.attention.value.weight",
                        TensorShape(kv_width, width),
                    ),
                    ParameterSpec(
                        f"{prefix}.attention.output.weight",
                        TensorShape(width, width),
                    ),
                    ParameterSpec(
                        f"{prefix}.feed_forward.gate.weight",
                        TensorShape(ff_width, width),
                    ),
                    ParameterSpec(
                        f"{prefix}.feed_forward.up.weight",
                        TensorShape(ff_width, width),
                    ),
                    ParameterSpec(
                        f"{prefix}.feed_forward.down.weight",
                        TensorShape(width, ff_width),
                    ),
                )
            )
        parameters.append(
            ParameterSpec(
                "output.weight",
                TensorShape(resolved.vocabulary_size, width),
            )
        )
        return cls(resolved, tuple(parameters))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.shape.element_count for parameter in self.parameters)

    @property
    def packed_weight_bytes(self) -> int:
        return sum(
            parameter.shape.packed_byte_count for parameter in self.parameters
        )

    def find(self, name: str) -> ParameterSpec:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)
