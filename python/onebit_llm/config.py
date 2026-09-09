"""Model configuration and validation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture for the nominal two-billion-parameter coding model."""

    name: str = "onebit-code-2b"
    vocabulary_size: int = 32_768
    context_length: int = 4_096
    model_width: int = 2_048
    layer_count: int = 40
    query_head_count: int = 16
    key_value_head_count: int = 8
    feed_forward_width: int = 5_544
    rotary_base: float = 10_000.0

    def __post_init__(self) -> None:
        integer_fields = {
            "vocabulary_size": self.vocabulary_size,
            "context_length": self.context_length,
            "model_width": self.model_width,
            "layer_count": self.layer_count,
            "query_head_count": self.query_head_count,
            "key_value_head_count": self.key_value_head_count,
            "feed_forward_width": self.feed_forward_width,
        }
        for field_name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not self.name:
            raise ValueError("name must not be empty")
        if self.model_width % self.query_head_count != 0:
            raise ValueError("model_width must be divisible by query_head_count")
        if self.query_head_count % self.key_value_head_count != 0:
            raise ValueError(
                "query_head_count must be divisible by key_value_head_count"
            )
        if self.rotary_base <= 0:
            raise ValueError("rotary_base must be positive")

    @property
    def head_width(self) -> int:
        return self.model_width // self.query_head_count

    @property
    def key_value_width(self) -> int:
        return self.key_value_head_count * self.head_width
