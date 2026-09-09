"""Stable interfaces for optional tokenizers and execution backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from .config import ModelConfig
from .manifest import ModelManifest
from .weights import WeightProvider


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 1

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> Sequence[int]:
        """Convert source text to token identifiers."""

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token identifiers to text."""


class InferenceBackend(ABC):
    """Execution extension point; the core package selects no backend."""

    @abstractmethod
    def generate(
        self,
        manifest: ModelManifest,
        weights: WeightProvider,
        prompt_token_ids: Sequence[int],
        options: GenerationOptions,
    ) -> Sequence[int]:
        """Return generated token identifiers, excluding the prompt."""


class OneBitModel:
    """Backend-neutral model facade."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._manifest = ModelManifest.build(config)

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def complete(
        self,
        prompt: str,
        tokenizer: Tokenizer,
        weights: WeightProvider,
        backend: InferenceBackend,
        options: GenerationOptions | None = None,
    ) -> str:
        prompt_tokens = tuple(tokenizer.encode(prompt))
        self._validate_tokens(prompt_tokens)
        generated_tokens = tuple(
            backend.generate(
                self._manifest,
                weights,
                prompt_tokens,
                options or GenerationOptions(),
            )
        )
        self._validate_tokens(generated_tokens)
        return tokenizer.decode(generated_tokens)

    def _validate_tokens(self, token_ids: Sequence[int]) -> None:
        vocabulary_size = self._manifest.config.vocabulary_size
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("token identifiers must be integers")
            if token_id < 0 or token_id >= vocabulary_size:
                raise ValueError(
                    f"token identifier {token_id} is outside the vocabulary"
                )
