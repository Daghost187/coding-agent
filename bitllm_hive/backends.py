"""Pluggable LLM backends for real model inference.

All backends implement ``LLMBackend``.  New backends can be registered at
runtime through ``BackendRegistry``.
"""

from __future__ import annotations

import abc
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar


class LLMBackend(abc.ABC):
    """Abstract base for any text-generation backend."""

    name: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """Return the model's text completion for ``prompt``."""
        ...

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper that builds a single user prompt from messages."""
        parts = []
        system = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return self.generate("\n".join(parts), system=system or None, temperature=temperature, max_tokens=max_tokens, **kwargs)


@dataclass
class MockBackend(LLMBackend):
    """Deterministic backend for offline demos and CI."""

    name: ClassVar[str] = "mock"
    templates: dict[str, str] = field(default_factory=dict)
    default_name: str = "solution"

    _BUILTINS: ClassVar[dict[str, str]] = {
        "add": "def add(a, b):\n    return a + b\n",
        "mul": "def multiply(a, b):\n    return a * b\n",
        "fib": "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    }

    def _topic(self, prompt: str) -> str | None:
        lowered = prompt.lower()
        for topic in ("fibonacci", "fib"):
            if topic in lowered:
                return "fib"
        for topic in ("multiply", "multiplication", "product", "mul"):
            if topic in lowered:
                return "mul"
        for topic in ("add", "sum", "addition", "plus"):
            if topic in lowered:
                return "add"
        return None

    def _extract_name(self, prompt: str) -> str:
        match = re.search(r"function\s+(\w+)", prompt, re.IGNORECASE)
        return match.group(1) if match else self.default_name

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        topic = self._topic(prompt)
        code = self.templates.get(topic, "") or self._BUILTINS.get(topic, "")
        if code:
            return code
        return f"def {self._extract_name(prompt)}(*args, **kwargs):\n    pass\n"


@dataclass
class OpenAICompatBackend(LLMBackend):
    """Any OpenAI-compatible ``/v1/chat/completions`` server.

    Defaults target a self-hosted endpoint (vLLM, llama.cpp server, TGI,
    LocalAI, LiteLLM, Ollama's ``/v1``...).  No API key is required unless the
    server enforces one.
    """

    name: ClassVar[str] = "openai_compat"
    base_url: str = field(default_factory=lambda: os.getenv("BITLLM_OPENAI_BASE_URL", "http://localhost:8000/v1"))
    model: str = field(default_factory=lambda: os.getenv("BITLLM_MODEL", "local-model"))
    api_key: str | None = field(default_factory=lambda: os.getenv("BITLLM_API_KEY"))
    require_key: bool = False
    timeout: int = 120

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2, max_tokens: int = 1024, **kwargs: Any) -> str:
        if self.require_key and not self.api_key:
            raise RuntimeError(f"{type(self).__name__} requires an API key")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data.update(kwargs.get("extra_body", {}))

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(data).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"]


@dataclass
class OpenAIBackend(OpenAICompatBackend):
    """Hosted OpenAI API.  Opt-in only; requires ``OPENAI_API_KEY``."""

    name: ClassVar[str] = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    require_key: bool = True


@dataclass
class OllamaBackend(LLMBackend):
    """Local Ollama backend (``ollama run`` server)."""

    name: ClassVar[str] = "ollama"
    model: str = field(default_factory=lambda: os.getenv("BITLLM_MODEL", "qwen2.5-coder:7b"))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    timeout: int = 180

    def __post_init__(self):
        if not self.base_url.startswith("http"):
            self.base_url = f"http://{self.base_url}"
        self.base_url = self.base_url.rstrip("/")

    @classmethod
    def is_available(cls, base_url: str | None = None, timeout: float = 0.5) -> bool:
        url = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not url.startswith("http"):
            url = f"http://{url}"
        try:
            with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout):
                return True
        except Exception:
            return False

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7, max_tokens: int = 1024, **kwargs: Any) -> str:
        data = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("response", "")


@dataclass
class TransformersBackend(LLMBackend):
    """Local Hugging Face ``transformers`` backend.  Imports lazily."""

    name: ClassVar[str] = "transformers"
    model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: str = "auto"

    _pipeline: Any | None = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("TransformersBackend requires `pip install transformers torch`") from exc
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name_or_path,
            device_map=self.device if self.device != "auto" else "auto",
            torch_dtype="auto",
        )
        return self._pipeline

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2, max_tokens: int = 1024, **kwargs: Any) -> str:
        pipe = self._load()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        out = pipe(messages, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=temperature)
        return out[0]["generated_text"][-1].get("content", "")


class BackendRegistry:
    """Central registry for backend factories."""

    _registry: ClassVar[dict[str, Callable[[dict[str, Any]], LLMBackend]]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[dict[str, Any]], LLMBackend]) -> None:
        cls._registry[name] = factory

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> LLMBackend:
        if name not in cls._registry:
            raise KeyError(f"Unknown backend '{name}'. Available: {list(cls._registry)}")
        return cls._registry[name](config or {})

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)

    @classmethod
    def reset(cls) -> None:
        cls._registry.clear()
        _register_defaults()


_DEFAULT_CACHE: dict[str, str] = {}


def resolve_default_backend(force: bool = False) -> str:
    """Pick the default backend name.

    Order: ``BITLLM_BACKEND`` env var -> local Ollama if reachable -> ``mock``.
    Hosted vendors are never chosen automatically.
    """
    if not force and "name" in _DEFAULT_CACHE:
        return _DEFAULT_CACHE["name"]
    env = os.getenv("BITLLM_BACKEND")
    if env:
        name = env
    elif OllamaBackend.is_available():
        name = "ollama"
    else:
        name = "mock"
    _DEFAULT_CACHE["name"] = name
    return name


def _register_defaults() -> None:
    BackendRegistry.register("mock", lambda c: MockBackend(**c))
    BackendRegistry.register("ollama", lambda c: OllamaBackend(**c))
    BackendRegistry.register("openai_compat", lambda c: OpenAICompatBackend(**c))
    BackendRegistry.register("local", lambda c: OpenAICompatBackend(**c))
    BackendRegistry.register("transformers", lambda c: TransformersBackend(**c))
    BackendRegistry.register("openai", lambda c: OpenAIBackend(**c))
    BackendRegistry.register("auto", lambda c: BackendRegistry.create(resolve_default_backend(), c))


_register_defaults()
