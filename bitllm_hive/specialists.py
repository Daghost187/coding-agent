"""Specialist coding agents and dynamic registry."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .backends import BackendRegistry, LLMBackend, resolve_default_backend


class Role(str, Enum):
    """Well-known specialist roles."""

    PROMPT_ENGINEER = "prompt_engineer"
    ARCHITECT = "architect"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    OPTIMIZER = "optimizer"
    HEALER = "healer"


DEFAULT_PROMPTS: dict[str, str] = {
    Role.PROMPT_ENGINEER: (
        "You are a prompt engineer. Refine and expand the user's task into a precise, "
        "actionable coding specification. Keep it short."
    ),
    Role.ARCHITECT: (
        "You are a software architect. Given a specification, produce a concise design: "
        "module names, public functions/classes, and key invariants. Output bullet points."
    ),
    Role.CODER: (
        "You are an expert Python programmer. Write clean, runnable, well-typed code. "
        "Return only the code inside a single markdown code block."
    ),
    Role.TESTER: (
        "You are a test engineer. Given a code snippet, write pytest-style assertions "
        "that exercise the main functionality. Return only the test code."
    ),
    Role.REVIEWER: (
        "You are a code reviewer. Review the provided code for bugs, edge cases, and "
        "style issues. Return a concise list of findings and a corrected version if needed."
    ),
    Role.OPTIMIZER: (
        "You are a performance engineer. Optimize the provided code for correctness, "
        "speed, and clarity. Return only the improved code."
    ),
    Role.HEALER: (
        "You are a code healer. Fix syntax errors and obvious logic bugs in the provided "
        "code. Return only the corrected code."
    ),
}


@dataclass
class Specialist:
    """A single specialist with a role, system prompt, and backend configuration."""

    role: str
    system_prompt: str
    backend_name: str = "auto"
    backend_config: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.2
    max_tokens: int = 1024
    tags: list[str] = field(default_factory=list)

    def backend(self) -> LLMBackend:
        return BackendRegistry.create(self.backend_name, self.backend_config)

    def resolved_backend_name(self) -> str:
        return resolve_default_backend() if self.backend_name == "auto" else self.backend_name

    def generate(self, prompt: str, context: str | None = None, **kwargs: Any) -> str:
        backend = self.backend()
        full = ""
        if context:
            full += f"Context:\n{context}\n\n"
        full += f"Task:\n{prompt}"
        return backend.generate(
            full,
            system=self.system_prompt,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "system_prompt": self.system_prompt,
            "backend": self.backend_name,
            "backend_config": dict(self.backend_config),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tags": list(self.tags),
        }


class SpecialistRegistry:
    """Factory and registry for named specialists."""

    def __init__(self):
        self._specialists: dict[str, Specialist] = {}

    def register(self, name: str, specialist: Specialist) -> None:
        self._specialists[name] = specialist

    def get(self, name: str) -> Specialist:
        if name not in self._specialists:
            raise KeyError(f"Unknown specialist '{name}'. Available: {list(self._specialists)}")
        return self._specialists[name]

    def unregister(self, name: str) -> None:
        self._specialists.pop(name, None)

    def names(self) -> list[str]:
        return list(self._specialists)

    def all(self) -> dict[str, Specialist]:
        return dict(self._specialists)

    def set_backend(self, backend_name: str, backend_config: dict[str, Any] | None = None, only: list[str] | None = None) -> None:
        """Point some or all specialists at a backend."""
        for name, spec in self._specialists.items():
            if only and name not in only:
                continue
            spec.backend_name = backend_name
            if backend_config is not None:
                spec.backend_config = dict(backend_config)

    def load_plugin(self, spec: str) -> None:
        """Import ``module`` or ``module:callable`` and let it register specialists/backends.

        The callable receives ``(registry, BackendRegistry)``.  If only a module is
        given, a top-level ``register(registry, backends)`` function is expected.
        """
        module_name, _, attr = spec.partition(":")
        module = importlib.import_module(module_name)
        hook: Callable[..., Any] = getattr(module, attr or "register")
        hook(self, BackendRegistry)

    @classmethod
    def default(cls) -> "SpecialistRegistry":
        registry = cls()
        for role, prompt in DEFAULT_PROMPTS.items():
            registry.register(
                role.value,
                Specialist(role=role.value, system_prompt=prompt),
            )
        return registry


def build_specialist_from_config(name: str, config: dict[str, Any]) -> Specialist:
    """Build a ``Specialist`` from a plain dict."""
    role = config.get("role", name)
    return Specialist(
        role=role,
        system_prompt=config.get("system_prompt", DEFAULT_PROMPTS.get(role, "You are a helpful coding assistant.")),
        backend_name=config.get("backend", "auto"),
        backend_config=config.get("backend_config", {}),
        temperature=config.get("temperature", 0.2),
        max_tokens=config.get("max_tokens", 1024),
        tags=list(config.get("tags", [])),
    )
