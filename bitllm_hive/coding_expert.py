"""Backward-compatible coding expert wrapper around ``backends`` and ``specialists``."""

from __future__ import annotations


from .backends import (
    BackendRegistry,
    LLMBackend,
    MockBackend as MockLLM,
    OpenAIBackend as APILLM,
)
from .specialists import Specialist

__all__ = [
    "LLMBackend",
    "MockLLM",
    "APILLM",
    "CodingExpert",
    "Specialist",
    "BackendRegistry",
]


class CodingExpert:
    """A coding expert that generates code using an ``LLMBackend``."""

    def __init__(
        self,
        expert_id: int = 0,
        llm: LLMBackend | None = None,
        system_prompt: str = "",
    ):
        self.expert_id = expert_id
        self.llm = llm or MockLLM()
        self.system_prompt = system_prompt or "Write clean, runnable Python code."

    def generate(self, prompt: str, context: str | None = None) -> str:
        full = f"{self.system_prompt}\nTask: {prompt}"
        if context:
            full += f"\nContext: {context}"
        return self.llm.generate(full, expert_id=self.expert_id)
