"""Pluggable coding expert backend."""

from __future__ import annotations

import abc
import os
import re
from typing import Any


class LLMBackend(abc.ABC):
    """Abstract base for any language model backend."""

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a code string for the given prompt."""
        ...


class MockLLM(LLMBackend):
    """Deterministic mock backend for offline demos and CI."""

    TEMPLATES = {
        "add": """def add(a, b):
    return a + b
""",
        "mul": """def multiply(a, b):
    return a * b
""",
        "fib": """def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
""",
    }

    def __init__(self, fallbacks: dict[str, str] | None = None):
        self.fallbacks = fallbacks or {}

    def _detect_topic(self, prompt: str) -> str | None:
        lowered = prompt.lower()
        for topic in ["fibonacci", "fib"]:
            if topic in lowered:
                return "fib"
        for topic in ["multiply", "multiplication", "product", "mul"]:
            if topic in lowered:
                return "mul"
        for topic in ["add", "sum", "addition", "plus"]:
            if topic in lowered:
                return "add"
        return None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        topic = self._detect_topic(prompt)
        code = self.TEMPLATES.get(topic, "") or self.fallbacks.get(topic, "")
        if not code:
            # Try to extract a function name from the prompt.
            match = re.search(r"function\s+(\w+)", prompt, re.IGNORECASE)
            if match:
                name = match.group(1)
            else:
                name = "solution"
            code = f"def {name}(*args, **kwargs):\n    pass\n"
        return code


class APILLM(LLMBackend):
    """OpenAI-compatible HTTP backend.  Set ``OPENAI_API_KEY`` or pass a token."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import urllib.request
        import json

        if not self.api_key:
            raise RuntimeError("APILLM requires an API key; set OPENAI_API_KEY")

        messages = [{"role": "system", "content": "You are a terse Python expert."}]
        messages.append({"role": "user", "content": prompt})
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(data).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"]


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
