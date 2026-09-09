"""Example plugin: registers a custom backend and a specialist that uses it."""

from bitllm_hive.backends import LLMBackend
from bitllm_hive.specialists import Specialist


class ShoutBackend(LLMBackend):
    name = "shout"

    def generate(self, prompt, system=None, **kwargs):
        return "def add(a, b):\n    return a + b  # SHOUT\n"


def register(registry, backends):
    backends.register("shout", lambda cfg: ShoutBackend())
    registry.register(
        "shouter",
        Specialist(role="shouter", system_prompt="Shout code.", backend_name="shout", tags=["demo"]),
    )
