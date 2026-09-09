"""Hive mind: a swarm of coding experts producing code in parallel.

Uses Ray when available and requested, otherwise falls back to the
standard library ``concurrent.futures`` pool.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Sequence

from .coding_expert import CodingExpert, LLMBackend, MockLLM


try:
    import ray
    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover
    _RAY_AVAILABLE = False


class HiveMind:
    """Coordinate N ``CodingExpert`` agents and collect their outputs."""

    def __init__(
        self,
        num_experts: int = 3,
        backend_factory: Callable[[], LLMBackend] | None = None,
        prompts: Sequence[str] | None = None,
        use_ray: bool = False,
    ):
        self.num_experts = max(1, num_experts)
        self.backend_factory = backend_factory or (lambda: MockLLM())
        self.prompts = prompts or [""] * self.num_experts
        self.use_ray = use_ray and _RAY_AVAILABLE
        self._ray_actors: list[Any] | None = None

        if self.use_ray:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self._make_actors()

    def _make_actors(self) -> None:
        ExpertActor = ray.remote(CodingExpert)  # type: ignore[var-annotated]
        self._ray_actors = [
            ExpertActor.remote(i, self.backend_factory(), prompt)
            for i, prompt in enumerate(self.prompts)
        ]

    def _expert_call(self, args: tuple[int, str, str | None]) -> dict[str, Any]:
        expert_id, prompt, context = args
        llm = self.backend_factory()
        expert = CodingExpert(expert_id=expert_id, llm=llm, system_prompt=self.prompts[expert_id])
        return {
            "expert_id": expert_id,
            "code": expert.generate(prompt, context),
        }

    def dispatch(self, prompt: str, context: str | None = None) -> list[dict[str, Any]]:
        """Run every expert on ``prompt`` and return their outputs."""
        if self.use_ray and self._ray_actors:
            refs = [actor.generate.remote(prompt, context) for actor in self._ray_actors]
            return ray.get(refs)

        args = [(i, prompt, context) for i in range(self.num_experts)]
        with ProcessPoolExecutor(max_workers=self.num_experts) as pool:
            return list(pool.map(self._expert_call, args))

    def dispatch_local(self, prompt: str, context: str | None = None) -> list[dict[str, Any]]:
        """Synchronous in-process fallback for quick tests."""
        return [self._expert_call((i, prompt, context)) for i in range(self.num_experts)]

    def close(self) -> None:
        if self.use_ray and ray.is_initialized():
            ray.shutdown()
