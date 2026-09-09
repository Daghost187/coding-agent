"""End-to-end orchestrator: specialists -> heal -> fuse -> learn."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .backends import LLMBackend
from .code_healer import CodeHealer
from .fusion import FusionEngine
from .hive_mind import HiveMind
from .self_improve import ExperienceBank, MetaLearner, Outcome
from .specialists import Specialist, SpecialistRegistry


@dataclass
class OrchestratorResult:
    """Result of running the orchestrator on a task."""

    task: str
    code: str
    score: float
    winner_id: int | None
    scoreboard: list[dict[str, Any]]
    specialists_used: list[str]


class Orchestrator:
    """Run a dynamic team of specialists against a coding task."""

    def __init__(
        self,
        registry: SpecialistRegistry | None = None,
        healer: CodeHealer | None = None,
        bank: ExperienceBank | None = None,
        parallel: bool = True,
    ):
        self.registry = registry or SpecialistRegistry.default()
        self.healer = healer or CodeHealer()
        self.bank = bank or ExperienceBank()
        self.meta = MetaLearner(self.bank)
        self.parallel = parallel

    def _run_specialist(self, name: str, task: str, context: str = "") -> dict[str, Any]:
        specialist = self.registry.get(name)
        code = specialist.generate(task, context=context)
        healed, _ = self.healer.heal(code)
        return {"expert_id": name, "code": healed}

    def solve(
        self,
        task: str,
        tests: str,
        specialists: list[str] | None = None,
    ) -> OrchestratorResult:
        """Run the chosen specialists, heal their outputs, fuse the best one, and learn."""
        team = specialists or self.meta.suggest_team(task)
        # Ensure at least a coder is present.
        if "coder" not in team and not any(self.registry.get(n).role == "coder" for n in team if n in self.registry.names()):
            team.append("coder")

        if self.parallel and len(team) > 1:
            with ThreadPoolExecutor(max_workers=len(team)) as pool:
                candidates = list(pool.map(lambda n: self._run_specialist(n, task), team))
        else:
            candidates = [self._run_specialist(n, task) for n in team]

        fusion = FusionEngine(tests=tests, healer=self.healer)
        fused = fusion.fuse(candidates, return_details=True)

        outcome = Outcome(
            task=task,
            specialists=team,
            winner=fused["winner_id"],
            score=float(fused["score"]),
            code=fused["code"],
        )
        self.bank.record(outcome)

        return OrchestratorResult(
            task=task,
            code=fused["code"],
            score=fused["score"],
            winner_id=fused["winner_id"],
            scoreboard=fused["scoreboard"],
            specialists_used=team,
        )
