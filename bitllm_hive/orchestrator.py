"""End-to-end orchestrator: specialists -> heal -> fuse -> measure -> adapt -> learn."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from .code_healer import CodeHealer
from .fusion import FusionEngine
from .metrics import AdaptivePolicy, MetricsTracker, PolicyConfig, Sample
from .self_improve import ExperienceBank, MetaLearner, Outcome
from .specialists import SpecialistRegistry


@dataclass
class OrchestratorResult:
    """Result of running the orchestrator on a task."""

    task: str
    code: str
    score: float
    winner_id: str | None
    scoreboard: list[dict[str, Any]]
    specialists_used: list[str]
    adjustments: list[str] = field(default_factory=list)


ProgressHook = Callable[[dict[str, Any]], None]


class Orchestrator:
    """Run a dynamic team of specialists against a coding task.

    Results are scored as they arrive, fed into ``MetricsTracker``, and the
    ``AdaptivePolicy`` tunes the specialists in place for the next run.
    """

    def __init__(
        self,
        registry: SpecialistRegistry | None = None,
        healer: CodeHealer | None = None,
        bank: ExperienceBank | None = None,
        tracker: MetricsTracker | None = None,
        policy: AdaptivePolicy | None = None,
        policy_config: PolicyConfig | None = None,
        parallel: bool = True,
        on_progress: ProgressHook | None = None,
    ):
        self.registry = registry or SpecialistRegistry.default()
        self.healer = healer or CodeHealer()
        self.bank = bank or ExperienceBank()
        self.meta = MetaLearner(self.bank)
        self.tracker = tracker or MetricsTracker()
        self.policy = policy or AdaptivePolicy(self.registry, self.tracker, policy_config)
        self.parallel = parallel
        self.on_progress = on_progress

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_progress:
            self.on_progress(event)

    def _run_specialist(self, name: str, task: str, context: str = "") -> dict[str, Any]:
        specialist = self.registry.get(name)
        start = time.perf_counter()
        error: str | None = None
        try:
            raw = specialist.generate(task, context=context)
        except Exception as exc:  # backend down, timeout, bad response...
            raw, error = "", f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - start
        healed, fixes = self.healer.heal(raw) if raw else ("", [])
        return {
            "expert_id": name,
            "backend": specialist.resolved_backend_name(),
            "code": healed,
            "latency_s": latency,
            "healed": bool(fixes),
            "error": error,
        }

    def _choose_team(self, task: str, specialists: list[str] | None) -> list[str]:
        team = list(specialists) if specialists else self.meta.suggest_team(task, top_k=self.policy.config.max_team)
        team = [n for n in team if n in self.registry.names()]
        if not team:
            team = ["coder"] if "coder" in self.registry.names() else self.registry.names()[:1]
        if not specialists:
            team = self.policy.pick_team(team)
        return team

    def solve(
        self,
        task: str,
        tests: str,
        specialists: list[str] | None = None,
        context: str = "",
    ) -> OrchestratorResult:
        team = self._choose_team(task, specialists)
        fusion = FusionEngine(tests=tests, healer=self.healer)
        candidates: list[dict[str, Any]] = []
        scored: list[tuple[int, dict[str, Any]]] = []
        before = len(self.policy.adjustments)

        self._emit({"event": "team", "team": team})

        def handle(cand: dict[str, Any]) -> None:
            score = fusion.score(cand["code"]) if cand["code"] else 0
            cand["score"] = score
            candidates.append(cand)
            scored.append((score, cand))
            self.tracker.record(
                Sample(
                    specialist=cand["expert_id"],
                    backend=cand["backend"],
                    task=task,
                    score=float(score),
                    latency_s=cand["latency_s"],
                    healed=cand["healed"],
                    code_len=len(cand["code"]),
                )
            )
            self._emit({"event": "result", **{k: cand[k] for k in ("expert_id", "backend", "score", "latency_s", "healed", "error")}})

        if self.parallel and len(team) > 1:
            with ThreadPoolExecutor(max_workers=len(team)) as pool:
                futures = [pool.submit(self._run_specialist, n, task, context) for n in team]
                for fut in as_completed(futures):
                    handle(fut.result())
        else:
            for n in team:
                handle(self._run_specialist(n, task, context))

        best_score, best = max(scored, key=lambda x: (x[0], -x[1]["latency_s"]))
        self.tracker.mark_win(best["expert_id"])

        outcome = Outcome(task=task, specialists=team, winner=best["expert_id"], score=float(best_score), code=best["code"])
        self.bank.record(outcome)

        adjustments = self.policy.adjustments[before:]
        for adj in adjustments:
            self._emit({"event": "adjust", "message": adj})

        return OrchestratorResult(
            task=task,
            code=best["code"],
            score=best_score,
            winner_id=best["expert_id"],
            scoreboard=[
                {"expert_id": c["expert_id"], "backend": c["backend"], "score": s, "latency_s": round(c["latency_s"], 3), "error": c["error"]}
                for s, c in scored
            ],
            specialists_used=team,
            adjustments=adjustments,
        )
