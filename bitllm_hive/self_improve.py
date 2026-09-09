"""Self-improvement through an experience bank and meta-learner."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Outcome:
    """One recorded result of a coding task."""

    task: str
    specialists: list[str]
    winner: str
    score: float
    code: str = ""
    timestamp: float | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ExperienceBank:
    """Append-only JSONL storage for past task outcomes."""

    def __init__(self, path: str | Path = "bitllm_experience.jsonl"):
        self.path = Path(path)
        self._entries: list[Outcome] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._entries.append(Outcome(**data))
                except Exception:
                    continue

    def record(self, outcome: Outcome) -> None:
        self._entries.append(outcome)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(outcome), default=str) + "\n")

    def entries(self) -> list[Outcome]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        if self.path.exists():
            self.path.unlink()


class MetaLearner:
    """Choose specialist teams and prompts based on past outcomes."""

    def __init__(self, bank: ExperienceBank | None = None):
        self.bank = bank or ExperienceBank()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(text.lower().split())

    def _similarity(self, a: str, b: str) -> float:
        tokens_a = self._tokenize(a)
        tokens_b = self._tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0
        inter = tokens_a & tokens_b
        return len(inter) / (math.sqrt(len(tokens_a)) * math.sqrt(len(tokens_b)))

    def suggest_team(self, task: str, top_k: int = 3) -> list[str]:
        """Return the specialist team with the highest weighted score for ``task``."""
        if not self.bank.entries():
            return ["coder", "tester", "optimizer"][:top_k]

        scores: dict[tuple[str, ...], float] = {}
        weights: dict[tuple[str, ...], float] = {}
        for outcome in self.bank.entries():
            sim = self._similarity(task, outcome.task)
            team = tuple(sorted(outcome.specialists))
            scores[team] = scores.get(team, 0.0) + outcome.score * sim
            weights[team] = weights.get(team, 0.0) + sim

        ranked = []
        for team in scores:
            w = weights[team]
            if w == 0:
                continue
            ranked.append((scores[team] / w, list(team)))

        ranked.sort(key=lambda x: x[0], reverse=True)
        if not ranked:
            return ["coder"]
        return ranked[0][1] if not top_k else ranked[0][1][:top_k]

    def best_backend_for_role(self, role: str, task: str) -> str:
        """Recommend a backend name for a role based on past task similarity."""
        best_score = -1.0
        best_backend = "mock"
        backend_scores: dict[str, float] = {}
        backend_weights: dict[str, float] = {}
        for outcome in self.bank.entries():
            if role not in outcome.specialists:
                continue
            sim = self._similarity(task, outcome.task)
            backend_scores[outcome.winner] = backend_scores.get(outcome.winner, 0.0) + outcome.score * sim
            backend_weights[outcome.winner] = backend_weights.get(outcome.winner, 0.0) + sim
        for backend, score in backend_scores.items():
            w = backend_weights[backend]
            avg = score / w if w else 0.0
            if avg > best_score:
                best_score = avg
                best_backend = backend
        return best_backend
