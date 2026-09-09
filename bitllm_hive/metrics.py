"""Live engineering metrics and an adaptive policy that tunes specialists on the fly."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .specialists import Specialist, SpecialistRegistry


@dataclass
class Sample:
    """One specialist run."""

    specialist: str
    backend: str
    task: str
    score: float
    latency_s: float
    healed: bool
    code_len: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class SpecialistStats:
    runs: int = 0
    wins: int = 0
    score_sum: float = 0.0
    latency_sum: float = 0.0
    heals: int = 0
    ewma_score: float = 0.5

    @property
    def success_rate(self) -> float:
        return self.score_sum / self.runs if self.runs else 0.0

    @property
    def avg_latency(self) -> float:
        return self.latency_sum / self.runs if self.runs else 0.0

    @property
    def heal_rate(self) -> float:
        return self.heals / self.runs if self.runs else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            success_rate=round(self.success_rate, 3),
            avg_latency=round(self.avg_latency, 3),
            heal_rate=round(self.heal_rate, 3),
            ewma_score=round(self.ewma_score, 3),
        )
        return d


Listener = Callable[[Sample], None]


class MetricsTracker:
    """Thread-safe rolling metrics per specialist, persisted as JSONL."""

    def __init__(self, path: str | Path | None = "bitllm_metrics.jsonl", alpha: float = 0.3):
        self.path = Path(path) if path else None
        self.alpha = alpha
        self._stats: dict[str, SpecialistStats] = {}
        self._samples: list[Sample] = []
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if "win" in data:
                self._stats.setdefault(data["win"], SpecialistStats()).wins += 1
                continue
            try:
                self._ingest(Sample(**data), persist=False)
            except TypeError:
                continue

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def record(self, sample: Sample) -> None:
        self._ingest(sample, persist=True)
        for listener in list(self._listeners):
            listener(sample)

    def mark_win(self, specialist: str) -> None:
        with self._lock:
            self._stats.setdefault(specialist, SpecialistStats()).wins += 1
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"win": specialist, "timestamp": time.time()}) + "\n")

    def _ingest(self, sample: Sample, persist: bool) -> None:
        with self._lock:
            st = self._stats.setdefault(sample.specialist, SpecialistStats())
            st.runs += 1
            st.score_sum += sample.score
            st.latency_sum += sample.latency_s
            st.heals += int(sample.healed)
            st.ewma_score = (1 - self.alpha) * st.ewma_score + self.alpha * sample.score
            self._samples.append(sample)
            if persist and self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(sample)) + "\n")

    def stats(self, specialist: str) -> SpecialistStats:
        with self._lock:
            return self._stats.get(specialist, SpecialistStats())

    def all_stats(self) -> dict[str, SpecialistStats]:
        with self._lock:
            return dict(self._stats)

    def samples(self) -> list[Sample]:
        with self._lock:
            return list(self._samples)

    def leaderboard(self) -> list[tuple[str, SpecialistStats]]:
        return sorted(
            self.all_stats().items(),
            key=lambda kv: (kv[1].ewma_score, kv[1].success_rate, -kv[1].avg_latency),
            reverse=True,
        )

    def report(self) -> str:
        rows = ["specialist          runs  wins  success  ewma   heal   latency(s)"]
        for name, st in self.leaderboard():
            rows.append(
                f"{name:<18} {st.runs:>5} {st.wins:>5}  {st.success_rate:>6.2f}  {st.ewma_score:>5.2f}  "
                f"{st.heal_rate:>5.2f}  {st.avg_latency:>9.3f}"
            )
        return "\n".join(rows)


@dataclass
class PolicyConfig:
    """Knobs for ``AdaptivePolicy``."""

    min_runs: int = 3
    cold_ewma: float = 0.35
    hot_ewma: float = 0.85
    temp_step: float = 0.1
    temp_min: float = 0.0
    temp_max: float = 1.0
    bench_ewma: float = 0.15
    max_team: int = 4


class AdaptivePolicy:
    """Adjust specialists in place as metrics stream in.

    Rules (all applied after every recorded sample):
      * low ewma  -> raise temperature (explore), and bench if very low
      * high ewma -> lower temperature (exploit)
      * team selection prefers highest ewma; benched specialists are skipped
    """

    def __init__(
        self,
        registry: SpecialistRegistry,
        tracker: MetricsTracker,
        config: PolicyConfig | None = None,
    ):
        self.registry = registry
        self.tracker = tracker
        self.config = config or PolicyConfig()
        self.benched: set[str] = set()
        self.adjustments: list[str] = []
        tracker.subscribe(self.on_sample)

    def on_sample(self, sample: Sample) -> None:
        name = sample.specialist
        if name not in self.registry.names():
            return
        st = self.tracker.stats(name)
        if st.runs < self.config.min_runs:
            return
        spec = self.registry.get(name)
        cfg = self.config

        if st.ewma_score < cfg.bench_ewma:
            if name not in self.benched:
                self.benched.add(name)
                self._log(f"bench {name} (ewma={st.ewma_score:.2f})")
            return

        if name in self.benched and st.ewma_score >= cfg.cold_ewma:
            self.benched.discard(name)
            self._log(f"unbench {name} (ewma={st.ewma_score:.2f})")

        if st.ewma_score < cfg.cold_ewma:
            self._nudge(spec, +cfg.temp_step, "explore")
        elif st.ewma_score > cfg.hot_ewma:
            self._nudge(spec, -cfg.temp_step, "exploit")

    def _nudge(self, spec: Specialist, delta: float, why: str) -> None:
        old = spec.temperature
        new = min(self.config.temp_max, max(self.config.temp_min, round(old + delta, 3)))
        if new != old:
            spec.temperature = new
            self._log(f"{spec.role}: temperature {old:.2f} -> {new:.2f} ({why})")

    def _log(self, msg: str) -> None:
        self.adjustments.append(msg)

    def pick_team(self, candidates: list[str]) -> list[str]:
        """Rank ``candidates`` by ewma, drop benched, cap at ``max_team``."""
        active = [c for c in candidates if c not in self.benched] or list(candidates)
        ranked = sorted(active, key=lambda n: self.tracker.stats(n).ewma_score, reverse=True)
        return ranked[: self.config.max_team]
