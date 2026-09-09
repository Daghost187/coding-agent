"""Fusion engine: score candidate code and merge the best result."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .code_healer import CodeHealer


class FusionEngine:
    """Score generated code against a set of tests and return the winner."""

    def __init__(self, tests: str, healer: CodeHealer | None = None):
        self.tests = tests
        self.healer = healer or CodeHealer()

    def score(self, code: str, timeout: float = 5.0) -> int:
        """Return the number of test lines that pass (0 if any hard failure)."""
        healed, _ = self.healer.heal(code)
        payload = f"{healed}\n{self.tests}\n"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", payload],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 0
        except Exception:
            return 0
        return 1 if proc.returncode == 0 else 0

    def score_all(self, candidates: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
        """Score a list of candidates and return ``(score, candidate)`` pairs."""
        scored = []
        for candidate in candidates:
            code = candidate.get("code", "")
            scored.append((self.score(code), candidate))
        return scored

    def fuse(
        self,
        candidates: list[dict[str, Any]],
        return_details: bool = False,
    ) -> str | dict[str, Any]:
        """Pick the highest-scoring candidate and return its code.

        If ``return_details`` is True, return a dict with the chosen code,
        its score, and the full scoreboard.
        """
        scored = self.score_all(candidates)
        best_score, best_candidate = max(scored, key=lambda x: x[0])
        best_code, _ = self.healer.heal(best_candidate.get("code", ""))

        if not return_details:
            return best_code

        return {
            "code": best_code,
            "score": best_score,
            "winner_id": best_candidate.get("expert_id"),
            "scoreboard": [
                {
                    "expert_id": cand.get("expert_id"),
                    "score": score,
                }
                for score, cand in scored
            ],
        }

    @staticmethod
    def run_in_temp(code: str, tests: str, timeout: float = 5.0) -> dict[str, Any]:
        """Run ``code`` and ``tests`` in a temporary file for debugging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "candidate.py"
            script.write_text(f"{code}\n{tests}\n", encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return {"returncode": -1, "stdout": "", "stderr": "timeout"}
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
