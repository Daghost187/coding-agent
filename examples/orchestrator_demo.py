#!/usr/bin/env python3
"""End-to-end orchestrator demo: specialists + heal + fuse + learn."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitllm_hive.orchestrator import Orchestrator


def main() -> None:
    orch = Orchestrator(parallel=True)
    result = orch.solve(
        task="Write a python function `add(a, b)` that returns the sum of a and b.",
        tests="""
assert add(2, 3) == 5
assert add(0, 0) == 0
assert add(-1, 1) == 0
""",
    )
    print("=== Specialists used ===")
    print(result.specialists_used)
    print("\n=== Fused code ===")
    print(result.code)
    print("\n=== Score ===")
    print(result.score)
    print("\n=== Scoreboard ===")
    for row in result.scoreboard:
        print(row)


if __name__ == "__main__":
    main()
