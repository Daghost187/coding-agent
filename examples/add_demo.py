#!/usr/bin/env python3
"""End-to-end demo: 1-bit BLLM coding agents generate, heal and fuse code."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from bitllm_hive import BinaryMLP, HiveMind, FusionEngine, CodeHealer


def main() -> None:
    # 1. Show a 1-bit forward pass.
    bllm = BinaryMLP(input_dim=8, hidden_dims=[16, 16], output_dim=4)
    sample = np.random.randn(1, 8).astype(np.float32)
    print("1-bit MLP output shape:", bllm(sample).shape)

    # 2. Hive mind: three experts generate ``add`` implementations.
    system_prompts = [
        "Write a Python function named `add`.",
        "Write a Python function `add` with type hints.",
        "Write a minimal Python function `add`.",
    ]
    hive = HiveMind(
        num_experts=3,
        prompts=system_prompts,
        use_ray=False,
    )
    prompt = "Write a python function `add(a, b)` that returns the sum of a and b."
    candidates = hive.dispatch_local(prompt)
    print("\nGenerated candidates:")
    for c in candidates:
        print(f"--- expert {c['expert_id']} ---")
        print(c["code"])

    # 3. Heal and fuse.
    tests = """
assert add(2, 3) == 5
assert add(0, 0) == 0
assert add(-1, 1) == 0
"""
    fusion = FusionEngine(tests=tests, healer=CodeHealer())
    result = fusion.fuse(candidates, return_details=True)
    print("\n=== Fused result ===")
    print(result["code"])
    print("Score:", result["score"])
    print("Winner expert:", result["winner_id"])


if __name__ == "__main__":
    main()
