# coding-agent

A modular, dynamic coding agent built around a 1-bit Binary Large Language
Model (BLLM) and a Ray/Hive-Mind swarm of coding experts.

## Architecture

```
┌─────────────────────────────────────────────┐
│  HiveMind (Ray / ProcessPool)               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Expert 0│ │ Expert 1│ │ Expert 2│ ...   │
│  │  BLLM   │ │  BLLM   │ │  BLLM   │       │
│  └────┬────┘ └────┬────┘ └────┬────┘       │
│       │           │           │              │
│       └───────────┼───────────┘              │
│                   ▼                          │
│          CodeHealer (repair)                 │
│                   │                          │
│                   ▼                          │
│          FusionEngine (score + merge)        │
└─────────────────────────────────────────────┘
```

## Modules

- `bitllm_hive.binary_llm` — 1-bit weight layers and a tiny BLLP scorer.
- `bitllm_hive.coding_expert` — pluggable coding expert with `MockLLM` and `APILLM` backends.
- `bitllm_hive.hive_mind` — parallel swarm of experts (`Ray` with `concurrent.futures` fallback).
- `bitllm_hive.code_healer` — first-aid repair for generated code.
- `bitllm_hive.fusion` — test-based scoring and winner selection.

## Quick start

```bash
pip install -r requirements.txt
python examples/add_demo.py
```

The demo shows a 1-bit forward pass, three experts generating an `add`
function, and the fusion engine picking the highest-scoring candidate.

## Optional Ray backend

Install Ray and set `use_ray=True` in `HiveMind` to distribute experts
across a Ray cluster:

```bash
pip install ray
```

## Notes

- The `MockLLM` backend is for offline demos and tests.
- Replace `MockLLM` with `APILLM` (or any `LLMBackend` subclass) to use a
  real model.
