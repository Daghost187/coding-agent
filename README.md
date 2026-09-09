# coding-agent

A modular, dynamic, self-improving coding agent built around a 1-bit Binary
Large Language Model (BLLM) and a Ray/Hive-Mind swarm of real LLM-powered
coding specialists.

## Architecture

```
Prompt / Task
      │
      ▼
┌──────────────────────────────────────────┐
│ Orchestrator                           │
│  SpecialistRegistry                    │
│   prompt_engineer / architect / coder  │
│   tester / reviewer / optimizer        │
└────────────────────┬───────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   LLMBackend   CodeHealer  FusionEngine
   (real model) (repair)    (run tests)
        │            │            │
        └────────────┴────────────┘
                     │
                     ▼
            ExperienceBank
            MetaLearner
```

## Modules

- `bitllm_hive.binary_llm` — 1-bit weight layers (`BinaryLinear`, `BinaryMLP`).
- `bitllm_hive.backends` — pluggable LLM backends (`openai`, `ollama`, `transformers`, `mock`).
- `bitllm_hive.specialists` — registry of specialist roles (coder, architect, tester, ...).
- `bitllm_hive.hive_mind` — parallel expert swarm (`Ray` with `concurrent.futures` fallback).
- `bitllm_hive.code_healer` — first-aid repair for generated code.
- `bitllm_hive.fusion` — test-based scoring and winner selection.
- `bitllm_hive.self_improve` — `ExperienceBank` + `MetaLearner`.
- `bitllm_hive.orchestrator` — end-to-end task runner.
- `bitllm_hive.config` — load orchestrators from JSON/YAML.

## Quick start

```bash
pip install -r requirements.txt
python examples/add_demo.py
python examples/orchestrator_demo.py
python -m unittest discover -s tests
```

## Orchestrator quick start

```python
from bitllm_hive.orchestrator import Orchestrator

orch = Orchestrator()
result = orch.solve(
    task="Write a python function add(a, b) that returns the sum.",
    tests="assert add(2, 3) == 5",
)
print(result.code)
```

## Real model backends

Configure per-specialist backends:

```python
from bitllm_hive.specialists import Specialist, SpecialistRegistry
from bitllm_hive.orchestrator import Orchestrator

registry = SpecialistRegistry.default()
registry.register(
    "coder",
    Specialist(
        role="coder",
        system_prompt="Write Python.",
        backend_name="openai",
        backend_config={"model": "gpt-4o-mini"},
    ),
)
orch = Orchestrator(registry=registry)
```

Or load from a JSON config (see `bitllm_hive.config`).

## Optional Ray backend

Install Ray and set `use_ray=True` in `HiveMind` to distribute experts
across a Ray cluster:

```bash
pip install ray
```

## Notes

- The `mock` backend is for offline demos and tests.
- Swap in `openai`, `ollama`, or `transformers` for real model inference.
- See `DESIGN.md` for extension patterns.

