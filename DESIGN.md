# Design: bitllm_hive

A modular, dynamic, self-improving coding agent swarm built around a 1-bit
Binary Large Language Model (BLLM) and real LLM backends.

## Goals

- **Modular**: every component (backend, specialist, healer, fusion, learning)
  is pluggable and replaceable.
- **Dynamic**: add new specialist roles and backends at runtime through a
  registry/config.
- **Bare-metal fast**: multiprocessing/threading by default; optional Ray for
  clusters.
- **Self-improving**: outcomes are recorded and used to select better
  specialist/backend teams for future tasks, and specialists are retuned
  *while* a run is in progress based on live engineering metrics.
- **Local-first, no vendor lock-in**: the default backend is whatever open
  model is reachable on the box (Ollama, or any OpenAI-compatible server such
  as vLLM / llama.cpp / LM Studio). Hosted APIs exist as opt-in plugins only.

## Components

```
Prompt / Task
      │
      ▼
┌─────────────┐
│ Orchestrator │
└──────┬──────┘
       │
       ▼
┌───────────────────────────────┐
│ SpecialistRegistry            │
│  prompt_engineer  (refine task) │
│  architect      (design)      │
│  coder          (write code)  │
│  tester         (write tests) │
│  reviewer       (review)        │
│  optimizer      (optimize)    │
└──────┬────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ LLMBackend (real model call)    │
│  ollama    - local Ollama       │
│  local     - any OpenAI-compat  │
│  transformers - HF pipeline     │
│  mock      - offline tests      │
│  openai    - hosted, opt-in     │
└──────┬──────────────────────────┘
       │
       ▼
┌──────────────┐
│ CodeHealer   │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐      ┌──────────────────────────┐
│ FusionEngine        │ ────►│ MetricsTracker (live)      │
│ (run tests + score) │ each │  score/latency/heal/EWMA   │
└──────┬──────────────┘ result└──────┬───────────────────┘
       │                     │ AdaptivePolicy: retune temp,
       ▼                     │ bench/unbench, rank team
┌──────────────────────┐              ▼
│ ExperienceBank       │◄───── SpecialistRegistry (mutated in place)
│ MetaLearner          │
│ (cross-run learning) │
└──────────────────────┘
```

## Live adaptation loop

Two feedback loops run at different timescales:

1. **Within a run** (`metrics.py`). Every specialist result is scored by the
   `FusionEngine` the moment it lands and recorded as a `Sample`
   (score, latency, healed?, code size, backend). `MetricsTracker` keeps
   per-specialist EWMA score, success rate, avg latency and heal rate, and
   `AdaptivePolicy` subscribes to it:
   - EWMA below `cold_ewma` -> raise temperature (explore).
   - EWMA above `hot_ewma` -> lower temperature (exploit).
   - EWMA below `bench_ewma` -> bench the specialist; it is re-admitted once
     it recovers above `cold_ewma`.
   - `pick_team` ranks active specialists by EWMA and caps the team size.
   All changes are written to the live `SpecialistRegistry`, so the very next
   dispatch already uses the new tuning. The orchestrator streams
   `team` / `result` / `adjust` events through `on_progress`.
2. **Across runs** (`self_improve.py`). Winners and scores go into the
   `ExperienceBank`; `MetaLearner` uses similarity-weighted history to propose
   the initial team for the next task.

A failing backend (server down, timeout) never sinks the squad: the specialist
scores 0 with an `error` field, the rest of the team is fused as usual, and
the policy benches the offender if it keeps failing.

## Backend resolution

`resolve_default_backend()` picks, in order: `BITLLM_BACKEND` env var, a
reachable local Ollama, then `mock`. Nothing hosted is ever chosen
automatically; use `--backend openai` or `backend_name="openai"` explicitly.

## Extending

### Add a new backend

```python
from bitllm_hive.backends import BackendRegistry, LLMBackend

class MyBackend(LLMBackend):
    name = "my"
    def generate(self, prompt, **kwargs):
        return "..."

BackendRegistry.register("my", lambda c: MyBackend(**c))
```

### Add a new specialist

```python
from bitllm_hive.specialists import SpecialistRegistry, Specialist

registry = SpecialistRegistry.default()
registry.register("swe", Specialist(role="swe", system_prompt="You are a site-reliability engineer.", backend_name="ollama"))
```

### Package it as a plugin

Any importable module with `register(registry, backends)` (or
`module:callable`) is a plugin. Drop your new backends and specialists in
there and load it with `--plugin`, `"plugins": [...]` in a config file, or
`registry.load_plugin(...)`. See `tests/sample_plugin.py`.

### Configure via JSON

```json
{
  "specialists": {
    "coder": {
      "role": "coder",
      "backend": "local",
      "backend_config": {"base_url": "http://localhost:8000/v1", "model": "qwen2.5-coder"},
      "temperature": 0.2
    }
  },
  "plugins": ["my_pkg.my_plugin"],
  "metrics": {"path": "bitllm_metrics.jsonl"},
  "policy": {"min_runs": 3, "hot_ewma": 0.85, "cold_ewma": 0.35, "bench_ewma": 0.15, "max_team": 4}
}
```

```python
from bitllm_hive.config import orchestrator_from_file
orch = orchestrator_from_file("config.json")
```

## 1-bit layer

`binary_llm.py` provides `BinaryLinear` and `BinaryMLP`.  They are not a full
language model, but demonstrate how a 1-bit weight network can be used as a
local scorer/router inside the swarm.
