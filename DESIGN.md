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
  specialist/backend teams for future tasks.

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
│  openai    - OpenAI-compatible  │
│  ollama    - local Ollama       │
│  transformers - HF pipeline     │
│  mock      - offline tests      │
└──────┬──────────────────────────┘
       │
       ▼
┌──────────────┐
│ CodeHealer   │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ FusionEngine        │
│ (run tests + score) │
└──────┬──────────────┘
       │
       ▼
┌──────────────────────┐
│ ExperienceBank       │
│ MetaLearner          │
│ (self-improve loop)  │
└──────────────────────┘
```

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
registry.register("swe", Specialist(role="swe", system_prompt="You are a site-reliability engineer.", backend_name="openai"))
```

### Configure via JSON

```json
{
  "specialists": {
    "coder": {
      "role": "coder",
      "backend": "openai",
      "backend_config": {"model": "gpt-4o"},
      "temperature": 0.2
    }
  }
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
