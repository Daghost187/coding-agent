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
- `bitllm_hive.backends` — pluggable LLM backends (`ollama`, `openai_compat`/`local`, `transformers`, `mock`; hosted `openai` is opt-in only).
- `bitllm_hive.metrics` — live `MetricsTracker` (score, latency, heal rate, EWMA) + `AdaptivePolicy` that retunes specialists mid-run.
- `bitllm_hive.cli` — `python -m bitllm_hive` shell entry point.
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

## CLI

```bash
# Local-first: uses Ollama if it's running on localhost, else the offline mock.
python -m bitllm_hive solve "Write a python function add(a, b) that returns the sum" \
  --tests "assert add(2, 3) == 5" --rounds 4

# Point at any self-hosted OpenAI-compatible server (vLLM, llama.cpp, LM Studio, ...)
python -m bitllm_hive solve @task.md --tests @tests.py \
  --backend local --base-url http://192.168.1.20:8000/v1 --model qwen2.5-coder

python -m bitllm_hive stats          # leaderboard: runs, wins, success, latency, heals, temp
python -m bitllm_hive backends       # registered backends + which one is default right now
python -m bitllm_hive specialists    # registered roles and their current tuning
python -m bitllm_hive --plugin my_pkg.my_plugin solve ... # load your own backends/specialists
python -m bitllm_hive reset          # wipe metrics + experience
```

While `solve` runs, `[result]` lines stream to stderr as each specialist
finishes and `[adjust]` lines show the policy retuning temperatures or
benching a specialist based on its metrics. Final code goes to stdout
(`--json` for the full scoreboard, `--out file.py` to save it).

Backend resolution order: `BITLLM_BACKEND` env var -> reachable local
Ollama -> `mock`. Hosted vendors are never picked automatically.

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
        backend_name="ollama",
        backend_config={"model": "qwen2.5-coder:7b"},
    ),
)
# or switch every specialist at once to a self-hosted OpenAI-compatible server:
registry.set_backend("local", {"base_url": "http://localhost:8000/v1", "model": "deepseek-coder"})
orch = Orchestrator(registry=registry)
```

Or load from a JSON config (see `bitllm_hive.config`).

### Plugins

A plugin is any importable module exposing `register(registry, backends)`:

```python
# my_plugin.py
from bitllm_hive.backends import LLMBackend
from bitllm_hive.specialists import Specialist

class MyBackend(LLMBackend):
    name = "mine"
    def generate(self, prompt, system=None, **kw):
        ...

def register(registry, backends):
    backends.register("mine", lambda cfg: MyBackend(**cfg))
    registry.register("security", Specialist(role="security", system_prompt="...", backend_name="mine"))
```

Load it with `--plugin my_plugin`, via `"plugins": ["my_plugin"]` in a config
file, or `registry.load_plugin("my_plugin")` in code.

## Optional Ray backend

Install Ray and set `use_ray=True` in `HiveMind` to distribute experts
across a Ray cluster:

```bash
pip install ray
```

## Notes

- The `mock` backend is for offline demos and tests.
- Swap in `ollama`, `local` (OpenAI-compatible), or `transformers` for real model inference; `openai` is available but never auto-selected.
- See `DESIGN.md` for extension patterns.

