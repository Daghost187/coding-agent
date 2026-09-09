"""Load orchestrator/hive configurations from JSON/YAML/dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .code_healer import CodeHealer
from .metrics import MetricsTracker, PolicyConfig
from .orchestrator import Orchestrator
from .self_improve import ExperienceBank
from .specialists import SpecialistRegistry, build_specialist_from_config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file.  YAML is supported if PyYAML is installed."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires `pip install pyyaml`") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def build_orchestrator(config: dict[str, Any]) -> Orchestrator:
    """Build an ``Orchestrator`` from a configuration dict.

    Example config::

        {
          "include_defaults": true,
          "plugins": ["my_pkg.specialists:register"],
          "specialists": {
            "coder": {"role": "coder", "backend": "ollama", "backend_config": {"model": "qwen2.5-coder:7b"}, "temperature": 0.2}
          },
          "healer": {"max_attempts": 5},
          "experience_bank": {"path": "experience.jsonl"},
          "metrics": {"path": "metrics.jsonl"},
          "policy": {"min_runs": 3, "hot_ewma": 0.85, "cold_ewma": 0.35, "max_team": 4},
          "parallel": true
        }
    """
    registry = SpecialistRegistry.default() if config.get("include_defaults", True) else SpecialistRegistry()
    for name, cfg in config.get("specialists", {}).items():
        registry.register(name, build_specialist_from_config(name, cfg))
    for plugin in config.get("plugins", []):
        registry.load_plugin(plugin)

    return Orchestrator(
        registry=registry,
        healer=CodeHealer(**config.get("healer", {})),
        bank=ExperienceBank(**config.get("experience_bank", {"path": "bitllm_experience.jsonl"})),
        tracker=MetricsTracker(**config.get("metrics", {"path": "bitllm_metrics.jsonl"})),
        policy_config=PolicyConfig(**config.get("policy", {})),
        parallel=config.get("parallel", True),
    )


def orchestrator_from_file(path: str | Path) -> Orchestrator:
    return build_orchestrator(load_config(path))
