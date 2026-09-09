"""Load orchestrator/hive configurations from JSON/YAML/dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .code_healer import CodeHealer
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
          "specialists": {
            "coder": {"role": "coder", "backend": "openai", "backend_config": {"model": "gpt-4o"}, "temperature": 0.2}
          },
          "healer": {"max_attempts": 5},
          "experience_bank": {"path": "experience.jsonl"}
        }
    """
    registry = SpecialistRegistry()
    for name, cfg in config.get("specialists", {}).items():
        registry.register(name, build_specialist_from_config(name, cfg))

    healer_cfg = config.get("healer", {})
    bank_cfg = config.get("experience_bank", {"path": "bitllm_experience.jsonl"})
    return Orchestrator(
        registry=registry,
        healer=CodeHealer(**healer_cfg),
        bank=ExperienceBank(**bank_cfg),
    )


def orchestrator_from_file(path: str | Path) -> Orchestrator:
    return build_orchestrator(load_config(path))
