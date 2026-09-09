"""Command-line interface: ``python -m bitllm_hive ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .backends import BackendRegistry, OllamaBackend, resolve_default_backend
from .code_healer import CodeHealer
from .config import build_orchestrator, load_config
from .metrics import MetricsTracker, PolicyConfig
from .orchestrator import Orchestrator
from .self_improve import ExperienceBank
from .specialists import SpecialistRegistry


def _read_text_arg(value: str | None) -> str:
    """Accept inline text, ``@file``, or ``-`` for stdin."""
    if value is None:
        return ""
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _build(args: argparse.Namespace) -> Orchestrator:
    if getattr(args, "config", None):
        orch = build_orchestrator(load_config(args.config))
    else:
        orch = Orchestrator(
            registry=SpecialistRegistry.default(),
            healer=CodeHealer(),
            bank=ExperienceBank(args.experience),
            tracker=MetricsTracker(args.metrics),
            policy_config=PolicyConfig(max_team=args.max_team),
            parallel=not args.serial,
        )
    for plugin in args.plugin or []:
        orch.registry.load_plugin(plugin)
    if args.backend:
        cfg: dict[str, Any] = {}
        if args.model:
            cfg["model"] = args.model
        if args.base_url:
            cfg["base_url"] = args.base_url
        orch.registry.set_backend(args.backend, cfg or None)
    elif args.model or args.base_url:
        for spec in orch.registry.all().values():
            if args.model:
                spec.backend_config["model"] = args.model
            if args.base_url:
                spec.backend_config["base_url"] = args.base_url
    return orch


def _print_event(event: dict[str, Any]) -> None:
    kind = event.get("event")
    if kind == "team":
        print(f"[team] {', '.join(event['team'])}", file=sys.stderr)
    elif kind == "result":
        flag = "healed" if event["healed"] else "clean"
        if event.get("error"):
            flag = f"ERROR {event['error']}"
        print(
            f"[result] {event['expert_id']:<16} backend={event['backend']:<12} "
            f"score={event['score']} {event['latency_s']:.2f}s {flag}",
            file=sys.stderr,
        )
    elif kind == "adjust":
        print(f"[adjust] {event['message']}", file=sys.stderr)


def cmd_solve(args: argparse.Namespace) -> int:
    orch = _build(args)
    orch.on_progress = None if args.quiet else _print_event
    task = _read_text_arg(args.task)
    tests = _read_text_arg(args.tests)
    context = _read_text_arg(args.context)
    team = args.specialists.split(",") if args.specialists else None

    last = None
    for i in range(args.rounds):
        if args.rounds > 1 and not args.quiet:
            print(f"--- round {i + 1}/{args.rounds} ---", file=sys.stderr)
        last = orch.solve(task, tests, specialists=team, context=context)

    assert last is not None
    if args.json:
        print(json.dumps({
            "code": last.code,
            "score": last.score,
            "winner": last.winner_id,
            "team": last.specialists_used,
            "scoreboard": last.scoreboard,
            "adjustments": last.adjustments,
        }, indent=2))
    else:
        print(last.code, end="" if last.code.endswith("\n") else "\n")
        if not args.quiet:
            print(f"\n[winner] {last.winner_id} score={last.score}", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(last.code, encoding="utf-8")
    if not args.quiet:
        print("\n" + orch.tracker.report(), file=sys.stderr)
    return 0 if last.score > 0 else 1


def cmd_stats(args: argparse.Namespace) -> int:
    tracker = MetricsTracker(args.metrics)
    if args.json:
        print(json.dumps({k: v.as_dict() for k, v in tracker.all_stats().items()}, indent=2))
    else:
        print(tracker.report())
    return 0


def cmd_backends(args: argparse.Namespace) -> int:
    default = resolve_default_backend(force=True)
    ollama = OllamaBackend.is_available()
    for name in BackendRegistry.available():
        marker = "*" if name == default else " "
        note = ""
        if name == "ollama":
            note = "(reachable)" if ollama else "(not running)"
        elif name == "openai":
            note = "(hosted, opt-in only)"
        print(f"{marker} {name:<14} {note}")
    print(f"\ndefault: {default}  (override with BITLLM_BACKEND or --backend)")
    return 0


def cmd_specialists(args: argparse.Namespace) -> int:
    registry = SpecialistRegistry.default()
    for plugin in args.plugin or []:
        registry.load_plugin(plugin)
    if args.json:
        print(json.dumps({n: s.as_dict() for n, s in registry.all().items()}, indent=2))
    else:
        for name, spec in registry.all().items():
            print(f"{name:<16} backend={spec.backend_name:<8} temp={spec.temperature:<4} {spec.system_prompt[:70]}...")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    for p in (args.metrics, args.experience):
        path = Path(p)
        if path.exists():
            path.unlink()
            print(f"removed {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bitllm_hive", description="1-bit BLLM specialist hive")
    p.add_argument("--metrics", default="bitllm_metrics.jsonl", help="metrics JSONL path")
    p.add_argument("--experience", default="bitllm_experience.jsonl", help="experience bank path")
    p.add_argument("--plugin", action="append", help="module[:callable] that registers specialists/backends (repeatable)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("solve", help="run the squad on a task")
    s.add_argument("task", help="task text, @file, or - for stdin")
    s.add_argument("--tests", "-t", required=True, help="assertion code, @file, or -")
    s.add_argument("--context", "-c", help="extra context, @file, or -")
    s.add_argument("--specialists", "-s", help="comma-separated team (default: learned)")
    s.add_argument("--backend", "-b", help="backend for all specialists (auto|ollama|local|openai_compat|transformers|mock|openai)")
    s.add_argument("--model", "-m", help="model name passed to the backend")
    s.add_argument("--base-url", help="base URL for ollama/local backends")
    s.add_argument("--config", help="JSON/YAML orchestrator config")
    s.add_argument("--rounds", "-r", type=int, default=1, help="repeat to let the policy adapt")
    s.add_argument("--max-team", type=int, default=4)
    s.add_argument("--serial", action="store_true", help="run specialists one at a time")
    s.add_argument("--out", "-o", help="write winning code to file")
    s.add_argument("--json", action="store_true")
    s.add_argument("--quiet", "-q", action="store_true")
    s.set_defaults(func=cmd_solve)

    st = sub.add_parser("stats", help="show live specialist metrics")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_stats)

    b = sub.add_parser("backends", help="list backends and the resolved default")
    b.set_defaults(func=cmd_backends)

    sp = sub.add_parser("specialists", help="list specialists")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_specialists)

    r = sub.add_parser("reset", help="wipe metrics and experience")
    r.set_defaults(func=cmd_reset)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
