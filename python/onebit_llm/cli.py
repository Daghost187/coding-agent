"""Command-line inspection tools for the model skeleton."""

import argparse
from collections.abc import Sequence

from .manifest import ModelManifest
from .weights import DirectoryWeightProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onebit-llm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe", help="print architecture and memory totals")
    subparsers.add_parser("list-tensors", help="print the parameter manifest")
    validate_parser = subparsers.add_parser(
        "validate-weights",
        help="validate a directory containing packed tensor files",
    )
    validate_parser.add_argument("directory")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    manifest = ModelManifest.build()

    if parsed.command == "describe":
        _print_description(manifest)
        return 0
    if parsed.command == "list-tensors":
        for parameter in manifest.parameters:
            print(
                f"{parameter.name}\t{parameter.shape.rows}\t"
                f"{parameter.shape.columns}\t"
                f"{parameter.shape.packed_byte_count}"
            )
        return 0
    if parsed.command == "validate-weights":
        provider = DirectoryWeightProvider(parsed.directory)
        for parameter in manifest.parameters:
            provider.load(parameter)
        print(f"validated_tensor_count={len(manifest.parameters)}")
        return 0
    raise AssertionError(f"unhandled command: {parsed.command}")


def _print_description(manifest: ModelManifest) -> None:
    config = manifest.config
    fields = (
        ("name", config.name),
        ("weight_encoding", "binary-sign-lsb-first"),
        ("vocabulary_size", config.vocabulary_size),
        ("context_length", config.context_length),
        ("model_width", config.model_width),
        ("layer_count", config.layer_count),
        ("query_head_count", config.query_head_count),
        ("key_value_head_count", config.key_value_head_count),
        ("head_width", config.head_width),
        ("feed_forward_width", config.feed_forward_width),
        ("tensor_count", len(manifest.parameters)),
        ("parameter_count", manifest.parameter_count),
        ("packed_weight_bytes", manifest.packed_weight_bytes),
    )
    for key, value in fields:
        print(f"{key}={value}")
