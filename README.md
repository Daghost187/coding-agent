# OneBit Code 2B Skeleton

A dependency-free reference skeleton for a decoder-only coding language model
with strictly binary trainable weights (`0 -> -1`, `1 -> +1`). The same model
contract is implemented in Python and C++20.

The default architecture contains **2,000,027,648 parameters** and requires
**250,003,456 bytes** for packed weights. No weights, tokenizer, training
framework, inference engine, accelerator integration, networking, hooks, or
telemetry are included.

## Architecture

- 32,768-token vocabulary
- 4,096-token context
- 2,048-wide hidden state
- 40 decoder layers
- 16 query heads and 8 key/value heads
- 5,544-wide gated feed-forward blocks
- untied token embedding and output matrices
- rotary positions and parameter-free normalization
- no biases; every trainable parameter belongs to a binary matrix

See [docs/architecture.md](docs/architecture.md) for the component contract
and extension boundaries.

## Python

Requires Python 3.10 or newer and uses only the standard library.

```sh
PYTHONPATH=python python3 -m onebit_llm describe
PYTHONPATH=python python3 -m onebit_llm list-tensors
PYTHONPATH=python python3 -m onebit_llm validate-weights /path/to/weights
```

The public API exposes immutable configuration and manifest objects, packed
binary tensors, a directory weight provider, and abstract tokenizer and
inference-backend interfaces.

## C++20

Requires a C++20 compiler and `make`. It uses only the C++ standard library.

```sh
make
./build/onebit-llm describe
./build/onebit-llm list-tensors
./build/onebit-llm validate-weights /path/to/weights
```

Consumers can include headers from `cpp/include` and link the source files in
`cpp/src` into their own build system.

## Weight files

Each tensor is stored independently as:

```text
<weight-directory>/<manifest-parameter-name>.bin
```

Bits are row-major and least-significant-bit first within each byte. A clear
bit represents `-1`; a set bit represents `+1`. File sizes must exactly match
the manifest. The independent-file layout is intentionally simple and can be
replaced through the `WeightProvider` interface.

## Verification

```sh
make test
```

This builds with warnings treated as errors, runs dependency-free unit tests,
and verifies that the Python and C++20 CLIs produce identical architecture
and memory totals.

## Scope

This repository is production-oriented **infrastructure**, not a trained or
production-capable model by itself. Producing useful code requires a
tokenizer, a binary-aware training stack, trained weights, an execution
backend, model evaluation, safety controls, and deployment-specific resource
limits. Those concerns attach through explicit interfaces instead of being
silently bundled into the core.
