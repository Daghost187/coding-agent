# Architecture and Extension Contract

## Parameter layout

The model is a pre-normalized, decoder-only transformer. Normalization and
rotary positions are parameter-free, and linear layers have no biases.
Consequently, every trainable parameter is represented by one binary matrix.

For model width `d`, key/value width `k`, feed-forward width `f`, vocabulary
size `v`, and layer count `L`, the parameter count is:

```text
2vd + L(2d² + 2kd + 3df)
```

With the default configuration:

```text
d = 2,048
k = 1,024
f = 5,544
v = 32,768
L = 40

parameters = 2,000,027,648
packed bytes = 250,003,456
```

The manifest is deterministic and ordered:

1. token embedding
2. seven matrices per decoder layer:
   - query, key, value, and attention output
   - feed-forward gate, up, and down
3. output projection

## Binary representation

The format stores one trainable weight per bit:

```text
0 = -1
1 = +1
```

Tensor data is row-major. Bit index zero is the least-significant bit of byte
zero. The final byte may contain unused high bits when a tensor's element
count is not divisible by eight.

This representation defines weight storage only. Activations, accumulators,
normalization, rotary positions, logits, and optimizer state are deliberately
left to training and inference backends.

## Stable core

The dependency-free core owns:

- configuration validation
- deterministic tensor names and shapes
- exact parameter and packed-memory accounting
- packed binary tensor validation and bit access
- a local directory weight provider
- token-range and generation-option validation

## Replaceable modules

### Tokenizer

Implement `Tokenizer` to attach a vocabulary and code-aware tokenization.
Token identifiers must remain within the configured vocabulary.

### Weight provider

Implement `WeightProvider` for memory mapping, archives, object storage,
sharding, compression, or device-local storage. Providers must return the
exact shape requested by the manifest.

### Inference backend

Implement `InferenceBackend` to execute embedding lookup, parameter-free RMS
normalization, rotary attention, gated feed-forward layers, causal KV caching,
logit sampling, and binary matrix kernels. The core does not select a numeric
precision or hardware target for non-weight values.

### Training backend

A training package should consume `ModelConfig` and `ModelManifest` as the
source of truth, maintain latent trainable values if required, project them
to `-1/+1` for forward passes, and export packed files matching the manifest.
Optimizer state is not part of the inference weight format.

## Integration rules

- Treat manifest names as stable external identifiers.
- Reject missing, extra-sized, or malformed tensors before inference.
- Version any replacement aggregate weight container explicitly.
- Keep tokenizer metadata and generation policy separate from binary weights.
- Evaluate generated code with compilers, tests, static analysis, and security
  checks before deployment.
- Do not infer production readiness from parameter count or bit width alone.
