# lodestone

A 1-bit transformer foundation in C++23. Weights are magnetic poles, `+1` or `-1`. The matmul does not multiply: it counts bits.

There is no dependency beyond a C++23 compiler. Scalar, AVX2, and AVX-512 VPOPCNTDQ kernels are built into one binary and chosen at runtime.

## The dot product

A value of `+1` is stored as bit 1. A value of `-1` is stored as bit 0. Feature `i` lives at bit `i % 64` of word `i / 64`. Bits past `K` are zero, so `K` does not have to be a multiple of 64.

For `a, b` in `{+1, -1}^K`:

```
dot(a, b) = K - 2 * popcount(a XOR b)
          = 2 * popcount(a XNOR b) - K
```

`binary_gemm` evaluates that for every pair of rows. `A` is `[M, K]`, `B` is `[N, K]`, both packed, and `C[m, n]` is the integer dot. The default call picks the kernel from the CPU and from `K`: one word per row uses a broadcast popcount, a full 512-bit row uses AVX-512 VPOPCNTDQ, a 256-bit row uses AVX2, and anything shorter stays on scalar `std::popcount`. Weight tiles are reused across the activation rows so a wide matmul does not reload `B` for every token.

`BitLinear` is the layer a network actually calls. The residual stream stays fp32. The forward pass is:

```
alpha   = mean(|x|)                         stop-gradient
sign(x) packed into bits
scale_n = mean(|W[n, :]|)                   stop-gradient
y_n     = alpha * scale_n * dot(sign(x), sign(W[n, :]))
```

Zero is a positive pole. Latent weights stay fp32 and train with a straight-through estimator: the forward uses `sign(w)`, and the backward treats that sign as the identity. The packed kernel is the forward matmul. The backward is a dense surrogate, which is what makes the signs movable.

Worked example, all of it exact in fp32:

```
x = [1, -2, 0, 3]          sign = [+1, -1, +1, +1]     alpha = 1.5
W = [0.5, -0.25, 0, -1]    sign = [+1, -1, +1, -1]     scale = 0.4375
dot = 2
y   = 1.5 * 0.4375 * 2 = 1.3125
```

## The block

```
h = x + BitLinear_out(Attention(BitLinear_qkv(RMSNorm(x))))
y = h + BitLinear_down(ReLU(BitLinear_up(RMSNorm(h)))^2)
```

Queries, keys, and values are projected with `BitLinear`, then RoPE rotates queries and keys. The score matrix, softmax, and value mix stay fp32, because those need a range a single bit does not have. The feed-forward uses squared ReLU, one up-projection and no gate. A final RMSNorm and a `BitLinear` head produce next-token logits. Decode recomputes the whole context; there is no KV cache.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/lodestone_bench
./build/lodestone --steps 800
```

`./build/lodestone` trains a one-layer model on the sentence `the lodestone keeps its poles` and greedily decodes it from the first character. It exits 0 when the sentence comes back whole.

```bash
./build/lodestone --steps 800 --seed 1 --log-every 50
```

## Using the library

```cpp
#include "lodestone/lodestone.hpp"

lodestone::Config cfg;
cfg.vocab = 32;
cfg.d_model = 64;
cfg.n_layers = 2;
cfg.n_heads = 4;
cfg.d_ff = 128;

lodestone::Transformer model(cfg, /*seed=*/1);
std::vector<int> tokens = {1, 5, 5, 9};
model.forward(tokens);
std::span<const float> logits = model.logits();  // [tokens, vocab]

lodestone::TrainConfig opt;
opt.lr = 0.001f;
lodestone::StepResult step = model.train_step(tokens, opt);
```

`train_step` zeroes gradients, runs forward and backward, clips the global grad norm, and applies AdamW. Set `use_adam` to false for SGD. Scales are not parameters. `BitLinear` latents are clamped to `[-1, 1]` after the update so a row's scale cannot run away. Adam still takes a roughly fixed-size step whenever a gradient is present, which will eventually flip poles that were already correct. The sentence demo stops at the first greedy reproduction for that reason.

`save` / `load` write the latent fp32 weights in a little-endian file whose magic is `LDSTONE1`. Optimizer moments are not stored. Loading packs the bits again.

The gemm is also a standalone entry point:

```cpp
std::vector<float> row = {1.f, -1.f, 1.f, 1.f};
std::vector<std::uint64_t> packed(lodestone::packed_words(row.size()));
lodestone::pack_signs(row, packed);
```

## What is left out

This is the kernel, the block, a trainer, and a checkpoint. It is not a pretrained language model. Activations inside `BitLinear` are signs plus one scale, not int8. Attention materializes the score matrix. Batch size is one sequence, with the token axis as the gemm row batch. There is no tokenizer, no KV cache, and no multi-device path.
