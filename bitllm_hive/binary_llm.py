"""Toy 1-bit (binary) weight layers using straight-through sign binarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def binarize(x: np.ndarray) -> np.ndarray:
    """Return +1/-1 binary tensor shaped like `x`."""
    return np.where(x >= 0, np.float32(1), np.float32(-1))


@dataclass
class BinaryLinear:
    """A 1-bit weight linear layer.

    Weights are stored as full-precision `real_W` but forwarded as
    ``W_b = sign(real_W)``.  A per-output scale ``α = mean(|real_W|)``
    is applied so the layer approximates the full-precision forward pass.
    """

    in_features: int
    out_features: int

    def __post_init__(self):
        rng = np.random.default_rng(0)
        self.real_W = rng.normal(0, 0.1, size=(self.out_features, self.in_features)).astype(np.float32)
        self.scale = np.mean(np.abs(self.real_W), axis=1, keepdims=True)
        self.binarized_W = binarize(self.real_W)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_b = binarize(x)
        return (x_b @ self.binarized_W.T) * self.scale.reshape(1, -1)


class BinaryMLP:
    """Small 1-bit MLP demo.

    Not a real language model — it shows how 1-bit layers can be wired
    into a classifier/regressor.  Use it as a tiny scorer or as a
    pluggable head inside a coding expert.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
    ):
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers = [BinaryLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers[:-1]:
            x = np.maximum(layer.forward(x), 0)  # ReLU-ish on binary features
        return self.layers[-1].forward(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
