"""Depth without nonlinearity is a lie.

Claim: five stacked linear layers collapse to a single linear map, so a
5-layer linear net is no stronger than 1 layer; both fail the ring task
identically, and inserting ReLUs between the same five layers suddenly
solves it.

This script:
  1. Builds the same noisy concentric-ring data as activations_rings.py.
  2. Trains a single linear layer.
  3. Trains five linear layers (no activations between them).
  4. Trains five layers with ReLU between every pair.
  5. Shows that the 1-layer and 5-linear-layer models both learn a straight
     decision boundary and achieve the same limited accuracy.
  6. Multiplies the five learned weight matrices and verifies that the
     resulting single matrix produces exactly the same outputs as the
     sequential 5-layer model.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from activations_rings import make_rings, accuracy, train, plot_boundary


class RingConfig(TypedDict):
    n: int
    inner_radius: float
    outer_radius: float
    noise: float
    seed: int


RING_CONFIG: RingConfig = {
    "n": 300,
    "inner_radius": 1.0,
    "outer_radius": 2.0,
    "noise": 0.15,
    "seed": 13,
}


def extract_linear_params(model: nn.Sequential):
    """Return a list of (weight, bias) tuples for every nn.Linear layer."""
    params = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            params.append((layer.weight.data, layer.bias.data))
    return params


def collapse_linear_layers(params):
    """Compute the single affine map equivalent to a stack of Linear layers.

    PyTorch Linear computes y = x @ W.T + b.  Composing layers gives
        y = x @ W_1.T @ W_2.T @ ... @ W_k.T + combined_bias
    so the effective weight is W_eff = W_k @ ... @ W_2 @ W_1 and the
    effective bias absorbs each layer's bias.

    Returns the effective (weight, bias) tuple.
    """
    *leading, last = params
    W_eff, b_eff = last

    # Walk backwards from layer k-1 to layer 1.  At each step W_eff equals
    # W_k @ W_{k-1} @ ... @ W_{i+1}, so b_i @ W_eff.T is exactly the bias
    # contribution of layer i after it has propagated through the later layers.
    for W, b in reversed(leading):
        b_eff = b_eff + b @ W_eff.T
        W_eff = W_eff @ W

    return W_eff, b_eff


class DeepLinearClassifier(nn.Module):
    """Five linear layers with no intervening activations."""
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepReLUClassifier(nn.Module):
    """Five linear layers with ReLU between each pair."""
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def single_layer_equivalent(params, x: torch.Tensor):
    """Return x @ W_eff.T + b_eff for the collapsed linear stack."""
    W_eff, b_eff = collapse_linear_layers(params)
    return x @ W_eff.T + b_eff


def main():
    X, y = make_rings(**RING_CONFIG)

    print("(a) Training single linear layer + sigmoid...")
    single = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    train(single, X, y, epochs=4000, lr=0.05)
    single_acc = accuracy(single, X, y)
    print(f"    -> accuracy: {single_acc:.2%}\n")

    print("(b) Training five stacked linear layers (no activations)...")
    deep_linear = DeepLinearClassifier(hidden=8)
    train(deep_linear, X, y, epochs=4000, lr=0.01)
    deep_linear_acc = accuracy(deep_linear, X, y)
    print(f"    -> accuracy: {deep_linear_acc:.2%}\n")

    print("(c) Training five layers with ReLU between each layer...")
    deep_relu = DeepReLUClassifier(hidden=8)
    train(deep_relu, X, y, epochs=4000, lr=0.01)
    deep_relu_acc = accuracy(deep_relu, X, y)
    print(f"    -> accuracy: {deep_relu_acc:.2%}\n")

    # --- Bonus: collapse the 5 linear weight matrices into one. ---
    linear_params = extract_linear_params(deep_linear.net)
    W_eff, b_eff = collapse_linear_layers(linear_params)

    print("--- Bonus: matrix product collapse ---")
    print(f"Number of Linear layers in deep-linear model: {len(linear_params)}")
    for i, (W, _) in enumerate(linear_params, start=1):
        print(f"  W_{i} shape: {tuple(W.shape)}")
    print(f"Effective W shape: {tuple(W_eff.shape)}   (should be (1, 2))")
    print(f"Effective b shape: {tuple(b_eff.shape)}   (should be (1,))")
    print("Effective matrix (W_eff | b_eff):", torch.cat([W_eff, b_eff.unsqueeze(1)], dim=1).numpy())

    with torch.no_grad():
        sequential_logits = deep_linear.net[:-1](X).squeeze()
        collapsed_logits = single_layer_equivalent(linear_params, X).squeeze()
        max_diff = (sequential_logits - collapsed_logits).abs().max().item()

    print(f"\nMax |output difference| between 5 linear layers and their collapsed matrix: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("The five linear layers are numerically identical to a single linear map.")
    else:
        print("WARNING: outputs do not match; check the matrix multiplication order.")

    # --- Visualization. ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_boundary(axes[0], single, X, y,
                  f"(a) 1 linear layer\naccuracy = {single_acc:.1%}")
    plot_boundary(axes[1], deep_linear, X, y,
                  f"(b) 5 linear layers\naccuracy = {deep_linear_acc:.1%}")
    plot_boundary(axes[2], deep_relu, X, y,
                  f"(c) 5 layers + ReLU\naccuracy = {deep_relu_acc:.1%}")

    fig.suptitle("Depth Without Nonlinearity is a Lie", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = "rings_depth_without_nonlinearity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved decision-boundary plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
