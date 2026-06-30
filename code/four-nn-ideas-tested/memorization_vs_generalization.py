"""Memorization vs generalization, and data closes the gap.

Claim: a high-capacity model trained on a tiny amount of noisy data can drive
its training loss to ~0 while its loss on held-out data remains high.  As the
dataset grows, the gap between training and test loss shrinks.

Build: a noisy linearly-separable classification task in R^d with label noise;
train the same strongly over-parameterized MLP on train sets of size 20, 200,
and 2000.  Record final train and test cross-entropy losses and plot the
generalization gap vs. dataset size.
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


DIM = 2
LABEL_NOISE = 0.25
TEST_SIZE = 2000
TRAIN_SIZES = [20, 200, 2000]
EPOCHS = 4000
LR = 0.01
SEED = 7


def make_noisy_data(d: int = DIM,
                    n_train_max: int = max(TRAIN_SIZES),
                    n_test: int = TEST_SIZE,
                    noise: float = LABEL_NOISE,
                    seed: int = SEED):
    """Generate a noisy learnable classification problem.

    True label y is determined by a simple linear rule on the first two
    features.  Then `noise` fraction of the labels are flipped uniformly at
    random.  The noise is what makes the task impossible to solve perfectly:
    any model that drives training loss to zero is memorizing noise rather
    than the rule.
    """
    torch.manual_seed(seed)

    def generate(n: int):
        X = torch.randn(n, d)
        # True rule: positive side of a hyperplane in the first two dims.
        score = 1.5 * X[:, 0] + 1.0 * X[:, 1] - 0.5
        y_true = (score > 0).float()
        flip = torch.rand(n) < noise
        y_noisy = y_true.clone()
        y_noisy[flip] = 1.0 - y_noisy[flip]
        return X, y_noisy.unsqueeze(1)

    X_train_full, y_train_full = generate(n_train_max)
    X_test, y_test = generate(n_test)
    return X_train_full, y_train_full, X_test, y_test


class OverParamMLP(nn.Module):
    """A 2-hidden-layer MLP.

    With input_dim=2 and hidden=32 this has ~1.1k parameters, i.e. many more
    than 20 data points but fewer than 2000.  Tiny data can be memorized;
    larger data forces the network to learn the true rule.
    """

    def __init__(self, input_dim: int = DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def bce_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Return average binary cross-entropy on a dataset."""
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(X)
        return criterion(logits, y).item()


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        return (preds == y).float().mean().item()


def train(model: nn.Module,
          X: torch.Tensor,
          y: torch.Tensor,
          epochs: int = EPOCHS,
          lr: float = LR,
          weight_decay: float = 5e-4) -> list[float]:
    """Train with Adam and mild weight decay; tiny sets can be memorized."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            history.append(loss.item())
            print(f"    epoch {epoch + 1:5d} | train loss {loss.item():.4f}")
    return history


def run_experiment():
    X_train_full, y_train_full, X_test, y_test = make_noisy_data()

    results = []

    for n_train in TRAIN_SIZES:
        print(f"\n=== Training size: {n_train} ===")
        model = OverParamMLP()
        Xn = X_train_full[:n_train]
        yn = y_train_full[:n_train]

        train(model, Xn, yn, epochs=EPOCHS, lr=LR)

        train_loss = bce_loss(model, Xn, yn)
        test_loss = bce_loss(model, X_test, y_test)
        train_acc = accuracy(model, Xn, yn)
        test_acc = accuracy(model, X_test, y_test)
        gap = test_loss - train_loss

        results.append({
            "n": n_train,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "gap": gap,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })

        print(f"  final train loss: {train_loss:.4f}  |  "
              f"test loss: {test_loss:.4f}  |  gap: {gap:.4f}")
        print(f"  final train acc : {train_acc:.2%}  |  "
              f"test acc : {test_acc:.2%}")

    return results


def plot_results(results: list[dict]):
    ns = [r["n"] for r in results]
    train_losses = [r["train_loss"] for r in results]
    test_losses = [r["test_loss"] for r in results]
    gaps = [r["gap"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: train vs test loss on a log scale for n.
    ax = axes[0]
    ax.plot(ns, train_losses, marker="o", label="train loss", linewidth=2)
    ax.plot(ns, test_losses, marker="s", label="test loss", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("training set size")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss vs. dataset size")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Right: the generalization gap.
    ax = axes[1]
    ax.plot(ns, gaps, marker="D", color="crimson", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlabel("training set size")
    ax.set_ylabel("test loss - train loss")
    ax.set_title("Generalization gap vs. dataset size")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Annotate the gap values above each point.
    for n, gap in zip(ns, gaps):
        ax.annotate(f"{gap:.3f}", (n, gap),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    fig.suptitle(
        "Memorization vs Generalization (Data Closes the Gap)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = "memorization_generalization_gap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")
    plt.show()


def main():
    results = run_experiment()
    print("\n--- Summary ---")
    print(f"{'n':>6}  {'train loss':>12}  {'test loss':>12}  {'gap':>12}  "
          f"{'train acc':>10}  {'test acc':>10}")
    for r in results:
        print(f"{r['n']:6d}  {r['train_loss']:12.4f}  {r['test_loss']:12.4f}  "
              f"{r['gap']:12.4f}  {r['train_acc']:10.2%}  {r['test_acc']:10.2%}")

    plot_results(results)


if __name__ == "__main__":
    main()
