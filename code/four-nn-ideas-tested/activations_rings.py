"""Why activations matter.

Generate two noisy concentric rings that are not linearly separable and compare:
(a) a single linear layer + sigmoid (logistic regression)
(b) one hidden layer with ReLU activations

The only architectural difference is the ReLU non-linearity; the second model
learns a closed, ring-shaped decision boundary while the first is stuck with
a straight line.
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def make_rings(n: int = 300,
               inner_radius: float = 1.0,
               outer_radius: float = 2.0,
               noise: float = 0.15,
               seed: int = 42):
    """Generate two noisy concentric rings.  Class 0 = inner, class 1 = outer."""
    torch.manual_seed(seed)
    n0 = n // 2
    n1 = n - n0

    theta0 = torch.rand(n0) * 2 * math.pi
    theta1 = torch.rand(n1) * 2 * math.pi

    r0 = torch.normal(mean=inner_radius, std=noise, size=(n0,))
    r1 = torch.normal(mean=outer_radius, std=noise, size=(n1,))

    x0 = torch.stack([r0 * torch.cos(theta0), r0 * torch.sin(theta0)], dim=1)
    x1 = torch.stack([r1 * torch.cos(theta1), r1 * torch.sin(theta1)], dim=1)

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).unsqueeze(1)
    return X, y


class LinearClassifier(nn.Module):
    """A single linear layer followed by a sigmoid."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReLUHiddenClassifier(nn.Module):
    """Two-layer network: Linear -> ReLU -> Linear -> Sigmoid."""
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        probs = model(X)
        preds = (probs >= 0.5).float()
        return (preds == y).float().mean().item()


def train(model: nn.Module,
          X: torch.Tensor,
          y: torch.Tensor,
          epochs: int = 4000,
          lr: float = 0.01) -> None:
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        probs = model(X)
        loss = criterion(probs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            acc = accuracy(model, X, y)
            print(f"  epoch {epoch + 1:5d} | loss {loss.item():.4f} | acc {acc:.4f}")


def plot_boundary(ax, model: nn.Module, X: torch.Tensor, y: torch.Tensor, title: str):
    """Plot data points and the 0.5-probability decision boundary."""
    # Build a fine meshgrid over the data range.
    padding = 0.5
    x_min, x_max = X[:, 0].min().item() - padding, X[:, 0].max().item() + padding
    y_min, y_max = X[:, 1].min().item() - padding, X[:, 1].max().item() + padding
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 300),
        torch.linspace(y_min, y_max, 300),
        indexing="xy",
    )
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)

    with torch.no_grad():
        zz = model(grid).reshape(xx.shape).numpy()

    ax.contourf(xx.numpy(), yy.numpy(), zz, levels=50, cmap="RdBu_r", alpha=0.6)
    ax.contour(xx.numpy(), yy.numpy(), zz, levels=[0.5], colors="black", linewidths=2)

    # Plot points.
    inner = (y.squeeze() == 0).nonzero(as_tuple=True)[0]
    outer = (y.squeeze() == 1).nonzero(as_tuple=True)[0]
    ax.scatter(X[inner, 0].numpy(), X[inner, 1].numpy(),
               c="blue", edgecolors="k", s=40, label="class 0 (inner)")
    ax.scatter(X[outer, 0].numpy(), X[outer, 1].numpy(),
               c="red", edgecolors="k", s=40, label="class 1 (outer)")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right")


def main():
    X, y = make_rings(n=300, inner_radius=1.0, outer_radius=2.0, noise=0.15, seed=13)

    print("(a) Training single linear layer + sigmoid...")
    linear_model = LinearClassifier()
    train(linear_model, X, y, epochs=4000, lr=0.05)
    linear_acc = accuracy(linear_model, X, y)

    print("\n(b) Training one ReLU hidden layer...")
    relu_model = ReLUHiddenClassifier(hidden=16)
    train(relu_model, X, y, epochs=4000, lr=0.01)
    relu_acc = accuracy(relu_model, X, y)

    print(f"\nFinal train accuracy:")
    print(f"  Linear model accuracy: {linear_acc:.2%}")
    print(f"  ReLU hidden layer accuracy: {relu_acc:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_boundary(axes[0], linear_model, X, y,
                  f"(a) Linear layer + sigmoid\naccuracy = {linear_acc:.1%}")
    plot_boundary(axes[1], relu_model, X, y,
                  f"(b) One ReLU hidden layer\naccuracy = {relu_acc:.1%}")

    fig.suptitle("Activations Exist for a Reason", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = "rings_decision_boundaries.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved decision-boundary plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
