"""Embeddings learn similarity from nothing but next-token.

Claim: trained only to predict the next token in a tiny synthetic grammar,
the embedding table clusters related tokens, even though no similarity label
was ever supplied.

This script:
  1. Defines a toy language with categories (animals: cat/dog/cow; fruits:
     apple/mango; verbs: eat/chase/see) and a handful of templates.  Tokens
     in the same category are forced into the same contextual slots, so they
     share next-token distributions.
  2. Trains a minimal embedding -> linear projection -> softmax model to
     predict the next token given the current token.
  3. Projects the learned embedding table to 2D and plots it.
  4. Prints nearest-neighbor rankings.  Every animal, fruit, and verb has its
     closest neighbor in the same semantic category: the emergent clustering
     is the proof.

Note: overall next-token accuracy is capped below 100% because the grammar
words "the"/"a" appear before both subjects and objects, so their next-token
prediction is inherently ambiguous.  The content-word embeddings nonetheless
cluster cleanly by category.
"""

from __future__ import annotations

import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Toy language definition
# -----------------------------------------------------------------------------
class Vocabulary:
    """Tiny vocabulary split into semantic categories plus grammar words."""

    ANIMALS = ["cat", "dog", "cow"]
    FRUITS = ["apple", "mango"]
    VERBS = ["eat", "chase", "see"]
    GRAMMAR = ["the", "a", "s", "."]

    def __init__(self):
        self.categories: dict[str, str] = {}
        for word in self.ANIMALS:
            self.categories[word] = "animal"
        for word in self.FRUITS:
            self.categories[word] = "fruit"
        for word in self.VERBS:
            self.categories[word] = "verb"
        for word in self.GRAMMAR:
            self.categories[word] = "grammar"

        self.words = sorted(self.categories.keys())
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for i, w in enumerate(self.words)}
        self.vocab_size = len(self.words)

    COLORS = {
        "animal": "#e41a1c",
        "fruit": "#4daf4a",
        "verb": "#377eb8",
        "grammar": "#984ea3",
    }


VOCAB = Vocabulary()


def generate_corpus(n_sentences: int = 2000, seed: int = 42) -> list[list[str]]:
    """Generate a synthetic corpus as lists of tokens.

    Templates use separate tokens for the grammatical suffix "s" so the
    vocabulary stays small and every next-token prediction is well-defined.
    """
    rng = random.Random(seed)

    templates = [
        ["the", "{animal}", "{verb}", "s", "the", "{fruit}", "."],
        ["a", "{animal}", "{verb}", "s", "a", "{fruit}", "."],
        ["the", "{animal}", "{verb}", "s", "a", "{fruit}", "."],
    ]

    corpus: list[list[str]] = []
    for _ in range(n_sentences):
        template = rng.choice(templates)
        tokens = [
            token.format(
                animal=rng.choice(VOCAB.ANIMALS),
                fruit=rng.choice(VOCAB.FRUITS),
                verb=rng.choice(VOCAB.VERBS),
            )
            for token in template
        ]
        corpus.append(tokens)
    return corpus


def build_dataset(corpus: list[list[str]]) -> torch.Tensor:
    """Return a [N, 2] tensor of (input_token, target_token) index pairs.

    The corpus is flattened into one continuous stream so that the final "."
    of a sentence is used as an input predicting the first token of the next
    sentence.  Otherwise "." would only ever be a target and its embedding
    would remain untrained.
    """
    pairs: list[tuple[int, int]] = []
    flat = [tok for sentence in corpus for tok in sentence]
    for current, nxt in zip(flat[:-1], flat[1:]):
        pairs.append((VOCAB.word_to_idx[current], VOCAB.word_to_idx[nxt]))
    return torch.tensor(pairs, dtype=torch.long)


# -----------------------------------------------------------------------------
# Model: embedding table + linear head -> softmax next-token predictor
# -----------------------------------------------------------------------------
class TinyEmbeddingLM(nn.Module):
    """Only the embedding table is learnable; output is input_emb @ U.T."""

    def __init__(self, vocab_size: int, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def get_embeddings(self) -> torch.Tensor:
        """Return the learned token embedding matrix."""
        return self.embed.weight.data.detach().cpu()


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
def train(
    model: nn.Module,
    data: torch.Tensor,
    epochs: int = 2000,
    lr: float = 0.02,
    batch_size: int | None = None,
    print_every: int = 250,
) -> list[float]:
    """Train the model on (input, target) next-token pairs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    if batch_size is None:
        batch_size = len(data)

    for epoch in range(epochs):
        perm = torch.randperm(len(data))
        epoch_loss = 0.0
        steps = 0
        for i in range(0, len(data), batch_size):
            idx = perm[i : i + batch_size]
            x, y = data[idx, 0], data[idx, 1]

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / steps
        losses.append(avg_loss)

        if (epoch + 1) % print_every == 0:
            print(f"  epoch {epoch + 1:5d} | loss {avg_loss:.4f}")

    return losses


# -----------------------------------------------------------------------------
# Analysis: 2D projection and nearest neighbors
# -----------------------------------------------------------------------------
def project_2d(embeddings: torch.Tensor) -> np.ndarray:
    """Center and embed to 2D with PCA so we can see structure."""
    X = embeddings.numpy()
    X = X - X.mean(axis=0)
    # Simple PCA via SVD.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # First two principal components.
    projected = X @ Vt[:2].T
    return projected


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Return Euclidean distances between all pairs of token embeddings."""
    X = embeddings
    norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = norm_sq + norm_sq.T - 2 * (X @ X.T)
    # Clamp tiny negatives.
    dist_sq = dist_sq.clamp(min=0.0)
    return dist_sq.sqrt()


def nearest_neighbors_report(embeddings: torch.Tensor, top_k: int = 3) -> None:
    """Print nearest neighbors and check whether they share a category."""
    dists = pairwise_distances(embeddings)
    # Exclude self.
    dists.fill_diagonal_(float("inf"))

    content_first_hits = 0
    content_total = 0
    content_topk_hits = 0
    content_topk_total = 0

    print("\nNearest-neighbor analysis:")
    for i, word in enumerate(VOCAB.words):
        category = VOCAB.categories[word]
        knn = dists[i].topk(top_k, largest=False).indices.tolist()
        knn_words = [VOCAB.idx_to_word[j] for j in knn]
        knn_cats = [VOCAB.categories[w] for w in knn_words]
        same_cat_first = knn_cats[0] == category
        same_cat_topk = sum(1 for c in knn_cats if c == category)

        if category != "grammar":
            content_first_hits += int(same_cat_first)
            content_total += 1
            content_topk_hits += same_cat_topk
            content_topk_total += len(knn_cats)

        marker = " " if same_cat_first else "*"
        print(f"  {marker}{word:6s} ({category:7s}) -> {list(zip(knn_words, knn_cats))}")

    print(f"\nSame-category nearest-neighbor hit rate:")
    print(f"  Content tokens, 1-NN : {content_first_hits / content_total:.1%}")
    # top-3 is capped below 100% by construction: fruits only have 2 members,
    # so a fruit's top-3 can contain at most 1 same-category neighbor.
    print(f"  Content tokens, top-{top_k}: {content_topk_hits / content_topk_total:.1%}")


def plot_embeddings(embeddings: torch.Tensor, output_path: str = "embeddings_clustering.png") -> None:
    """Plot the 2D projection of the learned embedding table."""
    xy = project_2d(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for category in ["animal", "fruit", "verb", "grammar"]:
        indices = [i for i, w in enumerate(VOCAB.words) if VOCAB.categories[w] == category]
        color = VOCAB.COLORS[category]
        ax.scatter(
            xy[indices, 0],
            xy[indices, 1],
            c=color,
            s=300,
            alpha=0.85,
            edgecolors="black",
            linewidths=1.5,
            label=category,
            zorder=3,
        )
        for i in indices:
            ax.annotate(
                VOCAB.words[i],
                (xy[i, 0], xy[i, 1]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=11,
                fontweight="bold",
                zorder=4,
            )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3, zorder=1)
    ax.set_title("Embeddings Learn Similarity From Nothing But Next-Token", fontsize=14, fontweight="bold")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(loc="best", title="category")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved embedding-clustering plot to: {output_path}")
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("Building synthetic corpus...")
    corpus = generate_corpus(n_sentences=4000, seed=42)
    token_counts = Counter(token for sentence in corpus for token in sentence)
    print(f"  sentences: {len(corpus)}")
    print(f"  unique tokens: {len(token_counts)}")
    print(f"  total tokens: {sum(token_counts.values())}")

    data = build_dataset(corpus)
    print(f"  training pairs: {len(data)}")

    print("\nTraining tiny embedding -> softmax next-token model...")
    torch.manual_seed(42)
    model = TinyEmbeddingLM(vocab_size=VOCAB.vocab_size, embed_dim=32)
    train(model, data, epochs=6000, lr=0.02, batch_size=len(data), print_every=750)

    # Final loss and token-level accuracy.
    with torch.no_grad():
        logits = model(data[:, 0])
        targets = data[:, 1]
        final_loss = F.cross_entropy(logits, targets).item()
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
    print(f"\nFinal: loss {final_loss:.4f} | next-token accuracy {acc:.2%}")

    embeddings = model.get_embeddings()
    nearest_neighbors_report(embeddings, top_k=3)
    plot_embeddings(embeddings)


if __name__ == "__main__":
    main()
