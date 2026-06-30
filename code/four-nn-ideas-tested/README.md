# Four Neural Network Ideas, Tested

Source code for the blog post **[Four Neural Network Ideas, Tested](https://therahulbhati.github.io/posts/four-nn-ideas-tested/)**.

Each demo is a self-contained Python script that trains a small network and a paired HTML widget that runs the same logic in the browser.

## Demos

### 1. Activations exist for a reason
**Files:** `activations_rings.py` · `activations_rings.html`

A linear model and a ReLU model race to separate a red ring from a blue one. The linear model can't draw a curved boundary no matter how long it trains — without a nonlinear activation, any stack of layers collapses to one matrix multiply.

```bash
python activations_rings.py
```

---

### 2. Depth without nonlinearity is a lie
**Files:** `depth_without_nonlinearity.py` · `depth_without_nonlinearity.html`

Three networks on the same rings: 1 linear layer, 5 linear layers, 5 layers with ReLU. The two linear models track each other almost perfectly because W₅·W₄·W₃·W₂·W₁ is just one matrix. The script multiplies the learned weight matrices and verifies that the product produces exactly the same outputs as the sequential 5-layer model.

> **Note:** imports helpers from `activations_rings.py` — both files must be in the same directory.

```bash
python depth_without_nonlinearity.py
```

---

### 3. Embeddings learn similarity from next-token alone
**Files:** `embeddings_clustering.py` · `embeddings_clustering.html`

A tiny model trained only to predict the next word in a toy grammar. No similarity labels were ever supplied. After training, words that appear in the same contexts cluster together in embedding space — the similarity is a side effect of the prediction task.

```bash
python embeddings_clustering.py
```

---

### 4. Memorization vs generalization
**Files:** `memorization_vs_generalization.py` · `memorization_vs_generalization.html`

The same over-parameterized MLP trained on 20, 200, and 2,000 points with 25% label noise. The 20-point model memorizes its training set; the 2,000-point model can't, so it has to generalize. The train/test gap shrinks with more data — usually the fix, not a better model.

```bash
python memorization_vs_generalization.py
```

---

## Setup

Requires Python 3.9+ and PyTorch 2.0+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To deactivate the environment when done:

```bash
deactivate
```

## HTML widgets

Each `.html` file is a fully self-contained interactive demo — open it directly in any browser, no server or dependencies needed.
