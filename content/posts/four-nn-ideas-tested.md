---
date: '2026-06-30T10:00:00+05:30'
draft: false
title: 'Four Neural Network Ideas, Tested'
tags:
  - 'Machine Learning'
  - 'Neural Networks'
  - 'Deep Learning'
  - 'Interactive Demos'
  - 'Education'
categories:
  - 'Machine Learning'
cover:
  image: /images/four-nn-ideas-tested-cover.png
---

Most neural-network explanations start with math. That's honest. But the ideas stick when you've actually broken something. Each section below is a live demo: click Run, watch it train, change a control, run it again.

1. [Activations exist for a reason](#1-activations-exist-for-a-reason)
2. [Depth without nonlinearity is a lie](#2-depth-without-nonlinearity-is-a-lie)
3. [Embeddings learn similarity from next-token alone](#3-embeddings-learn-similarity-from-next-token-alone)
4. [Memorization vs generalization](#4-memorization-vs-generalization)

{{< rawhtml >}}
<script>
  window.addEventListener('message', function(e) {
    if (!e.data || e.data.type !== 'resize') return;
    var frames = document.querySelectorAll('iframe[data-autoresize]');
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow === e.source) {
        frames[i].style.height = (e.data.h) + 'px';
        break;
      }
    }
  });
</script>
{{< /rawhtml >}}

---

## 1. Activations exist for a reason

A linear model and a ReLU model, side by side, trying to separate a red ring from a blue one. The linear model can't draw a curved boundary no matter how long it trains. Without a nonlinear activation, a stack of layers collapses to one matrix multiply. The curved shape you need is impossible to express.

{{< rawhtml >}}
<iframe
  data-autoresize
  src="/nn-demos/s1_1_activations_rings_v2.html"
  width="100%"
  height="480"
  style="border:none;border-radius:12px;display:block;"
></iframe>
{{< /rawhtml >}}

---

## 2. Depth without nonlinearity is a lie

Three networks on the same rings: 1 linear layer, 5 linear layers, 5 layers with ReLU. The two linear models track each other almost perfectly, because W₅·W₄·W₃·W₂·W₁ is just one matrix. The Proof panel computes the product so you don't have to take my word for it.

Adding depth only helps if something nonlinear happens between layers.

{{< rawhtml >}}
<iframe
  data-autoresize
  src="/nn-demos/s1_2_depth_without_nonlinearity_v2.html"
  width="100%"
  height="520"
  style="border:none;border-radius:12px;display:block;"
></iframe>
{{< /rawhtml >}}

---

## 3. Embeddings learn similarity from next-token alone

A small model trained to predict the next word in a toy language. No one labeled which words are similar. The model only sees sequences. After training, click any word on the map to see its nearest neighbors.

Words that appear in the same contexts end up close together. The similarity is a side effect of the prediction task, not something anyone designed.

{{< rawhtml >}}
<iframe
  data-autoresize
  src="/nn-demos/s1_3_embeddings_clustering_v3.html"
  width="100%"
  height="620"
  style="border:none;border-radius:12px;display:block;"
></iframe>
{{< /rawhtml >}}

---

## 4. Memorization vs generalization

The same network, three times, on 20, 200, and 2,000 points. Some labels are wrong on purpose. The 20-point model just memorizes its training set; the 2,000-point model can't, so it has to generalize.

The train/test gap shrinks with more data. That's usually the fix, not a better model.

{{< rawhtml >}}
<iframe
  data-autoresize
  src="/nn-demos/s1_4_memorization_vs_generalization_v2.html"
  width="100%"
  height="560"
  style="border:none;border-radius:12px;display:block;"
></iframe>
{{< /rawhtml >}}

---

The source (Python included) is on [GitHub](https://github.com/therahulbhati/therahulbhati.github.io/tree/main/code/four-nn-ideas-tested).