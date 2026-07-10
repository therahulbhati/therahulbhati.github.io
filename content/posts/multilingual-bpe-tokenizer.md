---
date: '2026-07-10T21:30:00+05:30'
draft: false
title: 'One Vocabulary, Four Languages: A BPE Tokenizer From Scratch'
tags:
  - 'Machine Learning'
  - 'Tokenization'
  - 'BPE'
  - 'NLP'
  - 'Multilingual'
  - 'Interactive Demos'
categories:
  - 'Machine Learning'
cover:
  image: /images/multilingual-bpe-tokenizer-cover.png
  alt: 'Hand-drawn comic of tokenization in transformers: raw text is broken into subword tokens, tokens become IDs, and the transformer model processes the IDs'
---

Commercial tokenizers quietly charge some languages more than others. I wanted to measure that myself. Then I wanted to see how far from-scratch engineering can push against it. So, an experiment. Take the Wikipedia **India** page in English, Hindi, Telugu, and one more language. Build a BPE tokenizer from scratch with a single 10,000-token vocabulary shared across all four. Optimize one number: `1000 / (X_max − X_min)`, where each `X` is fertility, the tokens a language needs per word. The metric only rewards *balance*. A tokenizer that's great at English and bad at Telugu scores worse than one that's mediocre at both.

That constraint sounds artificial. It isn't. It forces you into every real problem multilingual tokenizers have, one at a time. The finished tokenizer runs live at the end of this post: every number computed in your browser from the shipped merge table, nothing hardcoded.

1. [Fertility, and why the metric rewards balance](#1-fertility-and-why-the-metric-rewards-balance)
2. [BPE over codepoints, from scratch](#2-bpe-over-codepoints-from-scratch)
3. [Exact fertility ladders](#3-exact-fertility-ladders)
4. [The fourth language is an economics problem](#4-the-fourth-language-is-an-economics-problem)
5. [Never emit unk](#5-never-emit-unk)
6. [The markdown twist](#6-the-markdown-twist)
7. [The tokenizer, live](#7-the-tokenizer-live)
8. [What the numbers mean, honestly](#8-what-the-numbers-mean-honestly)

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

## 1. Fertility, and why the metric rewards balance

Fertility is tokens divided by words. A fertility of 1.0 means every word is a single token. A fertility of 2.0 means the tokenizer chops the average word in half. Ahia et al. (2023) measured the premium commercial tokenizers charge non-English languages and named it the *fertility tax*. The same sentence costs a Telugu speaker 2-3x the tokens it costs an English speaker. This experiment reproduces that tax, then fights it.

The score, `1000 / (X_max − X_min)`, only cares about the *gap* between your best and worst language. You can't win by making English great. You win by making four languages identical. That means deliberately spending vocabulary budget where it hurts most.

---

## 2. BPE over codepoints, from scratch

The core algorithm is Sennrich 2016: start with characters, repeatedly merge the most frequent adjacent pair, stop at the budget. Two implementation choices mattered.

**Codepoints, not bytes.** Indic characters are 3 bytes in UTF-8. Byte-level BPE would burn thousands of merges just reassembling single characters. Operating on Unicode codepoints keeps the base alphabet small and every merge meaningful.

**Incremental training.** The naive loop rescans the whole corpus after every merge, O(merges × corpus). The fix is a global pair-frequency table, a pair→words index, and a lazy max-heap. Each merge then only touches the words it actually changes. Same merges, ~100x faster. That speed is what made everything in section 3 affordable, because the optimizer retrains constantly.

---

## 3. Exact fertility ladders

Here's the observation that turned tuning from guesswork into arithmetic: when you train BPE on the same text you measure fertility on, **each merge's pair frequency is exactly the number of tokens it removes**. So fertility as a function of "how many merges did this language get" isn't something you estimate. It's a precomputed curve, exact to the token.

With one curve per language, allocation becomes a greedy loop: give the next merge to whichever language currently has the worst fertility. Tail merges move fertility by ~1/words, about 10⁻⁴, so the four curves can be pinned nearly equal. A short hill-climb against the real assembled tokenizer polishes the last crumbs.

On plaintext, this pinned all four languages to fertilities matching within **5 × 10⁻⁵** of each other.

The clean version of that story hides one bug worth sharing: merge lists from different languages interfere when languages share strings. English's high-ranked merges eat Telugu's URL patterns mid-word, stranding Telugu's longer merges. The fix is in section 6.

---

## 4. The fourth language is an economics problem

Three languages were fixed from the start: English, Hindi, Telugu. The fourth slot was free. I picked Bengali first: big language, big Wikipedia. Wrong call, and the reason is instructive.

What matters isn't the language's importance. It's **how many merges its India page consumes**. Bengali's page needed 3,433 merges to reach the target fertility. That pushed the four-language total past the budget, and everyone got stuck around 1.24. The Kannada India page is compact (1,574 words, 581 unique) and needed ~1,100 merges. Swapping Bengali for Kannada freed enough budget to pull *all four* languages below the bar.

I'd assumed Kannada would help because it's Telugu's sibling. It isn't that: Kannada and Telugu occupy different Unicode blocks and share exactly zero merges. The win is pure page economics.

---

## 5. Never emit unk

One rule in this experiment is blunt: if the tokenizer ever produces an `<unk>` on evaluation text, the run scores zero. A real tokenizer doesn't get to reject input. Hoping your alphabet covers the test text isn't a strategy. One stray curly quote and you're dead.

So `<unk>` doesn't exist in this tokenizer at all. The vocabulary reserves 256 **byte-fallback tokens** (the SentencePiece trick): any character outside the alphabet encodes as its UTF-8 bytes and decodes back losslessly. Emoji, CJK, IPA, anything. That costs 256 of the 10,000 slots. It buys structural immunity to the one rule that zeroes everything.

A close cousin of this bug nearly shipped: an earlier version lowercased English. It gave cleaner statistics. It also meant every capital letter in anyone else's rendition of the text would hit the fallback. Case stays.

---

## 6. The markdown twist

Halfway through, I changed the evaluation text on myself. Clean plaintext is a lab condition. The page an LLM actually ingests is **HTML converted to Markdown, links preserved**, and in the wild you don't get to choose the converter. Markdown changes the game completely. Suddenly all four pages are dominated by shared ASCII: URLs, `](/wiki/` syntax, percent-encoded paths where a single Devanagari character becomes nine ASCII characters.

That's what broke the per-language merge lists (the bug from section 3): four languages independently learning *different* decompositions of the *same* ASCII strings, then fighting over them at encode time. Measured fertility came out several units worse than the model. Fragmentation, not noise.

The architecture that works splits the problem where the scripts actually split:

- **One shared ASCII pool.** Every maximal ASCII run from all four pages trains a single merge list: one canonical construction per string, zero interference. English is pure ASCII, so this pool serves it entirely.
- **One native ladder per Indic language.** Trained only on native-script runs. Devanagari, Telugu, and Kannada are disjoint Unicode blocks, so the ladders provably can't conflict, and no merge ever crosses an ASCII↔native boundary.

Token counts become exactly additive per run. The fertility model is exact again. Allocation reduces to four knobs: shared-pool depth plus three native depths.

---

## 7. The tokenizer, live

Everything below is computed in your browser, right now, from the shipped tokenizer: fertilities, spread, score, the ≤ 1.2 check. Type your own text to watch it tokenize. Download the full token list, or the tokenizer itself.

{{< rawhtml >}}
<iframe
  data-autoresize
  src="/tokenizer/"
  width="100%"
  height="1400"
  style="border:none;border-radius:12px;display:block;"
></iframe>
{{< /rawhtml >}}

Standalone version: [therahulbhati.github.io/tokenizer](/tokenizer/).

---

## 8. What the numbers mean, honestly

Every number above reproduces from the saved tokenizer file. A claim that doesn't survive re-running the artifact isn't a result. Two disclosures matter more than any score:

**The denominator is a choice.** Fertility is tokens divided by words, and "word" isn't a God-given unit. Whitespace-split markdown, `\w+` matches, punctuation-split plaintext: the same tokenizer's fertility swings by more than a full unit across conventions. The demo reports several so no claim depends on which one you'd pick.

**Training on the text you measure is overfitting, and here it's deliberate.** This experiment measures the ceiling: how balanced can four languages get when the tokenizer is allowed to see the evaluation pages? An earlier version kept the eval pages held out. It trained on ~50 *different* India-topic articles, balanced by cross-validation, and scored a fraction of this one. The reason: ~28% of Telugu and Bengali words on an unseen page never appear in training, and a word the tokenizer has never seen must split. That gap is the measured cost of generalization. It's the fertility tax again: for the long tail of languages there's no trick, only more data.

---

The source (the from-scratch BPE, the ladder optimizer, and the widget build) is on [GitHub](https://github.com/therahulbhati/therahulbhati.github.io/tree/main/code/multilingual-bpe-tokenizer).
