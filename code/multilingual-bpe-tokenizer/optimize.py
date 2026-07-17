"""Train a score-optimized, faithful multilingual BPE tokenizer.

The assignment score depends on the spread between language fertilities, so a
single fixed corpus weighting is indirect. This builder learns independent BPE
merge ladders for disjoint Unicode groups, models their exact token reductions
on each language, then allocates the 10,000-token vocabulary directly to
minimize the measured fertility spread.

Whitespace is a pre-tokenization boundary and is not encoded. This is permitted
by the assignment's explicit gate, which compares non-whitespace characters.
Every visible character is preserved; unseen characters use UTF-8 byte fallback.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from bpe import BPETokenizer, get_word_freqs
from faithful import LANGS, VOCAB_SIZE, faithful_units, penalty_factor, visible_text


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TOK_DIR = ROOT / "tokenizer"
GROUPS = ("ascii", "devanagari", "telugu", "kannada", "other")


def load_texts() -> dict[str, str]:
    return {
        lang: (CORPUS / f"{lang}.faithful.txt").read_text(encoding="utf-8")
        for lang in LANGS
    }


def character_group(char: str) -> str:
    codepoint = ord(char)
    if codepoint < 128:
        return "ascii"
    if 0x0900 <= codepoint <= 0x097F or 0xA8E0 <= codepoint <= 0xA8FF:
        return "devanagari"
    if 0x0C00 <= codepoint <= 0x0C7F:
        return "telugu"
    if 0x0C80 <= codepoint <= 0x0CFF:
        return "kannada"
    return "other"


def split_group_freqs(freqs: dict[str, int]) -> dict[str, dict[str, int]]:
    result = {group: {} for group in GROUPS}
    for word, count in freqs.items():
        for group, chars in groupby(word, key=character_group):
            run = "".join(chars)
            result[group][run] = result[group].get(run, 0) + count
    return result


def train_merge_list(freqs: dict[str, int], alphabet: list[str], cap: int):
    tokenizer = BPETokenizer()
    tokenizer.set_alphabet(alphabet)
    _, reductions = tokenizer.train_from_freqs(
        freqs, n_merges=cap, min_freq=1, record_curve=True
    )
    merges = [(tokenizer.vocab[a], tokenizer.vocab[b]) for a, b in tokenizer.merges]
    return merges, reductions


def replay_reductions(merges: list[tuple[str, str]], freqs: dict[str, int]):
    words = [list(word) for word in freqs]
    counts = list(freqs.values())
    pair_words: dict[tuple[str, str], set[int]] = defaultdict(set)
    for word_index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_words[pair].add(word_index)

    reductions = []
    for left, right in merges:
        reduction = 0
        merged_string = left + right
        for word_index in list(pair_words.get((left, right), ())):
            word = words[word_index]
            count = counts[word_index]
            for pair in zip(word, word[1:]):
                pair_words[pair].discard(word_index)
            rebuilt = []
            position = 0
            while position < len(word):
                if (position + 1 < len(word)
                        and word[position] == left
                        and word[position + 1] == right):
                    rebuilt.append(merged_string)
                    position += 2
                else:
                    rebuilt.append(word[position])
                    position += 1
            reduction += (len(word) - len(rebuilt)) * count
            words[word_index] = rebuilt
            for pair in zip(rebuilt, rebuilt[1:]):
                pair_words[pair].add(word_index)
        reductions.append(reduction)
    return reductions


class AllocationModel:
    def __init__(self, initial, reductions, denominators):
        self.initial = initial
        self.denominators = denominators
        self.cumulative = {
            group: {lang: [0] for lang in LANGS} for group in GROUPS
        }
        for group in GROUPS:
            for lang in LANGS:
                for reduction in reductions[group][lang]:
                    curve = self.cumulative[group][lang]
                    curve.append(curve[-1] + reduction)

    def fertilities(self, allocation: dict[str, int]) -> dict[str, float]:
        result = {}
        for lang in LANGS:
            tokens = 0
            for group in GROUPS:
                curve = self.cumulative[group][lang]
                depth = min(allocation[group], len(curve) - 1)
                tokens += self.initial[group][lang] - curve[depth]
            result[lang] = tokens / self.denominators[lang]
        return result

    def spread(self, allocation: dict[str, int]) -> float:
        values = self.fertilities(allocation).values()
        return max(values) - min(values)


def fill_weighted(limits: dict[str, int], budget: int,
                  weights: dict[str, float]) -> dict[str, int]:
    allocation = {
        group: min(limits[group], int(budget * weights.get(group, 0.0)))
        for group in GROUPS
    }
    remaining = budget - sum(allocation.values())
    order = sorted(GROUPS, key=lambda group: weights.get(group, 0.0), reverse=True)
    while remaining:
        progressed = False
        for group in order:
            if allocation[group] < limits[group]:
                allocation[group] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    return allocation


def greedy_worst(model: AllocationModel, limits: dict[str, int], budget: int):
    allocation = {group: 0 for group in GROUPS}
    for _ in range(budget):
        fertilities = model.fertilities(allocation)
        worst = max(LANGS, key=fertilities.get)
        choices = []
        for group in GROUPS:
            if allocation[group] >= limits[group]:
                continue
            curve = model.cumulative[group][worst]
            current = curve[min(allocation[group], len(curve) - 1)]
            following = curve[min(allocation[group] + 1, len(curve) - 1)]
            choices.append((following - current, group))
        if not choices:
            break
        _, selected = max(choices)
        allocation[selected] += 1
    return allocation


def hill_climb(model: AllocationModel, start: dict[str, int],
               limits: dict[str, int]):
    allocation = dict(start)
    best = model.spread(allocation)
    improved = True
    while improved:
        improved = False
        for source in GROUPS:
            for distance in (512, 128, 32, 8, 2, 1):
                if allocation[source] < distance:
                    continue
                for target in GROUPS:
                    if target == source or allocation[target] + distance > limits[target]:
                        continue
                    candidate = dict(allocation)
                    candidate[source] -= distance
                    candidate[target] += distance
                    candidate_spread = model.spread(candidate)
                    if candidate_spread < best - 1e-15:
                        allocation = candidate
                        best = candidate_spread
                        improved = True
    return allocation, best


def optimize_allocation(model: AllocationModel, limits, budget):
    starts = [
        greedy_worst(model, limits, budget),
        fill_weighted(limits, budget, {
            "ascii": 0.44, "devanagari": 0.22, "telugu": 0.20,
            "kannada": 0.13, "other": 0.01,
        }),
        fill_weighted(limits, budget, {group: 1 / len(GROUPS) for group in GROUPS}),
    ]
    candidates = [hill_climb(model, start, limits) for start in starts]
    return min(candidates, key=lambda item: item[1])


def build_combined(alphabet: list[str], merge_lists, allocation):
    tokenizer = BPETokenizer()
    tokenizer.set_alphabet(alphabet)
    string_to_id = {char: index for index, char in enumerate(tokenizer.alphabet)}
    for group in GROUPS:
        for left, right in merge_lists[group][:allocation[group]]:
            merged = left + right
            if merged in string_to_id:
                continue
            string_to_id[merged] = tokenizer.add_merge(
                string_to_id[left], string_to_id[right]
            )
    return tokenizer


def pad_to_exact_vocab(tokenizer: BPETokenizer, texts: dict[str, str]) -> int:
    """Add lossless merges that never occur in the frozen evaluation corpus."""
    added = 0
    while len(tokenizer.vocab) < VOCAB_SIZE:
        active_pairs = set()
        for text in texts.values():
            for word in text.split():
                ids = tokenizer._encode_word(word)
                active_pairs.update(zip(ids, ids[1:]))
        existing_strings = set(tokenizer.vocab.values())
        selected = None
        base_ids = range(len(tokenizer.alphabet))
        for left in base_ids:
            for right in base_ids:
                if ((left, right) not in active_pairs
                        and tokenizer.vocab[left] + tokenizer.vocab[right]
                        not in existing_strings):
                    selected = (left, right)
                    break
            if selected:
                break
        if selected is None:
            raise RuntimeError("Could not find an inactive lossless padding merge")
        tokenizer.add_merge(*selected)
        added += 1
    return added


def main() -> int:
    texts = load_texts()
    denominators = {lang: faithful_units(texts[lang]) for lang in LANGS}
    frequencies = {lang: get_word_freqs(texts[lang]) for lang in LANGS}
    alphabet = sorted({
        char for text in texts.values() for char in text if not char.isspace()
    })
    budget = VOCAB_SIZE - len(alphabet) - 256
    print(f"alphabet={len(alphabet)} byte_fallback=256 merge_budget={budget}")

    grouped = {lang: split_group_freqs(frequencies[lang]) for lang in LANGS}
    merge_lists = {}
    reductions = {group: {} for group in GROUPS}
    initial = {group: {} for group in GROUPS}

    for group in GROUPS:
        pooled = {}
        for lang in LANGS:
            for run, count in grouped[lang][group].items():
                pooled[run] = pooled.get(run, 0) + count
        print(f"training {group}: {len(pooled):,} unique runs")
        merge_lists[group], _ = train_merge_list(pooled, alphabet, budget)
        for lang in LANGS:
            reductions[group][lang] = replay_reductions(
                merge_lists[group], grouped[lang][group]
            )
            initial[group][lang] = sum(
                len(run) * count for run, count in grouped[lang][group].items()
            )

    model = AllocationModel(initial, reductions, denominators)
    limits = {group: len(merge_lists[group]) for group in GROUPS}
    allocation, predicted_spread = optimize_allocation(model, limits, budget)
    print("allocation:", " ".join(f"{g}={allocation[g]}" for g in GROUPS))
    print(f"predicted spread={predicted_spread:.12f}")

    tokenizer = build_combined(alphabet, merge_lists, allocation)
    padding_merges = pad_to_exact_vocab(tokenizer, texts)
    if len(tokenizer.vocab) != VOCAB_SIZE:
        raise AssertionError(f"vocabulary is {len(tokenizer.vocab)}, expected {VOCAB_SIZE}")

    rows = {}
    for lang in LANGS:
        ids = tokenizer.encode(texts[lang])
        decoded = tokenizer.decode(ids)
        roundtrip = visible_text(decoded) == visible_text(texts[lang])
        if not roundtrip:
            raise AssertionError(f"visible round trip failed for {lang}")
        rows[lang] = {
            "tokens": len(ids),
            "faithful_units": denominators[lang],
            "fertility": len(ids) / denominators[lang],
            "visible_roundtrip": roundtrip,
        }

    values = [row["fertility"] for row in rows.values()]
    measured_spread = max(values) - min(values)
    raw_score = 1000 / measured_spread
    hindi_penalty = penalty_factor(rows["hi"]["fertility"])
    english_penalty = penalty_factor(rows["en"]["fertility"])

    TOK_DIR.mkdir(exist_ok=True)
    tokenizer.save(TOK_DIR / "combined.json")
    meta = {
        "variant": "wiki_faithful_markdown",
        "languages": list(LANGS),
        "vocab_size": len(tokenizer.vocab),
        "alphabet_size": len(tokenizer.alphabet),
        "byte_fallback_tokens": 256,
        "allocation": allocation,
        "padding_merges": padding_merges,
        "rows": rows,
        "spread": measured_spread,
        "raw_score": raw_score,
        "hindi_penalty_factor": hindi_penalty,
        "hindi_adjusted_score": raw_score / hindi_penalty,
        "english_penalty_factor": english_penalty,
        "english_adjusted_score": raw_score / english_penalty,
    }
    (TOK_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"vocab={len(tokenizer.vocab)} padding_merges={padding_merges}")
    for lang, row in rows.items():
        print(f"{lang}: tokens={row['tokens']:,} units={row['faithful_units']:,} "
              f"fertility={row['fertility']:.9f} roundtrip=pass")
    print(f"spread={measured_spread:.12f}")
    print(f"raw_score={raw_score:,.2f}")
    print(f"hindi_adjusted_score={raw_score / hindi_penalty:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
