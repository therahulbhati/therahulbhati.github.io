"""Build the 10k-vocab multilingual BPE tokenizer.

Evaluation setup for this experiment: each India page's HTML is converted to
MARKDOWN with links preserved (the converter is treated as unknown), the saved
tokenizer runs on that text, and X = tokens/words per language;
`1000/(X_max - X_min)` is the score and any <unk> zeroes the run. So:

  * the TRAINING corpus is the html2text markdown rendition of the pages -
    link syntax, URLs, and numbers appear exactly as words, and BPE learns
    `](/wiki/`, percent-encodings, years, etc. by frequency, the same way it
    learns letters;
  * the alphabet covers both markdown renditions (html2text + markdownify)
    AND the raw plaintext; anything else encodes via 256 byte-fallback tokens
    (SentencePiece-style), so <unk> is structurally impossible;
  * fertilities are optimized on the html2text markdown under whitespace word
    counting and reported under every plausible convention.

Architecture: SHARED ASCII POOL + PER-LANGUAGE NATIVE LADDERS.

With markdown, all four pages carry heavy ASCII content (URLs, English link
titles, syntax) with *different* pair statistics per page. Training separate
per-language merge lists and concatenating them breaks: one language's
higher-ranked ASCII merges eat another's URL patterns mid-word and strand its
longer merges (measured fertility ends up several units above the model). The
sound split follows the scripts:

  1. Every word is a sequence of maximal runs: ASCII runs and native-script
     runs. Merges never cross a run boundary (no merge string mixes the two),
     so token counts are exactly additive per run.
  2. ONE shared BPE list is trained on the pooled ASCII runs of all four
     pages - a single canonical construction for every ASCII string, zero
     interference. English text is pure ASCII, so this pool serves English
     entirely.
  3. One native ladder per Indic language is trained on its native runs.
     Devanagari/Telugu/Kannada are disjoint Unicode blocks: provably no
     conflicts between ladders.
  4. Exact model: tokens_l(k_s, k_l) = ascii_tokens_l(k_s) +
     native_tokens_l(k_l), where the per-language ascii curve comes from
     replaying the shared merge list against that language's own ASCII-run
     frequencies. Greedy allocation ("fund the language with the worst
     modeled fertility"; a shared-pool step helps everyone) spends the budget
     10,000 - alphabet - 256, then a short hill-climb over the four knobs
     (k_s, k_hi, k_te, k_kn) polishes against the real tokenizer.
"""

import json
import re
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from bpe import BPETokenizer, get_word_freqs
from data import (load_md_texts, load_md_texts_alt, load_raw_texts,
                  load_texts)

LANGS = ["en", "hi", "te", "kn"]
NATIVE_LANGS = ["hi", "te", "kn"]      # en is pure-ASCII: served by the pool
KNOBS = ["shared"] + NATIVE_LANGS
VOCAB_CAP = 10_000
TOK_DIR = Path(__file__).parent / "tokenizer"


def build_alphabet(*text_dicts: dict[str, str]) -> list[str]:
    chars: set[str] = set()
    for d in text_dicts:
        for t in d.values():
            chars.update(t.replace(" ", ""))
    return sorted(chars)


def word_runs(word: str):
    """Maximal runs of ASCII vs non-ASCII characters."""
    for is_ascii, grp in groupby(word, key=lambda c: ord(c) < 128):
        yield is_ascii, "".join(grp)


def run_freqs(freqs: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Split a word-frequency dict into (ascii_run_freqs, native_run_freqs)."""
    ascii_f: dict[str, int] = {}
    native_f: dict[str, int] = {}
    for w, c in freqs.items():
        for is_ascii, run in word_runs(w):
            tgt = ascii_f if is_ascii else native_f
            tgt[run] = tgt.get(run, 0) + c
    return ascii_f, native_f


def train_merge_list(freqs: dict[str, int], alphabet: list[str],
                     cap: int) -> tuple[list[tuple[str, str]], list[int], int]:
    """BPE merge list over `freqs` as (left, right) STRING pairs, plus each
    merge's token reduction on this corpus and the initial token count."""
    tok = BPETokenizer()
    tok.set_alphabet(alphabet)
    init_tokens = sum(len(w) * c for w, c in freqs.items())
    _, reductions = tok.train_from_freqs(freqs, n_merges=cap, min_freq=1,
                                         record_curve=True)
    str_merges = [(tok.vocab[a], tok.vocab[b]) for a, b in tok.merges]
    return str_merges, reductions, init_tokens


def replay_reductions(merges: list[tuple[str, str]],
                      freqs: dict[str, int]) -> list[int]:
    """Token reduction of each merge, in order, on a DIFFERENT corpus than it
    was trained on - gives each language's exact fertility curve under the
    shared ASCII merge list."""
    words = [list(w) for w in freqs]
    counts = list(freqs.values())
    pair_words: dict = defaultdict(set)
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_words[p].add(wi)
    reds = []
    for lstr, rstr in merges:
        red = 0
        merged_str = lstr + rstr
        for wi in list(pair_words.get((lstr, rstr), ())):
            w = words[wi]
            c = counts[wi]
            for p in zip(w, w[1:]):
                pair_words[p].discard(wi)
            out = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == lstr and w[i + 1] == rstr:
                    out.append(merged_str)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            red += (len(w) - len(out)) * c
            words[wi] = out
            for p in zip(out, out[1:]):
                pair_words[p].add(wi)
        reds.append(red)
    return reds


def build_combined(alphabet: list[str],
                   merge_seq: list[tuple[str, str]]) -> BPETokenizer:
    """One tokenizer from a sequence of string-pair merges (shared prefix
    first, then the native prefixes; their string sets are disjoint by
    construction, so first-wins dedup is just belt-and-braces)."""
    tok = BPETokenizer()
    tok.set_alphabet(alphabet)
    str_to_id = {c: i for i, c in enumerate(tok.alphabet)}
    for lstr, rstr in merge_seq:
        merged = lstr + rstr
        if merged in str_to_id:
            continue
        str_to_id[merged] = tok.add_merge(str_to_id[lstr], str_to_id[rstr])
    return tok


def measured_fertilities(tok: BPETokenizer, freqs, n_words) -> dict[str, float]:
    return {l: sum(c * len(tok._encode_word(w)) for w, c in freqs[l].items())
               / n_words[l] for l in LANGS}


def spread(f: dict[str, float]) -> float:
    xs = sorted(f.values())
    return xs[-1] - xs[0]


class Model:
    """Exact fertility model: cumulative-reduction curves per language for
    the shared pool and for its own native ladder."""

    def __init__(self, shared_red_by_lang, ascii_init, native_red, native_init,
                 n_words):
        def cum(xs):
            out = [0]
            for x in xs:
                out.append(out[-1] + x)
            return out
        self.sc = {l: cum(shared_red_by_lang[l]) for l in LANGS}
        self.nc = {l: cum(native_red[l]) for l in NATIVE_LANGS}
        self.ai = ascii_init
        self.ni = native_init
        self.nw = n_words

    def fert(self, l: str, k: dict[str, int]) -> float:
        ks = min(k["shared"], len(self.sc[l]) - 1)
        toks = self.ai[l] - self.sc[l][ks]
        if l in self.nc:
            kl = min(k[l], len(self.nc[l]) - 1)
            toks += self.ni[l] - self.nc[l][kl]
        return toks / self.nw[l]

    def ferts(self, k):
        return {l: self.fert(l, k) for l in LANGS}


def greedy_allocate(model: Model, limits: dict[str, int],
                    budget: int) -> dict[str, int]:
    """Fund the knob that lowers the currently-worst language: the worst
    language's own native ladder if it has one, else (worst = en) the shared
    pool. Exact model, no duplicates - every step costs one vocab slot."""
    k = {kn: 0 for kn in KNOBS}
    for _ in range(budget):
        f = model.ferts(k)
        order = sorted(LANGS, key=lambda l: -f[l])
        stepped = False
        for worst in order:
            knob = worst if worst in NATIVE_LANGS else "shared"
            if k[knob] < limits[knob]:
                k[knob] += 1
                stepped = True
                break
            if worst in NATIVE_LANGS and k["shared"] < limits["shared"]:
                k["shared"] += 1   # native ladder exhausted: shared still helps
                stepped = True
                break
        if not stepped:
            break
    return k


def model_hill_climb(model: Model, k, limits, budget):
    """Local search over the four knobs on the EXACT model (free to evaluate,
    unlike rebuilding the real tokenizer): shift d slots from one knob to
    another, at several scales. Budget-preserving swaps ONLY - the experiment
    requires the full 10,000-token vocab, and dropping merges to equalize
    fertilities upward is the degenerate direction of the spread metric.
    First-improvement, repeated until a full sweep finds nothing."""
    def sp(kk):
        return spread(model.ferts(kk))
    best = sp(k)
    improved = True
    while improved:
        improved = False
        for i in KNOBS:
            for d in (512, 128, 32, 8, 2, 1):
                if k[i] < d:
                    continue
                for j in KNOBS:
                    if j == i or k[j] + d > limits[j]:
                        continue
                    kk = dict(k)
                    kk[i] -= d
                    kk[j] += d
                    s = sp(kk)
                    if s < best - 1e-12:
                        best, k, improved = s, kk, True
    print(f"model hill-climb: spread {best:.6f}  "
          + " ".join(f"{kn}={k[kn]}" for kn in KNOBS))
    return k


def hill_climb(k, limits, rebuild, max_steps: int = 4):
    """Short polish against the REAL combined tokenizer (each candidate is a
    full rebuild + re-encode, so this only cleans up the model's ~1e-2
    residual, it doesn't explore)."""
    best_sp = rebuild(k)
    print(f"measured spread at model optimum: {best_sp:.6f}")
    for step in range(1, max_steps + 1):
        moves = []
        for i in KNOBS:
            for d in (1, 8, 64):
                if k[i] < d:
                    continue
                for j in KNOBS:   # budget-preserving swaps only
                    if j != i and k[j] + d <= limits[j]:
                        kk = dict(k)
                        kk[i] -= d
                        kk[j] += d
                        moves.append(kk)
        best_move = None
        for kk in moves:
            sp = rebuild(kk)
            if sp is not None and sp < best_sp:
                best_sp, best_move = sp, kk
        if best_move is None:
            break
        k = best_move
        print(f"  polish {step}: spread {best_sp:.6f}  "
              + " ".join(f"{kn}={k[kn]}" for kn in KNOBS))
    return k, best_sp


def convention_report(tok, tag, texts, wc=None):
    """Fertility per language under a given text rendition; `wc` overrides
    the word count (e.g. \\w+ matches instead of whitespace chunks)."""
    f = {}
    for l in LANGS:
        n = wc(texts[l]) if wc else len(texts[l].split())
        f[l] = tok.count_tokens(texts[l]) / n
    xs = sorted(f.values())
    sp = xs[-1] - xs[0]
    print(f"\n{tag}:")
    for l in LANGS:
        print(f"  {l}: {f[l]:.6f}")
    print(f"  spread = {sp:.3e}   score = {1000 / sp:,.1f}   "
          f"<=1.2: {sum(x <= 1.2 for x in xs)}/4")
    return f, sp


def main():
    # Training and primary eval: the html2text markdown rendition -
    # the closest rendition to the evaluation text.
    texts = load_md_texts()
    alt_texts = load_md_texts_alt()
    raw_texts = load_raw_texts()
    freqs = {l: get_word_freqs(t) for l, t in texts.items()}
    n_words = {l: len(texts[l].split()) for l in LANGS}
    alphabet = build_alphabet(texts, alt_texts, raw_texts)
    A = len(alphabet)
    budget = VOCAB_CAP - A - 256   # 256 byte-fallback ids reserved after A
    print(f"alphabet: {A} codepoints (md + alt-md + raw plaintext), "
          f"+256 byte-fallback tokens, merge budget = {budget}")

    # Split every page into ASCII runs (shared pool) and native runs.
    ascii_f, native_f = {}, {}
    for l in LANGS:
        ascii_f[l], native_f[l] = run_freqs(freqs[l])
    pooled_ascii: dict[str, int] = {}
    for l in LANGS:
        for w, c in ascii_f[l].items():
            pooled_ascii[w] = pooled_ascii.get(w, 0) + c

    shared_list, _, _ = train_merge_list(pooled_ascii, alphabet, cap=budget)
    print(f"shared ASCII pool: {len(pooled_ascii):,} unique runs, "
          f"merge list {len(shared_list):,}")
    shared_red_by_lang = {l: replay_reductions(shared_list, ascii_f[l])
                          for l in LANGS}
    ascii_init = {l: sum(len(w) * c for w, c in ascii_f[l].items())
                  for l in LANGS}

    native_list, native_red, native_init = {}, {}, {}
    for l in NATIVE_LANGS:
        native_list[l], native_red[l], native_init[l] = \
            train_merge_list(native_f[l], alphabet, cap=budget)
        print(f"  {l} native ladder: {len(native_f[l]):,} unique runs, "
              f"merge list {len(native_list[l]):,}")

    model = Model(shared_red_by_lang, ascii_init, native_red, native_init,
                  n_words)
    limits = {"shared": len(shared_list),
              **{l: len(native_list[l]) for l in NATIVE_LANGS}}
    k = greedy_allocate(model, limits, budget)
    mf = model.ferts(k)
    print("greedy (exact model):", " ".join(f"{kn}={k[kn]}" for kn in KNOBS))
    print("model fertilities:",
          " ".join(f"{l}={mf[l]:.4f}" for l in LANGS),
          f"spread {spread(mf):.6f}")
    k = model_hill_climb(model, k, limits, budget)

    def merge_seq(kk):
        seq = list(shared_list[:kk["shared"]])
        for l in NATIVE_LANGS:
            seq.extend(native_list[l][:kk[l]])
        return seq

    def rebuild(kk):
        tok = build_combined(alphabet, merge_seq(kk))
        if len(tok.vocab) > VOCAB_CAP:
            return None
        return spread(measured_fertilities(tok, freqs, n_words))

    k, _ = hill_climb(k, limits, rebuild)

    tok = build_combined(alphabet, merge_seq(k))
    ferts = measured_fertilities(tok, freqs, n_words)
    print(f"\nvocab: {len(tok.vocab)} (cap {VOCAB_CAP})")
    print(f"{'lang':>5} {'words':>8} {'tokens':>8} {'fertility':>11}")
    for l in LANGS:
        n_tok = tok.count_tokens(texts[l])
        print(f"{l:>5} {n_words[l]:>8,} {n_tok:>8,} {ferts[l]:>11.6f}")

    xs = sorted(ferts.values())
    sp = xs[-1] - xs[0]
    score = 1000 / sp if sp > 0 else float("inf")
    print(f"\nspread = {sp:.3e}   SCORE = {score:,.1f}")
    print(f"X_i <= 1.2 (efficiency bar): {sum(x <= 1.2 for x in xs)}/{len(xs)} languages")

    # Same tokenizer under every plausible evaluation convention - the
    # converter and word-counting rule aren't fixed, so nothing is claimed
    # beyond what each convention reproduces.
    wcount = lambda t: len(re.findall(r"\w+", t))
    wplus_f, wplus_sp = convention_report(
        tok, "markdown, \\w+ word count", texts, wc=wcount)
    alt_f, alt_sp = convention_report(
        tok, "alternate converter (markdownify), whitespace", alt_texts)
    plain_f, plain_sp = convention_report(
        tok, "plaintext extract, raw whitespace", raw_texts)

    TOK_DIR.mkdir(exist_ok=True)
    tok.save(TOK_DIR / "combined.json")
    (TOK_DIR / "meta.json").write_text(json.dumps({
        "langs": LANGS,
        "words": n_words,
        "fertilities": ferts, "spread": sp, "score": score,
        "conventions": {
            "md_wsplit": {"fertilities": ferts, "spread": sp},
            "md_wordregex": {"fertilities": wplus_f, "spread": wplus_sp},
            "md_alt_converter": {"fertilities": alt_f, "spread": alt_sp},
            "plaintext_raw": {"fertilities": plain_f, "spread": plain_sp},
        },
        "vocab_size": len(tok.vocab),
        "knobs": k,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {TOK_DIR / 'combined.json'}")


if __name__ == "__main__":
    main()
