# Multilingual BPE Tokenizer

A from-scratch byte-pair-encoding tokenizer for the Wikipedia **India** page in
four languages — English (en), Hindi (hi), Telugu (te), Kannada (kn) — under a
single shared 10,000-token vocabulary.

The experiment scores `1000 / (X_max − X_min)` over the four per-language
fertilities `X = tokens/words`. The rules: evaluation runs on the India pages
only, converted from **HTML to Markdown with links preserved** (the converter
is treated as unknown; the criterion is 100% recoverability), the tokenizer
may be built from anything, no text may be thrown away to improve the score,
the vocab must be 10,000 overall, and **any `<unk>` zeroes the run**.

**Result on the html2text markdown rendition (primary): fertilities
2.94 / 2.97 / 3.18 / 3.08, spread 2.368 × 10⁻¹, score ≈ 4,223 — with zero
byte-fallback words and a lossless round-trip on every rendition tested.**

Live, self-verifying widget (all numbers computed in-browser from the shipped
tokenizer): https://therahulbhati.github.io/tokenizer/

## Result (`evaluate.py`, from the saved artifact)

```
vocab size: 9996  (cap 10000)  OK

--- HTML->Markdown, html2text (primary: training & evaluation format)
 lang    words   tokens   fertility  roundtrip  fallback-words
   en   36,287  106,740    2.941549       pass      0 ( 0.0%)
   hi   16,893   50,194    2.971290       pass      0 ( 0.0%)
   te    6,617   21,031    3.178329       pass      0 ( 0.0%)
   kn    2,943    9,061    3.078831       pass      0 ( 0.0%)
  X_max - X_min = 2.368e-01   SCORE = 4,223.3

--- HTML->Markdown, markdownify (converter-robustness check)
   en 3.800   hi 3.329   te 3.092   kn 2.950     spread 0.851

--- plaintext extract, whitespace split
   en 1.995   hi 1.868   te 1.317   kn 1.352     spread 0.678
```

Fertility swings by more than a full unit between renditions of the *same
pages* — markdown link syntax and percent-encoded URLs dominate the token
count. That is why every convention is reported: an evaluator's convention
is their choice, and no claim here depends on guessing it.

## No `<unk>`, by construction

- The **alphabet covers every character in three renditions** of the pages
  (html2text markdown, markdownify markdown, plaintext extract — 359
  codepoints, case preserved).
- **256 byte-fallback tokens** (SentencePiece-style) replace `<unk>` entirely:
  any character outside the alphabet encodes as its UTF-8 bytes and decodes
  losslessly. `evaluate.py` stress-tests emoji/CJK/IPA. There is no `<unk>`
  id in the vocabulary at all — the zero-rule cannot trigger.

## Architecture: shared ASCII pool + native ladders

With markdown, all four pages carry heavy ASCII content (URLs, `](/wiki/`
syntax, percent-encoded paths where one Devanagari character becomes nine
ASCII characters) with *different* pair statistics per page. The naive
design — one merge list per language, concatenated — breaks here: languages
independently learn different decompositions of the same ASCII strings and
fight over them at encode time; measured fertility lands several units above
prediction. The sound split follows the scripts:

1. **Runs.** Every word is a sequence of maximal ASCII runs and native-script
   runs. No merge string mixes the two, so merges never cross a run boundary
   and token counts are exactly additive per run.
2. **One shared ASCII pool.** All four pages' ASCII runs train a single merge
   list — one canonical construction per string, zero interference. English
   is pure ASCII, so the pool serves it entirely.
3. **One native ladder per Indic language,** trained on its native runs.
   Devanagari/Telugu/Kannada are disjoint Unicode blocks: provably no
   conflicts.
4. **Exact model.** Because train text = eval text, each merge's pair
   frequency is exactly the tokens it removes; replaying the shared list
   against each language's own ASCII runs gives per-language fertility
   curves that are exact, not estimated. Allocation over the four knobs
   (shared depth + three native depths) is greedy "fund the worst", then a
   swap-only hill-climb on the model (budget-preserving: the full 10,000
   vocab is required, and equalizing by discarding merges is the degenerate
   direction of the spread metric), then a short measured polish.

The final artifact is a single standard BPE tokenizer — alphabet (359) + byte
tokens (256) + 9,381 ordered merges, vocab 9,996 — in `tokenizer/combined.json`.
Encoding semantics are 100% Sennrich BPE; the construction is per-pool BPE
training plus explicit vocabulary allocation.

## Design history (kept because the failures are the lessons)

- **Plaintext era:** training on the API plaintext extracts with
  punctuation/digit-splitting cleaning reached all-four-under-1.2 (X = 1.17,
  spread 4.8 × 10⁻⁵). Switching the evaluation text to markdown invalidated
  the format; the equalization machinery carried over.
- **Lowercasing** was reverted before it could hurt: a lowercased alphabet
  turns every capital letter of anyone else's rendition into fallback.
- **Bengali → Kannada:** the fourth language is free choice, and the binding
  constraint is how many merges its page consumes (Bengali ~3,400 vs Kannada
  ~1,100 at the time). Not script kinship — Kannada and Telugu share zero
  merges — just page economics.
- **Held-out generalization footnote:** an earlier honest split (trained on
  ~50 different India-topic articles, balanced by nested cross-validation,
  eval page never touched) showed ~28% of te/bn eval words never occur in
  training and must split — the Ahia et al. 2023 "fertility tax". Training
  on the measured page is overfitting; here it is deliberate, measuring the
  balance ceiling rather than generalization. Both facts are disclosed.

## Files

| file | purpose |
|---|---|
| `data.py` | Fetches the pages (API plaintext + parsed HTML), strips site chrome, renders markdown via html2text (primary) and markdownify (robustness); NFC + whitespace canonicalization. |
| `bpe.py` | From-scratch codepoint BPE with byte fallback: incremental `train_from_freqs` (lazy-heap Sennrich merges, per-merge token-reduction recording), `encode`/`decode`, `save`/`load`. |
| `optimize.py` | Splits words into ASCII/native runs, trains the shared pool + native ladders, replays exact fertility curves, allocates the budget (greedy + swap-only model hill-climb + measured polish), saves `tokenizer/`. |
| `evaluate.py` | Reloads the saved tokenizer; reports all renditions, vocab cap, lossless round-trip on every word, and the byte-fallback stress test. |
| `build_widget.py`, `widget/` | Static, self-verifying results page (computes everything live in the browser from the shipped tokenizer); hosted at therahulbhati.github.io/tokenizer/. |

## Run it

```bash
pip install html2text markdownify beautifulsoup4
python3 data.py       # fetch/cache pages + markdown renditions (first run)
python3 optimize.py   # build + allocate the tokenizer, save tokenizer/
python3 evaluate.py   # reproduce the reported numbers from the saved file
python3 build_widget.py   # regenerate widget/data.js
```

## References

- Sennrich, Haddow & Birch (2016) — BPE for NMT; the core merge algorithm.
- Kudo & Richardson (2018), *SentencePiece* — the byte-fallback mechanism.
- Ahia, Mielke et al. (2023), *Do All Languages Cost the Same? Tokenization in
  the Era of Commercial Language Models* — the per-language "fertility tax"
  measured in the held-out footnote.
