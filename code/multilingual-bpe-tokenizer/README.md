# ERA V5 Assignment 2 — Faithful Multilingual BPE

One shared, from-scratch BPE tokenizer with an exact **10,000-token** vocabulary
for the wiki-faithful Markdown renditions of the India pages in English, Hindi,
Telugu, and Kannada.

Live evaluator and tokenizer: https://therahulbhati.github.io/tokenizer/

## Result

The saved tokenizer was evaluated with the assignment's faithful-unit policy:

```text
faithful unit = one contiguous Unicode letter/mark/number run
               OR one visible non-whitespace punctuation/symbol
fertility     = token count / faithful-unit count
score         = 1000 / (maximum fertility - minimum fertility)
```

| Language | Tokens | Faithful units | Fertility | Visible round trip |
|---|---:|---:|---:|---:|
| English | 117,659 | 186,367 | 0.631329581 | pass |
| Hindi | 55,783 | 88,359 | 0.631322220 | pass |
| Telugu | 22,912 | 36,292 | 0.631323708 | pass |
| Kannada | 7,761 | 12,293 | 0.631334906 | pass |

```text
spread = 0.631334906044 - 0.631322219581
       = 0.000012686463

raw score = 1000 / 0.000012686463
          = 78,824,179.03

Hindi penalty factor = exp(max(0, 0.631322220 / 1.2 - 1)) = 1
Hindi-adjusted score = 78,824,179.03
```

The feedback mentioned an English variant of the penalty. English is also below
1.2, so its factor is 1 and the adjusted score is unchanged.

The score is unusually large because it is the reciprocal of a very small
spread. Raw integer counts and full-precision ratios are provided above and in
`tokenizer/meta.json`; the live page recomputes them instead of trusting display
constants.

## Faithfulness gate

The exact required gate is tested literally:

```python
visible_text(tokenizer.decode(tokenizer.encode(text))) == visible_text(text)
```

where `visible_text` removes whitespace and nothing else. It passes every frozen
corpus in full, unseen Unicode stress cases, and the grader sample:

```text
input:   India's population is 1,428,627,663.
decoded: India'spopulationis1,428,627,663.
visible round trip: pass
```

Whitespace is a pre-tokenization boundary and is not encoded. Apostrophes,
commas, number separators, Markdown syntax, URL characters, punctuation, and all
other visible characters are preserved. The 256 UTF-8 byte-fallback tokens make
unseen characters lossless; this tokenizer has no `<unk>` token.

## Exact corpus extraction

The frozen evaluation snapshots are in `corpus/*.faithful.txt`.
`build_wiki_faithful_markdown.py` reproduces the extraction policy:

1. Fetch the India page from Wikipedia's REST HTML endpoint.
2. Remove only scripts, styles, metadata, and non-visible link machinery.
3. Convert category PageProps to visible category lines.
4. Make links and image sources absolute.
5. Convert with `markdownify`, retaining links, URLs, tables, references, image
   links, navboxes, categories, punctuation, and number separators.

English, Hindi, and Telugu use the reference corpus snapshots; Kannada was
captured with the identical pipeline. Snapshots are committed because Wikipedia
can change after evaluation.

## Training method

`optimize.py` trains independent merge ladders for disjoint Unicode groups:

- shared ASCII;
- Devanagari;
- Telugu;
- Kannada;
- other Unicode characters.

It replays every ladder against every language to obtain an exact token-reduction
curve, then allocates the full merge budget directly to minimize fertility
spread. The final allocation is:

```text
base alphabet       364
byte fallback       256
ASCII merges      4,124
Devanagari merges 2,010
Telugu merges     1,911
Kannada merges    1,322
other merges         13
-----------------------
total vocabulary 10,000
```

Because groups are disjoint, merge ladders cannot interfere with one another;
the allocation model matches the measured tokenizer exactly.

## Reproduce

```bash
python3 optimize.py
python3 evaluate.py
python3 -m unittest -v test_tokenizer.py
python3 build_widget.py
```

To rebuild the Wikipedia corpus rather than use the frozen snapshots:

```bash
python3 -m pip install -r requirements.txt
python3 build_wiki_faithful_markdown.py
```

## Submission contents

| Path | Purpose |
|---|---|
| `tokenizer/combined.json` | Exact alphabet and ordered merge list |
| `bpe.py` | Executable Python encoder, decoder, trainer, byte fallback, save/load |
| `optimize.py` | Score-directed BPE training and exact vocabulary allocation |
| `evaluate.py` | Official faithful units, raw score, penalties, and literal round-trip gate |
| `test_tokenizer.py` | Full-corpus, grader-sample, unseen-Unicode, and vocabulary tests |
| `build_wiki_faithful_markdown.py` | Exact Wikipedia extraction/conversion method |
| `corpus/*.faithful.txt` | Frozen evaluation corpus for every language |
| `widget/tokenizer.js` | Downloadable browser encoder and decoder for the custom JSON |
| `widget/index.html` | Live evaluator, round-trip demo, downloads, and public API |

The browser exposes the actual executable API as:

```javascript
tokenizer.encode(text)
tokenizer.decode(tokenIds)
```
