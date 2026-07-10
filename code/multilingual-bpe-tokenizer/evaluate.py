"""Evaluate the saved combined tokenizer on the four India Wikipedia pages -
the same pages it was built from (train = eval is deliberate in this
experiment; it measures the balance ceiling, not generalization). The
evaluation text is an HTML->Markdown rendition of the pages (the converter is
treated as unknown) and any <unk> zeroes the run, so this reports the
markdown convention first, others alongside, and verifies that <unk> cannot
occur.

Checks and reports:
  * per-language words, tokens, fertility X_i = tokens/words on the html2text
    markdown rendition (primary - the training and evaluation text)
  * the same on the markdownify rendition (converter-robustness check) and
    on the plaintext extract (no markup)
  * sorted fertilities, spread = X_max - X_min, and score = 1000 / spread
  * vocab size <= 10000
  * round-trip decode(encode(word)) == word on EVERY word of every rendition
    - byte fallback makes this lossless even for characters outside the
    alphabet, and there is no <unk> id at all
"""

from bpe import BPETokenizer
from data import load_md_texts, load_md_texts_alt, load_raw_texts
from optimize import LANGS, TOK_DIR, VOCAB_CAP


def report(tag, texts, tok):
    print(f"\n--- {tag}")
    print(f"{'lang':>5} {'words':>8} {'tokens':>8} {'fertility':>11} "
          f"{'roundtrip':>10} {'fallback-words':>15}")
    ferts = {}
    rt_ok = True
    for l in LANGS:
        words = texts[l].split()
        n_tok = 0
        n_fb = 0
        for w in words:
            ids = tok._encode_word(w)
            n_tok += len(ids)
            if any(tok.byte_base <= i < tok.byte_base + 256 for i in ids):
                n_fb += 1
            if tok.decode(ids) != w:
                rt_ok = False
        ferts[l] = n_tok / len(words)
        print(f"{l:>5} {len(words):>8,} {n_tok:>8,} {ferts[l]:>11.6f} "
              f"{('pass' if rt_ok else 'FAIL'):>10} "
              f"{n_fb:>6} ({n_fb / len(words):>5.1%})")
    xs = sorted(ferts.values())
    spread = xs[-1] - xs[0]
    score = 1000 / spread if spread > 0 else float("inf")
    order = sorted(ferts, key=ferts.get)
    print("  sorted: " + "  <  ".join(f"{l} {ferts[l]:.6f}" for l in order))
    print(f"  X_max - X_min = {spread:.3e}   SCORE = 1000/spread = {score:,.1f}")
    print(f"  X_i <= 1.2 : {sum(x <= 1.2 for x in xs)}/{len(xs)} languages")
    return rt_ok


def main():
    tok = BPETokenizer.load(TOK_DIR / "combined.json")
    print(f"vocab size: {len(tok.vocab)}  (cap {VOCAB_CAP})  "
          f"{'OK' if len(tok.vocab) <= VOCAB_CAP else 'OVER CAP'}")

    ok = report("HTML->Markdown, html2text (primary: training & evaluation format)",
                load_md_texts(), tok)
    ok &= report("HTML->Markdown, markdownify (converter-robustness check)",
                 load_md_texts_alt(), tok)
    ok &= report("plaintext extract, whitespace split", load_raw_texts(), tok)

    # <unk> cannot exist; prove byte fallback round-trips arbitrary input.
    stress = "Émile+Zola🙂中文— café’s ½×Ω"
    ok_stress = all(tok.decode(tok._encode_word(w)) == w for w in stress.split())
    print(f"\nlossless round-trip on all page words : {bool(ok)}")
    print(f"byte-fallback stress test (emoji/CJK/…): "
          f"{'pass' if ok_stress else 'FAIL'}")
    print("(<unk> does not exist in this tokenizer: characters outside the "
          "alphabet encode as UTF-8 byte tokens and decode losslessly)")


if __name__ == "__main__":
    main()
