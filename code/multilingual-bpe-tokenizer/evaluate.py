#!/usr/bin/env python3
"""Run the assignment's faithful Markdown evaluator on the saved tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

from bpe import BPETokenizer
from faithful import (
    LANGS,
    VOCAB_SIZE,
    faithful_units,
    penalty_factor,
    visible_text,
)


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TOKENIZER_PATH = ROOT / "tokenizer" / "combined.json"
GRADER_SAMPLE = "India's population is 1,428,627,663."


def main() -> int:
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    rows = {}
    failures = []

    print(f"vocab size: {len(tokenizer.vocab):,} / {VOCAB_SIZE:,}")
    print(f"{'lang':>5} {'tokens':>10} {'units':>10} {'fertility':>13} {'roundtrip':>11}")
    for lang in LANGS:
        text = (CORPUS / f"{lang}.faithful.txt").read_text(encoding="utf-8")
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        roundtrip = visible_text(decoded) == visible_text(text)
        if not roundtrip:
            failures.append(lang)
        units = faithful_units(text)
        rows[lang] = {
            "tokens": len(ids),
            "faithful_units": units,
            "fertility": len(ids) / units,
            "visible_roundtrip": roundtrip,
        }
        print(f"{lang:>5} {len(ids):>10,} {units:>10,} "
              f"{rows[lang]['fertility']:>13.9f} "
              f"{('pass' if roundtrip else 'FAIL'):>11}")

    fertilities = [row["fertility"] for row in rows.values()]
    spread = max(fertilities) - min(fertilities)
    raw_score = 1000 / spread
    hindi_penalty = penalty_factor(rows["hi"]["fertility"])
    english_penalty = penalty_factor(rows["en"]["fertility"])

    sample_ids = tokenizer.encode(GRADER_SAMPLE)
    sample_decoded = tokenizer.decode(sample_ids)
    sample_pass = visible_text(sample_decoded) == visible_text(GRADER_SAMPLE)
    if not sample_pass:
        failures.append("grader_sample")

    print(f"\nspread: {spread:.12f}")
    print(f"raw score: {raw_score:,.2f}")
    print(f"Hindi penalty factor: {hindi_penalty:.9f}")
    print(f"Hindi-adjusted score: {raw_score / hindi_penalty:,.2f}")
    print(f"English penalty factor (feedback variant): {english_penalty:.9f}")
    print(f"English-adjusted score: {raw_score / english_penalty:,.2f}")
    print("\ngrader sample:")
    print(f"  input   : {GRADER_SAMPLE}")
    print(f"  decoded : {sample_decoded}")
    print(f"  visible round trip: {'pass' if sample_pass else 'FAIL'}")

    result = {
        "vocab_size": len(tokenizer.vocab),
        "rows": rows,
        "spread": spread,
        "raw_score": raw_score,
        "hindi_penalty_factor": hindi_penalty,
        "hindi_adjusted_score": raw_score / hindi_penalty,
        "english_penalty_factor": english_penalty,
        "english_adjusted_score": raw_score / english_penalty,
        "grader_sample": {
            "input": GRADER_SAMPLE,
            "decoded": sample_decoded,
            "visible_roundtrip": sample_pass,
        },
    }
    print("\nJSON result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if len(tokenizer.vocab) != VOCAB_SIZE:
        failures.append("vocab_size")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
