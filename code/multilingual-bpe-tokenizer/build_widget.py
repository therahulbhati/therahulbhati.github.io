"""Bundle the exact tokenizer, corpus snapshots, and metrics for the widget."""

from __future__ import annotations

import json
import shutil
import unicodedata
from pathlib import Path

from faithful import LANGS, LANGUAGE_NAMES, VOCAB_SIZE, WIKI_TITLES


ROOT = Path(__file__).resolve().parent
WIDGET = ROOT / "widget"
CORPUS = ROOT / "corpus"
TOKENIZER = ROOT / "tokenizer" / "combined.json"
META = ROOT / "tokenizer" / "meta.json"


def main() -> int:
    tokenizer = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8"))
    texts = {
        lang: (CORPUS / f"{lang}.faithful.txt").read_text(encoding="utf-8")
        for lang in LANGS
    }
    observed_characters = {char for text in texts.values() for char in text}
    character_policy = {
        str(ord(char)): [
            unicodedata.category(char)[0] in "LMN",
            char.isspace(),
        ]
        for char in observed_characters
    }
    bundle = {
        "langs": list(LANGS),
        "language_names": LANGUAGE_NAMES,
        "titles": WIKI_TITLES,
        "vocab_size": VOCAB_SIZE,
        "alphabet": tokenizer["alphabet"],
        "merges": tokenizer["merges"],
        "texts": texts,
        # Freeze Python's assignment-relevant Unicode classification for every
        # observed code point. Browser/Node Unicode tables differ on a handful
        # of characters, which otherwise changes both token and unit counts.
        "character_policy": character_policy,
        "meta": meta,
    }

    WIDGET.mkdir(exist_ok=True)
    (WIDGET / "data.js").write_text(
        "const DATA = " + json.dumps(bundle, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    shutil.copyfile(TOKENIZER, WIDGET / "tokenizer.json")
    corpus_out = WIDGET / "corpus"
    corpus_out.mkdir(exist_ok=True)
    for lang in LANGS:
        shutil.copyfile(
            CORPUS / f"{lang}.faithful.txt",
            corpus_out / f"{lang}.faithful.txt",
        )
    print(f"wrote widget bundle with {len(tokenizer['alphabet']) + 256 + len(tokenizer['merges']):,} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
