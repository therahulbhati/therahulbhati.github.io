"""Bundle the tokenizer and the four cleaned India pages into widget/data.js.

The widget (widget/index.html) re-implements cleaning + BPE encoding in
JavaScript and computes every displayed number - tokens, fertilities, spread,
score - live in the browser from this bundle. Nothing on the page is
hardcoded, so what the widget shows is what anyone re-running the tokenizer
gets. `meta` is included only so the page can flag a mismatch if the live
computation ever disagrees with the build.
"""

import json
from pathlib import Path

from data import EVAL_TITLES, load_md_texts, load_raw_texts
from optimize import LANGS, TOK_DIR, VOCAB_CAP

ROOT = Path(__file__).parent
OUT = ROOT / "widget" / "data.js"


def main():
    texts = load_md_texts()        # primary: HTML->Markdown rendition
    raw_texts = load_raw_texts()   # secondary: plaintext extract
    tok = json.loads((TOK_DIR / "combined.json").read_text(encoding="utf-8"))
    meta = json.loads((TOK_DIR / "meta.json").read_text(encoding="utf-8"))
    bundle = {
        "langs": LANGS,
        "titles": EVAL_TITLES,
        "vocab_cap": VOCAB_CAP,
        "alphabet": tok["alphabet"],
        "merges": tok["merges"],
        "texts": {l: texts[l] for l in LANGS},
        "raw_texts": {l: raw_texts[l] for l in LANGS},
        "meta": {"fertilities": meta["fertilities"], "spread": meta["spread"],
                 "score": meta["score"], "vocab_size": meta["vocab_size"]},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("const DATA = " + json.dumps(bundle, ensure_ascii=False)
                   + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
