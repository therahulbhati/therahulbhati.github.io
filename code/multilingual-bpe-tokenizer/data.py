"""Fetch and cache Wikipedia articles for English, Hindi, Telugu, Kannada.

The evaluation setup for this experiment: each India page's HTML is converted
to MARKDOWN with links preserved (the converter is treated as unknown; the
criterion is 100% recoverability), the tokenizer runs on that markdown, and
X = tokens/words. Any <unk> zeroes the run, and no text may be thrown away to
improve the score.

Corpora per language:
  * load_md_texts()     - the India page's HTML rendered to Markdown via
    html2text (links preserved). THE training and primary eval corpus.
  * load_md_texts_alt() - the same HTML via markdownify: a second, different
    rendition used only to measure robustness to the converter choice - never
    for tuning.
  * load_texts() / load_raw_texts() - the API plaintext extracts (cleaned /
    untouched), kept for secondary convention reporting.
  * TRAIN_TITLES - seeds for the held-out generalization experiment in the
    README footnote (predates the Bengali -> Kannada swap).

All text is NFC-normalized and whitespace-canonicalized to single spaces so
`text.split()` word counting and round-trips are well defined. Words are NOT
punctuation- or digit-split in the markdown pipeline: the evaluation text
arrives with numbers and markdown syntax attached, and BPE learns them by
frequency like any other characters. Case is preserved throughout. The old
punctuation-splitting cleanup survives as clean_split(), used only for the
secondary plaintext-convention report.
"""

import json
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

EVAL_TITLES = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "kn": "ಭಾರತ",
}

# Verified to resolve (directly or via redirect) on each language's Wikipedia.
TRAIN_TITLES = {
    "en": [
        "History of India", "Geography of India", "Culture of India",
        "Economy of India", "Indian independence movement",
        "States and union territories of India", "Indian cuisine",
        "Demographics of India",
    ],
    "hi": [
        "भारत का इतिहास", "भारत का भूगोल", "भारत की संस्कृति",
        "भारतीय अर्थव्यवस्था", "भारतीय स्वतन्त्रता आन्दोलन",
        "भारत के राज्य तथा केन्द्र-शासित प्रदेश", "भारतीय खाना",
        "भारत की जनसांख्यिकी",
    ],
    "te": [
        "భారతదేశ చరిత్ర", "భారతదేశ రూపురేఖలు", "భారతీయ సంస్కృతి",
        "భారత ఆర్ధిక వ్యవస్థ", "భారత స్వాతంత్ర్యోద్యమం",
        "భారతదేశ రాష్ట్రాలు, కేంద్రపాలిత ప్రాంతాలు", "భారతీయ వంటకాలు",
    ],
    # (the held-out experiment in the README footnote also used Bengali seeds;
    # they were removed when the 4th language switched to Kannada)
}

DATA_DIR = Path(__file__).parent / "data"
USER_AGENT = "multilingual-bpe-tokenizer/1.0 (rjbhati009@gmail.com)"

# How many distinct India-topic articles to train each language on. The seed
# TRAIN_TITLES above are the starting point; the rest are discovered honestly by
# following article links out of those seeds (all India-topic by construction).
# The eval India page is always excluded, so train/eval stay disjoint.
TARGET_TRAIN_ARTICLES = 50
MIN_EXTRACT_CHARS = 800   # skip stubs / disambiguation pages


def _api(lang: str, params: dict) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            # Wikipedia rate-limits aggressively; back off and retry.
            if e.code != 429:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Wikipedia API kept rate-limiting for {lang}")


def canonicalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def clean_split(text: str) -> str:
    """Punctuation-splitting cleanup: split punctuation, symbols, and digits
    off as standalone whitespace-delimited words. Used ONLY for the secondary
    plaintext-convention report - the evaluation markdown arrives with
    punctuation, digits, and link syntax attached, so the tokenizer must not
    depend on this splitting."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch != " " and (cat[0] in ("P", "S") or cat == "Nd"):
            out.append(f" {ch} ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def fetch_html(lang: str, title: str) -> str:
    """Rendered page HTML via the parse API (what a browser shows, minus the
    site chrome) - the input the markdown renditions are made from."""
    data = _api(lang, {
        "action": "parse",
        "page": title,
        "redirects": 1,
        "prop": "text",
        "format": "json",
    })
    html = data["parse"]["text"]["*"]
    if not html:
        raise RuntimeError(f"Empty HTML for {lang}:{title}")
    return html


def _html_paths(lang: str) -> Path:
    return DATA_DIR / f"{lang}.html"


def _load_html() -> dict[str, str]:
    DATA_DIR.mkdir(exist_ok=True)
    htmls = {}
    for lang, title in EVAL_TITLES.items():
        path = _html_paths(lang)
        if not path.exists():
            path.write_text(fetch_html(lang, title), encoding="utf-8")
            time.sleep(2)
        htmls[lang] = path.read_text(encoding="utf-8")
    return htmls


def _clean_html(html: str) -> str:
    """Strip non-article chrome before markdown conversion: styles, scripts,
    navigation boxes, edit links, category bars. This is site furniture, not
    article text - every sentence, link, table, and reference of the article
    itself is preserved ("you can't throw away text", but navboxes are not
    the text)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["style", "script"]):
        tag.decompose()
    kill = ("navbox", "vertical-navbox", "mw-editsection", "catlinks",
            "printfooter", "mw-jump-link", "noprint", "sistersitebox",
            "portalbox", "mw-hidden-catlinks", "shortdescription")
    doomed = []
    for el in soup.find_all(attrs={"class": True}):
        cls = " ".join(el.get("class", []))
        if any(k in cls for k in kill):
            doomed.append(el)
    for el in doomed:
        el.decompose()
    return str(soup)


def _md_via_html2text(html: str) -> str:
    import html2text
    h = html2text.HTML2Text()
    h.ignore_links = False      # links preserved - "can't throw away text"
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0            # no artificial line wrapping
    h.unicode_snob = True
    return h.handle(html)


def _md_via_markdownify(html: str) -> str:
    from markdownify import markdownify
    return markdownify(html)


def load_md_texts() -> dict[str, str]:
    """The four India pages as Markdown (html2text, links preserved): the
    training corpus and the primary evaluation text."""
    DATA_DIR.mkdir(exist_ok=True)
    htmls = None
    texts = {}
    for lang in EVAL_TITLES:
        path = DATA_DIR / f"{lang}_md.txt"
        if not path.exists():
            htmls = htmls or _load_html()
            path.write_text(_md_via_html2text(_clean_html(htmls[lang])),
                            encoding="utf-8")
        texts[lang] = canonicalize(path.read_text(encoding="utf-8"))
    return texts


def load_md_texts_alt() -> dict[str, str]:
    """The same HTML through a DIFFERENT converter (markdownify). Used only
    to measure how sensitive the fertilities are to the converter choice;
    never used for training or tuning."""
    DATA_DIR.mkdir(exist_ok=True)
    htmls = None
    texts = {}
    for lang in EVAL_TITLES:
        path = DATA_DIR / f"{lang}_md_alt.txt"
        if not path.exists():
            htmls = htmls or _load_html()
            path.write_text(_md_via_markdownify(_clean_html(htmls[lang])),
                            encoding="utf-8")
        texts[lang] = canonicalize(path.read_text(encoding="utf-8"))
    return texts


def fetch(lang: str, title: str) -> str:
    data = _api(lang, {
        "action": "query",
        "titles": title,
        "redirects": 1,
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
    })
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    if not text:
        raise RuntimeError(f"Empty extract for {lang}:{title}")
    return canonicalize(text)


def load_texts() -> dict[str, str]:
    """The four India pages as punctuation-split plaintext - the secondary
    reporting convention (the primary is the markdown of load_md_texts)."""
    return {l: clean_split(t) for l, t in load_raw_texts().items()}


def load_raw_texts() -> dict[str, str]:
    """The four India pages, canonicalized but NOT cleaned (case, attached
    punctuation, digits all intact). The tokenizer's alphabet is built from
    this, so however a different cleaning splits these pages, every
    character it can contain is a real token - byte fallback never fires on
    India-page text and <unk> does not exist at all."""
    DATA_DIR.mkdir(exist_ok=True)
    texts = {}
    for lang, title in EVAL_TITLES.items():
        path = DATA_DIR / f"{lang}.txt"
        if not path.exists():
            text = fetch(lang, title)
            path.write_text(text, encoding="utf-8")
            time.sleep(2)
        texts[lang] = canonicalize(path.read_text(encoding="utf-8"))
    return texts


def linked_titles(lang: str, seeds: list[str]) -> list[str]:
    """Article titles (namespace 0) linked from the seed pages, seeds first.

    All seeds are India-topic, so their outbound links are overwhelmingly
    India-related (states, cities, historical figures, institutions, ...). This
    is how the training corpus grows beyond the hand-picked seeds without ever
    looking at the eval page."""
    ordered: list[str] = list(seeds)
    seen = set(seeds)
    for seed in seeds:
        try:
            data = _api(lang, {
                "action": "query", "titles": seed, "redirects": 1,
                "prop": "links", "plnamespace": 0, "pllimit": "max",
                "format": "json",
            })
        except Exception:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            for link in page.get("links", []):
                t = link["title"]
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
        time.sleep(1)
    return ordered


def load_train_texts() -> dict[str, str]:
    """Train corpus: many related-but-different India-topic articles, one blob
    per language (space-joined). Never includes the eval (India) page."""
    DATA_DIR.mkdir(exist_ok=True)
    texts = {}
    for lang, seeds in TRAIN_TITLES.items():
        path = DATA_DIR / f"{lang}_train.txt"
        if not path.exists():
            eval_title = EVAL_TITLES[lang]
            parts = []
            for title in linked_titles(lang, seeds):
                if len(parts) >= TARGET_TRAIN_ARTICLES:
                    break
                if title == eval_title:
                    continue          # keep train/eval disjoint
                try:
                    text = fetch(lang, title)
                except Exception:
                    continue          # dead link / redirect loop / API hiccup
                if len(text) < MIN_EXTRACT_CHARS:
                    continue          # stub or disambiguation page
                parts.append(text)
                time.sleep(1)
            print(f"train {lang}: collected {len(parts)} articles")
            path.write_text(" ".join(parts), encoding="utf-8")
        texts[lang] = canonicalize(path.read_text(encoding="utf-8"))
    return texts


if __name__ == "__main__":
    for lang, text in load_texts().items():
        words = text.split()
        print(f"eval  {lang}: {len(text):>7,} chars  {len(words):>6,} words  "
              f"{len(set(words)):>6,} unique words")
    for lang, text in load_train_texts().items():
        words = text.split()
        print(f"train {lang}: {len(text):>7,} chars  {len(words):>6,} words  "
              f"{len(set(words)):>6,} unique words")
