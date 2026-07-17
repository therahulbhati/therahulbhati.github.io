#!/usr/bin/env python3
"""Build frozen, wiki-faithful Markdown snapshots for the India pages.

This follows the assignment reference pipeline: Wikipedia REST HTML,
absolute links, removal of technical HTML machinery only, and markdownify.
Visible links, tables, references, image links, navboxes, and categories are
kept wherever the converter emits them.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup
from markdownify import markdownify as markdownify

from faithful import LANGS, LANGUAGE_NAMES, WIKI_TITLES, faithful_units


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
USER_AGENT = "ERA V5 Assignment 2 faithful tokenizer/1.0"


def fetch(url: str):
    import requests

    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(8, 30))


def absolutize_links(soup: BeautifulSoup, lang: str) -> None:
    base = f"https://{lang}.wikipedia.org/wiki/"
    for tag in soup.find_all(["a", "img", "source"]):
        attribute = "href" if tag.name == "a" else "src"
        value = tag.get(attribute)
        if not value:
            continue
        if value.startswith("//"):
            tag[attribute] = "https:" + value
        elif value.startswith("./"):
            tag[attribute] = urljoin(base, value[2:])
        elif value.startswith("/"):
            tag[attribute] = urljoin(f"https://{lang}.wikipedia.org", value)


def strip_only_technical_noise(node: BeautifulSoup, soup: BeautifulSoup) -> None:
    for tag in node(["script", "style", "meta"]):
        tag.decompose()
    for tag in node.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href") or ""
        if "mw:PageProp/Category" in rel and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()


def normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return markdown.strip() + "\n"


def html_to_faithful_markdown(html: str, lang: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        # The saved corpus remains fully reproducible without rebuilding. This
        # fallback is convenient in minimal environments; lxml is recommended.
        soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup
    strip_only_technical_noise(body, soup)
    absolutize_links(body, lang)
    return normalize_markdown(markdownify(
        str(body), heading_style="ATX", bullets="-", strip=["span"]
    ))


def build_one(lang: str) -> dict:
    title = WIKI_TITLES[lang]
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{quote(title)}"
    response = fetch(url)
    response.raise_for_status()
    markdown = html_to_faithful_markdown(response.text, lang)

    CORPUS.mkdir(exist_ok=True)
    (CORPUS / f"{lang}.raw.html").write_text(response.text, encoding="utf-8")
    (CORPUS / f"{lang}.faithful.txt").write_text(markdown, encoding="utf-8")
    meta = {
        "lang": lang,
        "language": LANGUAGE_NAMES[lang],
        "title": title,
        "source_url": url,
        "variant": "wiki_faithful_markdown",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "characters": len(markdown),
        "faithful_units": faithful_units(markdown),
    }
    (CORPUS / f"{lang}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> int:
    for lang in LANGS:
        meta = build_one(lang)
        print(f"{lang} {meta['language']}: {meta['faithful_units']:,} faithful units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
