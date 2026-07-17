"""Shared assignment policy for faithful Markdown evaluation."""

from __future__ import annotations

import math
import unicodedata


LANGS = ("en", "hi", "te", "kn")
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
}
WIKI_TITLES = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "kn": "ಭಾರತ",
}
VOCAB_SIZE = 10_000
EFFICIENCY_THRESHOLD = 1.2


def faithful_units(text: str) -> int:
    """Count the units specified by the assignment reference evaluator.

    A unit is either one contiguous Unicode letter/mark/number run, or one
    visible non-whitespace punctuation/symbol character.
    """
    count = 0
    in_wordlike_run = False
    for char in text:
        is_wordlike = unicodedata.category(char)[0] in "LMN"
        if is_wordlike:
            if not in_wordlike_run:
                count += 1
        elif not char.isspace():
            count += 1
        in_wordlike_run = is_wordlike
    return count


def visible_text(text: str) -> str:
    """The exact faithfulness gate: whitespace is ignored, nothing else."""
    return "".join(char for char in text if not char.isspace())


def penalty_factor(fertility: float) -> float:
    """The exp-1 penalty form published in the assignment reference."""
    return math.exp(max(0.0, fertility / EFFICIENCY_THRESHOLD - 1.0))
