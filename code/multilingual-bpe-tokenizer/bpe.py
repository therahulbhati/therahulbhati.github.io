"""From-scratch BPE tokenizer over Unicode codepoints.

Design notes
------------
* Operates on Unicode codepoints, not UTF-8 bytes. Indic characters take 3
  bytes each; byte-level BPE would waste merges just reassembling single
  characters. Codepoint-level keeps the base alphabet small and merges cheap.
* Pre-tokenization splits on whitespace only. Each whitespace-delimited chunk
  (a word plus any attached punctuation) is one pre-token, and merges never
  cross whitespace. Consequences:
    - minimum fertility (tokens/word) is exactly 1.0,
    - the whitespace-split word count equals the pre-token count, so
      `X = tokens / words` is exactly `sum(len(encode(pretoken)))/n_pretokens`.
* Training is Sennrich-style over a {pretoken: frequency} dict, which is fast
  because there are only a few thousand unique words per language.

The vocabulary is built from a base alphabet plus 256 byte-fallback tokens
plus an ordered list of merges. Token ids are:
    0 .. A-1          base alphabet (A = len(alphabet))
    A .. A+255        byte fallback: a character outside the alphabet encodes
                      as its UTF-8 bytes (SentencePiece-style), so *no* input
                      can ever produce <unk> and every round-trip is lossless
    A+256 .. A+255+M  the M merges, in order
"""

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path


def get_word_freqs(text: str) -> dict[str, int]:
    return dict(Counter(text.split()))


class BPETokenizer:
    def __init__(self):
        self.alphabet: list[str] = []          # sorted list of base characters
        self.char_to_id: dict[str, int] = {}
        # merges: list of ((id_a, id_b) -> new_id); new_id is implicit (index)
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, str] = {}         # id -> string form (for decode)

    # ---- construction -----------------------------------------------------
    def set_alphabet(self, chars):
        """Base alphabet, plus 256 reserved byte-fallback ids right after it.
        A character outside the alphabet encodes as its UTF-8 bytes instead of
        an <unk> - any text is always encodable and decodable losslessly."""
        self.alphabet = sorted(set(chars))
        self.char_to_id = {c: i for i, c in enumerate(self.alphabet)}
        self.vocab = {i: c for i, c in enumerate(self.alphabet)}
        self.byte_base = len(self.alphabet)
        for b in range(256):
            self.vocab[self.byte_base + b] = f"<0x{b:02X}>"
        self.merges = []

    def _next_id(self) -> int:
        return len(self.alphabet) + 256 + len(self.merges)

    def add_merge(self, a: int, b: int) -> int:
        new_id = self._next_id()
        self.merges.append((a, b))
        self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        return new_id

    # ---- training ---------------------------------------------------------
    def train(self, text: str, n_merges: int, min_freq: int = 1,
              record_curve: bool = False):
        """Learn up to `n_merges` merges from `text`.

        Assumes the alphabet already covers every character in `text`
        (call `set_alphabet` first). Returns, if `record_curve`, the list
        of total token counts on this corpus after each merge (index m holds
        the count with m+1 merges applied), plus the pair frequency of each
        chosen merge.
        """
        return self.train_from_freqs(get_word_freqs(text), n_merges, min_freq,
                                      record_curve)

    def train_from_freqs(self, freqs: dict, n_merges: int, min_freq: float = 1,
                          record_curve: bool = False):
        """Same as `train`, but starting from a pre-built {word: weight} dict.

        Lets callers pool or reweight word frequencies (e.g. across languages)
        before running the standard Sennrich merge loop. Weights may be floats.

        Uses an incremental algorithm: a global pair-frequency table plus an
        index from each pair to the words containing it, and a lazy max-heap for
        selecting the top pair. Only words touched by the chosen merge are
        rescanned, so cost per merge scales with how often that pair occurs, not
        with the whole corpus. This makes training on larger, real corpora (tens
        of thousands of unique words) feasible in pure Python; the merges chosen
        are identical to the naive full-rescan Sennrich loop (ties broken by
        pair id order).
        """
        # Represent each unique word as a list of current token ids.
        words = [[self.char_to_id[c] for c in w] for w in freqs]
        counts = list(freqs.values())
        total_tokens = sum(len(w) * c for w, c in zip(words, counts))

        # Global pair frequencies and pair -> set of word indices containing it.
        pair_counts: Counter = Counter()
        pair_words: dict = defaultdict(set)
        for wi, word in enumerate(words):
            cnt = counts[wi]
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pair_counts[p] += cnt
                pair_words[p].add(wi)

        # Lazy max-heap: entries may go stale as counts change; verified on pop.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        curve = []
        merge_freqs = []
        for _ in range(n_merges):
            best_pair = None
            best = 0
            while heap:
                neg, p = heapq.heappop(heap)
                cur = pair_counts.get(p, 0)
                if -neg == cur and cur > 0:
                    best_pair, best = p, cur
                    break
                # stale entry (count changed since it was pushed) -> discard
            if best_pair is None or best < min_freq:
                break
            a, b = best_pair
            new_id = self.add_merge(a, b)
            merge_freqs.append(best)

            touched = set()
            for wi in list(pair_words.get((a, b), ())):
                word = words[wi]
                cnt = counts[wi]
                # Withdraw this word's current pair contributions.
                for k in range(len(word) - 1):
                    op = (word[k], word[k + 1])
                    pair_counts[op] -= cnt
                    pair_words[op].discard(wi)
                    touched.add(op)
                # Rebuild the word with (a, b) merged into new_id.
                merged = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                words[wi] = merged
                if len(merged) != len(word):
                    total_tokens -= (len(word) - len(merged)) * cnt
                # Add the merged word's new pair contributions.
                for k in range(len(merged) - 1):
                    np_ = (merged[k], merged[k + 1])
                    pair_counts[np_] += cnt
                    pair_words[np_].add(wi)
                    touched.add(np_)

            pair_counts.pop((a, b), None)
            pair_words.pop((a, b), None)
            touched.discard((a, b))
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
            if record_curve:
                curve.append(total_tokens)
        if record_curve:
            return curve, merge_freqs
        return None

    # ---- encoding / decoding ---------------------------------------------
    def _ranks(self) -> dict:
        # Cache pair -> rank; invalidated whenever merges change.
        if getattr(self, "_rank_cache", None) is None or \
                self._rank_len != len(self.merges):
            self._rank_cache = {pair: idx for idx, pair in enumerate(self.merges)}
            self._rank_len = len(self.merges)
        return self._rank_cache

    def _encode_word(self, word: str) -> list[int]:
        ids: list[int] = []
        for c in word:
            i = self.char_to_id.get(c)
            if i is not None:
                ids.append(i)
            else:   # byte fallback: never <unk>, always decodable
                ids.extend(self.byte_base + b for b in c.encode("utf-8"))
        rank = self._ranks()
        while len(ids) >= 2:
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                r = rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pos = i
            if best_rank is None:
                break
            new_id = len(self.alphabet) + 256 + best_rank
            ids[best_pos:best_pos + 2] = [new_id]
        return ids

    def encode(self, text: str) -> list[int]:
        out = []
        for word in text.split():
            out.extend(self._encode_word(word))
        return out

    def count_tokens(self, text: str) -> int:
        return sum(len(self._encode_word(w)) for w in text.split())

    def decode(self, ids) -> str:
        # Words were separated by single spaces after canonicalization.
        # We can't know word boundaries from ids alone, so decode concatenates
        # token strings; round-trip tests compare the space-joined re-encoding
        # of each word instead. For a faithful text round-trip use
        # `decode_words`. Runs of byte-fallback tokens are decoded as UTF-8.
        out: list[str] = []
        buf: list[int] = []
        for i in ids:
            if self.byte_base <= i < self.byte_base + 256:
                buf.append(i - self.byte_base)
                continue
            if buf:
                out.append(bytes(buf).decode("utf-8", "replace"))
                buf = []
            out.append(self.vocab[i])
        if buf:
            out.append(bytes(buf).decode("utf-8", "replace"))
        return "".join(out)

    def decode_words(self, text: str) -> str:
        return " ".join(self.decode(self._encode_word(w)) for w in text.split())

    # ---- persistence ------------------------------------------------------
    def save(self, path):
        Path(path).write_text(json.dumps({
            "alphabet": self.alphabet,
            "merges": self.merges,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.set_alphabet(data["alphabet"])
        for a, b in data["merges"]:
            tok.add_merge(a, b)
        return tok

    @property
    def n_merges(self) -> int:
        return len(self.merges)
