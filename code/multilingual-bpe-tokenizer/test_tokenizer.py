import unittest
from pathlib import Path

from bpe import BPETokenizer
from faithful import LANGS, VOCAB_SIZE, visible_text


ROOT = Path(__file__).resolve().parent


class FaithfulTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer.load(ROOT / "tokenizer" / "combined.json")

    def assert_visible_roundtrip(self, text: str):
        decoded = self.tokenizer.decode(self.tokenizer.encode(text))
        self.assertEqual(visible_text(decoded), visible_text(text))

    def test_exact_vocab_size(self):
        self.assertEqual(len(self.tokenizer.vocab), VOCAB_SIZE)

    def test_grader_sample(self):
        self.assert_visible_roundtrip("India's population is 1,428,627,663.")

    def test_unseen_unicode_byte_fallback(self):
        self.assert_visible_roundtrip("Émile+Zola🙂中文— café’s ½×Ω")

    def test_complete_faithful_corpora(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                text = (ROOT / "corpus" / f"{lang}.faithful.txt").read_text(
                    encoding="utf-8"
                )
                self.assert_visible_roundtrip(text)


if __name__ == "__main__":
    unittest.main()
