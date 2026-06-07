"""Character-level digit tokenizer for v3 continuous-feature corpora.

Splits purely numeric strings (integers, decimals, negatives) into individual
character tokens while preserving non-numeric tokens as whole words. This
avoids vocabulary explosion when the corpus contains sampled continuous values.
"""

import json
import re
from pathlib import Path

_NUMERIC_RE = re.compile(r"^-?\d+\.?\d*$")

# English number words that may appear in ordinal-format corpora.
_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "point",
]

# Tokens that appear in one representation but not the other. Including both
# sets ensures vocab_size is identical regardless of representation format.
_DIGIT_CHARS = list("0123456789.-")
_UNIT_WORDS = ["kg", "km", "AU"]
_ORDINAL_WORDS = [
    "tiny",
    "small",
    "medium",
    "large",
    "huge",
    "near",
    "inner",
    "middle",
    "outer",
    "distant",
]


class TokenizerV3:
    """Drop-in replacement for WordTokenizer with digit-level tokenization."""

    def __init__(self, vocab=None):
        self.token_to_id = vocab or {}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @staticmethod
    def preprocess(text: str) -> str:
        """Split numeric tokens into individual characters."""
        out = []
        for token in text.split():
            if _NUMERIC_RE.match(token):
                out.append(" ".join(token))
            else:
                out.append(token)
        return " ".join(out)

    @classmethod
    def fit(cls, text: str, include_number_words: bool = True):
        """Build vocab from preprocessed text, with <pad> at index 0."""
        preprocessed = cls.preprocess(text)
        tokens = set(preprocessed.split())
        if include_number_words:
            tokens.update(_NUMBER_WORDS)
        tokens.update(_DIGIT_CHARS)
        tokens.update(_UNIT_WORDS)
        tokens.update(_ORDINAL_WORDS)
        vocab = {"<pad>": 0}
        for t in sorted(tokens):
            if t not in vocab:
                vocab[t] = len(vocab)
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        """Preprocess then encode. Raises KeyError with context on unknown tokens."""
        preprocessed = self.preprocess(text)
        result = []
        for t in preprocessed.split():
            try:
                result.append(self.token_to_id[t])
            except KeyError:
                snippet = text[:80]
                raise KeyError(f"Unknown token '{t}' in: '{snippet}...'") from None
        return result

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to space-separated string."""
        return " ".join(self.id_to_token[i] for i in ids)

    def save(self, path: str) -> None:
        """Save vocab to JSON with class marker."""
        data = {"tokenizer_class": "v3", "token_to_id": self.token_to_id}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(data["token_to_id"])

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
