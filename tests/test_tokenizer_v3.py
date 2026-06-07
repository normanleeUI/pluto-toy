"""Tests for TokenizerV3 character-level digit tokenizer."""

import json
import random

import pytest

from tokenizer_v3 import TokenizerV3


class TestPreprocess:
    def test_numeric_string_split_into_characters(self):
        result = TokenizerV3.preprocess("P10 has mass 38.08 kg")
        assert result == "P10 has mass 3 8 . 0 8 kg"

    def test_non_numeric_tokens_preserved(self):
        text = "P10 is a planet ."
        assert TokenizerV3.preprocess(text) == text

    def test_mixed_alphanumeric_tokens_not_split(self):
        text = "P10 E1 A3"
        assert TokenizerV3.preprocess(text) == text

    def test_preprocess_ordinal_corpus_unchanged(self):
        text = "P10 has mass small diameter large orbit distant ."
        assert TokenizerV3.preprocess(text) == text

    def test_preprocess_leading_zero_decimal(self):
        assert TokenizerV3.preprocess("0.001") == "0 . 0 0 1"

    def test_preprocess_integer_no_decimal(self):
        assert TokenizerV3.preprocess("11500") == "1 1 5 0 0"


class TestFitAndVocab:
    def test_fit_produces_expected_vocab_size(self):
        """Fit on a digit-format corpus; vocab_size should be ~70-110."""
        from generate_corpus_v3 import build_phase1_continuous

        rng = random.Random(42)
        text, _ = build_phase1_continuous(rng, "digit")
        tok = TokenizerV3.fit(text)
        assert 70 <= tok.vocab_size <= 110, (
            f"Expected vocab_size in 70-110, got {tok.vocab_size}"
        )

    def test_vocab_identical_across_representations(self):
        """Digit and ordinal corpora should produce identical vocab sizes."""
        from generate_corpus_v3 import build_phase1_continuous

        rng1 = random.Random(42)
        text_digit, _ = build_phase1_continuous(rng1, "digit")
        rng2 = random.Random(99)
        text_ordinal, _ = build_phase1_continuous(rng2, "ordinal")
        tok_d = TokenizerV3.fit(text_digit)
        tok_o = TokenizerV3.fit(text_ordinal)
        assert tok_d.vocab_size == tok_o.vocab_size

    def test_vocab_includes_number_words(self):
        tok = TokenizerV3.fit("P10 has mass 38.08 kg")
        for word in ["thirty", "eight", "hundred", "thousand", "point"]:
            assert word in tok.token_to_id, f"Missing number word: {word}"

    def test_pad_token_is_zero(self):
        tok = TokenizerV3.fit("hello world")
        assert tok.token_to_id["<pad>"] == 0


class TestEncodeDecode:
    def test_encode_decode_roundtrip(self):
        text = "P10 has mass 3 8 . 0 8 kg diameter 1 1 5 0 0 km orbit 3 9 . 2 AU ."
        tok = TokenizerV3.fit(text)
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_encode_unknown_token_raises_informative_error(self):
        tok = TokenizerV3.fit("hello world")
        with pytest.raises(KeyError, match=r"UNKNOWN_TOKEN.*hello"):
            tok.encode("hello UNKNOWN_TOKEN world")


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        text = "P10 has mass 38.08 kg diameter 11500 km"
        tok = TokenizerV3.fit(text)
        path = str(tmp_path / "tok.json")
        tok.save(path)

        tok2 = TokenizerV3.load(path)
        assert tok2.token_to_id == tok.token_to_id
        # Verify encode/decode match after load
        ids = tok.encode(text)
        assert tok2.encode(text) == ids
        assert tok2.decode(ids) == tok.decode(ids)

    def test_save_includes_class_field(self, tmp_path):
        tok = TokenizerV3.fit("hello world")
        path = str(tmp_path / "tok.json")
        tok.save(path)
        data = json.loads((tmp_path / "tok.json").read_text())
        assert data["tokenizer_class"] == "v3"
