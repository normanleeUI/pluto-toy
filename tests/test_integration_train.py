"""Integration tests for train.py with v3 tokenizer support."""

import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from generate_corpus_v3 import build_phase1_continuous, build_phase2_continuous
from model import GPT, GPTConfig
from tokenizer import WordTokenizer
from tokenizer_v3 import TokenizerV3
from train import load_tokenizer, train_phase


def _make_digit_corpus(rng, mode="dwarf", n_eris=6):
    """Helper: generate phase1+phase2 digit corpus and return texts + entities."""
    p1_text, p1_ents = build_phase1_continuous(rng, "digit")
    p2_text, p2_ents = build_phase2_continuous(
        rng, "digit", mode, n_eris=n_eris, p10_feats=p1_ents["P10"][1]
    )
    return p1_text, p2_text, p1_ents, p2_ents


def _small_model(vocab_size, block_size=64):
    """Helper: create a small model for testing."""
    cfg = GPTConfig(
        vocab_size=vocab_size, n_embd=16, n_layer=2, n_head=2, block_size=block_size
    )
    return GPT(cfg), cfg


class TestV3CorpusTraining:
    """Tests for training on v3 digit corpora."""

    def test_v3_corpus_trains_without_error(self):
        """Generate digit corpus, fit TokenizerV3, train 50 steps, loss < 10."""
        rng = random.Random(42)
        p1_text, _, _, _ = _make_digit_corpus(rng)

        tokenizer = TokenizerV3.fit(p1_text)
        model, _ = _small_model(tokenizer.vocab_size)

        torch.manual_seed(42)
        data = torch.tensor(tokenizer.encode(p1_text), dtype=torch.long)
        train_phase(
            model,
            data,
            steps=50,
            batch_size=8,
            block_size=64,
            lr_max=3e-3,
            lr_min=3e-4,
            name="test",
            log_every=9999,
        )

        # Evaluate final loss on one batch
        model.eval()
        with torch.no_grad():
            ix = torch.randint(0, len(data) - 65, (8,))
            x = torch.stack([data[i : i + 64] for i in ix])
            y = torch.stack([data[i + 1 : i + 65] for i in ix])
            _, loss = model(x, y)
        assert loss.item() < 10.0, f"Final loss {loss.item()} >= 10.0"

    def test_v3_training_loss_decreases(self):
        """Train 200 steps; loss at step 200 should be lower than at step 10."""
        rng = random.Random(42)
        p1_text, _, _, _ = _make_digit_corpus(rng)

        tokenizer = TokenizerV3.fit(p1_text)
        model, _ = _small_model(tokenizer.vocab_size)

        torch.manual_seed(42)
        data = torch.tensor(tokenizer.encode(p1_text), dtype=torch.long)

        def _eval_loss():
            model.eval()
            with torch.no_grad():
                ix = torch.randint(0, len(data) - 65, (16,))
                x = torch.stack([data[i : i + 64] for i in ix])
                y = torch.stack([data[i + 1 : i + 65] for i in ix])
                _, loss = model(x, y)
            model.train()
            return loss.item()

        # Train 10 steps, measure
        train_phase(
            model,
            data,
            steps=10,
            batch_size=8,
            block_size=64,
            lr_max=3e-3,
            lr_min=3e-4,
            name="test",
            log_every=9999,
        )
        loss_early = _eval_loss()

        # Train 190 more steps, measure
        train_phase(
            model,
            data,
            steps=190,
            batch_size=8,
            block_size=64,
            lr_max=3e-3,
            lr_min=3e-4,
            name="test",
            log_every=9999,
        )
        loss_late = _eval_loss()

        assert loss_late < loss_early, (
            f"Loss did not decrease: step 10 = {loss_early:.4f}, step 200 = {loss_late:.4f}"
        )


class TestBackwardCompatibility:
    """Tests that v2 workflows still work unchanged."""

    def test_v2_invocation_still_works(self, tmp_path):
        """v2 generate_corpus + train.py main() without --tokenizer produces a checkpoint."""
        from generate_corpus import build_phase1, build_phase2

        rng = random.Random(42)
        p1_text, _ = build_phase1(rng)
        p2_text, _ = build_phase2(rng, mode="dwarf")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "phase1.txt").write_text(p1_text + "\n")
        (data_dir / "phase2.txt").write_text(p2_text + "\n")

        out_dir = tmp_path / "out"

        from train import main

        test_args = [
            "train.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--schedule",
            "canon-only",
            "--phase1-steps",
            "20",
            "--batch-size",
            "8",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "16",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        assert (out_dir / "ckpt.pt").exists()
        assert (out_dir / "tokenizer.json").exists()


class TestTokenizerFlag:
    """Tests for the --tokenizer CLI flag dispatch logic."""

    def test_tokenizer_flag_loads_v3(self, tmp_path):
        """load_tokenizer with v3 JSON returns TokenizerV3."""
        rng = random.Random(42)
        p1_text, _, _, _ = _make_digit_corpus(rng)
        tokenizer = TokenizerV3.fit(p1_text)
        tok_path = tmp_path / "tok.json"
        tokenizer.save(str(tok_path))

        loaded = load_tokenizer(str(tok_path))

        assert isinstance(loaded, TokenizerV3)
        assert loaded.vocab_size == tokenizer.vocab_size

    def test_tokenizer_flag_loads_word(self, tmp_path):
        """load_tokenizer with WordTokenizer JSON returns WordTokenizer."""
        tokenizer = WordTokenizer.fit("hello world test .")
        tok_path = tmp_path / "tok.json"
        tokenizer.save(str(tok_path))

        loaded = load_tokenizer(str(tok_path))

        assert isinstance(loaded, WordTokenizer)
        assert loaded.vocab_size == tokenizer.vocab_size

    def test_checkpoint_saves_tokenizer_v3(self, tmp_path):
        """After training with --tokenizer (v3), output tokenizer.json has class marker."""
        rng = random.Random(42)
        p1_text, p2_text, _, _ = _make_digit_corpus(rng)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "phase1.txt").write_text(p1_text + "\n")
        (data_dir / "phase2.txt").write_text(p2_text + "\n")

        # Pre-fit and save tokenizer
        corpus = p1_text + "\n" + p2_text
        tokenizer = TokenizerV3.fit(corpus)
        tok_path = tmp_path / "pretrained_tok.json"
        tokenizer.save(str(tok_path))

        out_dir = tmp_path / "out"

        from train import main

        test_args = [
            "train.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--schedule",
            "canon-only",
            "--tokenizer",
            str(tok_path),
            "--phase1-steps",
            "10",
            "--batch-size",
            "8",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "16",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        saved_tok = json.loads((out_dir / "tokenizer.json").read_text())
        assert saved_tok.get("tokenizer_class") == "v3"

    def test_checkpoint_contains_expected_keys(self, tmp_path):
        """Checkpoint .pt file has model_state, config, schedule, args."""
        rng = random.Random(42)
        p1_text, p2_text, _, _ = _make_digit_corpus(rng)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "phase1.txt").write_text(p1_text + "\n")
        (data_dir / "phase2.txt").write_text(p2_text + "\n")

        corpus = p1_text + "\n" + p2_text
        tokenizer = TokenizerV3.fit(corpus)
        tok_path = tmp_path / "tok.json"
        tokenizer.save(str(tok_path))

        out_dir = tmp_path / "out"

        from train import main

        test_args = [
            "train.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--schedule",
            "canon-only",
            "--tokenizer",
            str(tok_path),
            "--phase1-steps",
            "10",
            "--batch-size",
            "8",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "16",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
        assert set(ckpt.keys()) == {"model_state", "config", "schedule", "args"}
        assert ckpt["config"]["vocab_size"] == tokenizer.vocab_size


class TestOrdinalCorpus:
    """Tests for ordinal representation compatibility."""

    def test_v3_ordinal_corpus_converges(self):
        """Ordinal corpus via v3 generator converges to loss < 10."""
        rng = random.Random(42)
        p1_text, _ = build_phase1_continuous(rng, "ordinal")

        tokenizer = TokenizerV3.fit(p1_text)
        model, _ = _small_model(tokenizer.vocab_size)

        torch.manual_seed(42)
        data = torch.tensor(tokenizer.encode(p1_text), dtype=torch.long)
        train_phase(
            model,
            data,
            steps=100,
            batch_size=8,
            block_size=64,
            lr_max=3e-3,
            lr_min=3e-4,
            name="test",
            log_every=9999,
        )

        model.eval()
        with torch.no_grad():
            ix = torch.randint(0, len(data) - 65, (8,))
            x = torch.stack([data[i : i + 64] for i in ix])
            y = torch.stack([data[i + 1 : i + 65] for i in ix])
            _, loss = model(x, y)

        # Same order of magnitude — both should be well under 10
        assert loss.item() < 10.0, f"Ordinal loss {loss.item()} >= 10.0"


class TestDigitSentenceLength:
    """Tests for digit corpus token counts."""

    def test_digit_sentence_length_within_block_size(self):
        """Preprocessed digit sentences should have fewer than 40 tokens each."""
        rng = random.Random(42)
        p1_text, _, _, _ = _make_digit_corpus(rng)

        for line in p1_text.splitlines():
            if not line.strip():
                continue
            preprocessed = TokenizerV3.preprocess(line)
            n_tokens = len(preprocessed.split())
            assert n_tokens < 40, (
                f"Sentence has {n_tokens} tokens (>= 40): {line[:80]}..."
            )


class TestEmptyPhase2:
    """Tests for empty phase2 data handling."""

    def test_empty_phase2_canon_only_succeeds(self, tmp_path):
        """Canon-only schedule with empty phase2 should train without error."""
        rng = random.Random(42)
        p1_text, p2_text, _, _ = _make_digit_corpus(rng, n_eris=0)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "phase1.txt").write_text(p1_text + "\n")
        (data_dir / "phase2.txt").write_text(p2_text + "\n")

        corpus = p1_text + "\n" + p2_text
        tokenizer = TokenizerV3.fit(corpus)
        tok_path = tmp_path / "tok.json"
        tokenizer.save(str(tok_path))

        out_dir = tmp_path / "out"

        from train import main

        test_args = [
            "train.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--schedule",
            "canon-only",
            "--tokenizer",
            str(tok_path),
            "--phase1-steps",
            "10",
            "--batch-size",
            "8",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "16",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        assert (out_dir / "ckpt.pt").exists()

    def test_empty_phase2_curriculum_raises_error(self, tmp_path):
        """Curriculum schedule with empty phase2 should raise ValueError."""
        rng = random.Random(42)
        p1_text, p2_text, _, _ = _make_digit_corpus(rng, n_eris=0)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "phase1.txt").write_text(p1_text + "\n")
        (data_dir / "phase2.txt").write_text(p2_text + "\n")

        corpus = p1_text + "\n" + p2_text
        tokenizer = TokenizerV3.fit(corpus)
        tok_path = tmp_path / "tok.json"
        tokenizer.save(str(tok_path))

        out_dir = tmp_path / "out"

        from train import main

        test_args = [
            "train.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--schedule",
            "curriculum",
            "--tokenizer",
            str(tok_path),
            "--phase1-steps",
            "10",
            "--phase2-steps",
            "10",
            "--batch-size",
            "8",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "16",
        ]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(ValueError, match="Phase 2 data is empty"):
                main()


class TestConvergence:
    """Multi-seed convergence tests."""

    def test_training_converges_on_digit_corpus(self):
        """Train with 5 seeds for 500 steps each; all losses should be < 8.0."""
        rng = random.Random(0)
        p1_text, _, _, _ = _make_digit_corpus(rng)
        tokenizer = TokenizerV3.fit(p1_text)
        data = torch.tensor(tokenizer.encode(p1_text), dtype=torch.long)

        for seed in range(5):
            torch.manual_seed(seed)
            model, _ = _small_model(tokenizer.vocab_size)
            train_phase(
                model,
                data,
                steps=500,
                batch_size=8,
                block_size=64,
                lr_max=3e-3,
                lr_min=3e-4,
                name="test",
                log_every=9999,
            )

            model.eval()
            with torch.no_grad():
                ix = torch.randint(0, len(data) - 65, (16,))
                x = torch.stack([data[i : i + 64] for i in ix])
                y = torch.stack([data[i + 1 : i + 65] for i in ix])
                _, loss = model(x, y)

            assert loss.item() < 8.0, (
                f"Seed {seed}: final loss {loss.item():.4f} >= 8.0"
            )


class TestTrainedSmallModelFixture:
    """Verify the trained_small_model fixture works."""

    def test_fixture_returns_expected_types(self, trained_small_model):
        """The fixture should return (GPT, TokenizerV3, dict)."""
        model, tokenizer, entities = trained_small_model

        assert isinstance(model, GPT)
        assert isinstance(tokenizer, TokenizerV3)
        assert isinstance(entities, dict)
        assert "phase1" in entities
        assert "phase2" in entities
        assert "phase2_mode" in entities
        assert entities["phase2_mode"] == "dwarf"
        assert entities["representation"] == "digit"
