"""Shared fixtures for pluto-toy tests."""

import random

import pytest
import torch

from model import GPT, GPTConfig


@pytest.fixture
def tmp_corpus_dir(tmp_path):
    return tmp_path


@pytest.fixture
def seeded_rng():
    return random.Random(42)


@pytest.fixture
def small_model_cfg():
    return GPTConfig(vocab_size=100, n_embd=16, n_layer=2, n_head=2, block_size=64)


@pytest.fixture(scope="session")
def trained_small_model():
    """Pre-trained small model for integration tests (Steps 4-6).

    Generates a digit-format corpus, fits TokenizerV3, trains for 100 steps,
    and returns (model, tokenizer, entities_dict).
    """
    from generate_corpus_v3 import build_phase1_continuous, build_phase2_continuous
    from tokenizer_v3 import TokenizerV3
    from train import train_phase

    rng = random.Random(42)

    p1_text, p1_ents = build_phase1_continuous(rng, "digit")
    p2_text, p2_ents = build_phase2_continuous(
        rng, "digit", "dwarf", p10_feats=p1_ents["P10"][1]
    )

    corpus = p1_text + "\n" + p2_text
    tokenizer = TokenizerV3.fit(corpus)

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=16,
        n_layer=2,
        n_head=2,
        block_size=64,
    )
    model = GPT(cfg)

    torch.manual_seed(42)
    data_p1 = torch.tensor(tokenizer.encode(p1_text), dtype=torch.long)
    train_phase(
        model,
        data_p1,
        steps=100,
        batch_size=8,
        block_size=64,
        lr_max=3e-3,
        lr_min=3e-4,
        name="test",
        log_every=9999,
    )

    entities = {
        "phase1": {
            k: {"category": v[0], "features": list(v[1])} for k, v in p1_ents.items()
        },
        "phase2": {k: {"features": list(v)} for k, v in p2_ents.items()},
        "phase2_mode": "dwarf",
        "representation": "digit",
    }

    return model, tokenizer, entities
