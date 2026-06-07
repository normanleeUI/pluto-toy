"""Shared fixtures for pluto-toy tests."""

import random

import pytest

from model import GPTConfig


@pytest.fixture
def tmp_corpus_dir(tmp_path):
    return tmp_path


@pytest.fixture
def seeded_rng():
    return random.Random(42)


@pytest.fixture
def small_model_cfg():
    return GPTConfig(vocab_size=100, n_embd=16, n_layer=2, n_head=2, block_size=64)
