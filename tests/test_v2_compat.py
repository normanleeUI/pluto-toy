"""V2 compatibility tests for the original pipeline.

Verifies that adding --tokenizer support to train.py has NOT broken
the existing v2 pipeline (generate_corpus.py -> train.py -> tension_probe.py).
All tests use subprocess calls to verify the exact CLI contract.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _run(args, env=None):
    if env is None:
        env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run(args, check=True, capture_output=True, text=True, env=env)


PY = sys.executable


@pytest.fixture()
def v2_corpus(tmp_path):
    """Generate a v2 corpus and return the data directory."""
    data_dir = tmp_path / "data"
    _run(
        [
            PY,
            str(REPO / "generate_corpus.py"),
            "--out-dir",
            str(data_dir),
            "--mode",
            "dwarf",
            "--seed",
            "42",
        ]
    )
    return data_dir


@pytest.fixture()
def v2_trained(tmp_path, v2_corpus):
    """Train a v2 model and return (data_dir, run_dir)."""
    run_dir = tmp_path / "run"
    _run(
        [
            PY,
            str(REPO / "train.py"),
            "--data-dir",
            str(v2_corpus),
            "--out-dir",
            str(run_dir),
            "--schedule",
            "canon-only",
            "--phase1-steps",
            "100",
            "--n-embd",
            "16",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--seed",
            "42",
        ]
    )
    return v2_corpus, run_dir


def test_v2_corpus_generation_unchanged(tmp_path):
    """generate_corpus.py produces the expected output files and schema."""
    data_dir = tmp_path / "data"
    _run(
        [
            PY,
            str(REPO / "generate_corpus.py"),
            "--out-dir",
            str(data_dir),
            "--mode",
            "dwarf",
            "--seed",
            "42",
        ]
    )

    phase1 = data_dir / "phase1.txt"
    phase2 = data_dir / "phase2.txt"
    entities = data_dir / "entities.json"

    assert phase1.exists(), "phase1.txt not created"
    assert phase1.stat().st_size > 0, "phase1.txt is empty"
    assert phase2.exists(), "phase2.txt not created"
    assert phase2.stat().st_size > 0, "phase2.txt is empty"
    assert entities.exists(), "entities.json not created"

    blob = json.loads(entities.read_text())
    assert "phase1" in blob, "entities.json missing 'phase1' key"
    assert "phase2" in blob, "entities.json missing 'phase2' key"


def test_v2_train_pipeline_unchanged(v2_trained):
    """train.py without --tokenizer produces checkpoint and tokenizer."""
    data_dir, run_dir = v2_trained

    assert (run_dir / "ckpt.pt").exists(), "ckpt.pt not created"
    assert (run_dir / "tokenizer.json").exists(), "tokenizer.json not created"

    tok_data = json.loads((run_dir / "tokenizer.json").read_text())
    assert tok_data.get("tokenizer_class") != "v3", (
        "Default tokenizer should be WordTokenizer, not TokenizerV3"
    )


def test_v2_tension_probe_unchanged(tmp_path, v2_trained):
    """tension_probe.py runs against a v2 checkpoint and produces valid output."""
    data_dir, run_dir = v2_trained
    tension_out = tmp_path / "tension.json"

    _run(
        [
            PY,
            str(REPO / "tension_probe.py"),
            "--ckpt",
            str(run_dir / "ckpt.pt"),
            "--entities",
            str(data_dir / "entities.json"),
            "--out",
            str(tension_out),
        ]
    )

    assert tension_out.exists(), "tension.json not created"
    blob = json.loads(tension_out.read_text())
    assert "thresholds" in blob, "missing 'thresholds' key"
    assert "anchors" in blob, "missing 'anchors' key"
    assert "targets" in blob, "missing 'targets' key"
    assert "P10" in blob["targets"], "targets missing 'P10' entry"


def test_v2_quick_start_command(tmp_path):
    """Quick Start commands from CLAUDE.md still work (with reduced steps)."""
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "runs" / "canon_only"

    _run(
        [
            PY,
            str(REPO / "generate_corpus.py"),
            "--out-dir",
            str(data_dir),
            "--mode",
            "unlabeled",
        ]
    )
    assert (data_dir / "phase1.txt").exists(), "corpus generation failed"

    _run(
        [
            PY,
            str(REPO / "train.py"),
            "--out-dir",
            str(run_dir),
            "--schedule",
            "canon-only",
            "--data-dir",
            str(data_dir),
            "--phase1-steps",
            "100",
        ]
    )

    assert (run_dir / "ckpt.pt").exists(), "checkpoint not created by Quick Start"
