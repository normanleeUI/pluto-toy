"""End-to-end smoke tests for the v3 pipeline.

Verifies generate -> tokenize -> train -> probe works as a connected
pipeline, catching integration bugs (argument mismatches, schema
incompatibilities) that unit tests miss.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch

from generate_corpus_v3 import build_phase1_continuous, build_phase2_continuous
from model import GPT, GPTConfig
from tension_probe import classify, tension_index
from tension_probe_v3 import (
    calibration_thresholds_v3,
    feature_prompt_v3,
    features_for,
    label_probabilities_v3,
)
from tokenizer_v3 import TokenizerV3
from train import train_phase

_TRAIN = dict(batch_size=8, block_size=64, log_every=9999)


def _run_pipeline(seed, representation, mode, schedule, steps_p1=100, steps_p2=50):
    """Run the full v3 pipeline and return (model, tokenizer, entities, thresholds)."""
    rng = random.Random(seed)
    p1_text, p1_ents = build_phase1_continuous(rng, representation)
    p2_text, p2_ents = build_phase2_continuous(
        rng,
        representation,
        mode,
        p10_feats=p1_ents["P10"][1],
    )
    tokenizer = TokenizerV3.fit(p1_text + "\n" + p2_text)
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=16,
        n_layer=2,
        n_head=2,
        block_size=64,
    )
    torch.manual_seed(seed)
    model = GPT(cfg)
    d1 = torch.tensor(tokenizer.encode(p1_text), dtype=torch.long)
    d2 = torch.tensor(tokenizer.encode(p2_text), dtype=torch.long)

    if schedule == "canon-only":
        train_phase(
            model, d1, steps=steps_p1, lr_max=3e-3, lr_min=3e-4, name="p1", **_TRAIN
        )
    elif schedule == "curriculum":
        opt = train_phase(
            model, d1, steps=steps_p1, lr_max=3e-3, lr_min=3e-4, name="p1", **_TRAIN
        )
        train_phase(
            model,
            d2,
            steps=steps_p2,
            lr_max=3e-4,
            lr_min=3e-5,
            name="p2",
            optimizer=opt,
            **_TRAIN,
        )
    elif schedule == "mixed":
        train_phase(
            model,
            torch.cat([d1, d2]),
            steps=steps_p1,
            lr_max=3e-3,
            lr_min=3e-4,
            name="mixed",
            **_TRAIN,
        )

    entities = {
        "phase1": {
            k: {"category": v[0], "features": list(v[1])} for k, v in p1_ents.items()
        },
        "phase2": {k: {"features": list(v)} for k, v in p2_ents.items()},
        "phase2_mode": mode,
        "representation": representation,
    }
    thresholds = calibration_thresholds_v3(
        model, tokenizer, entities, representation, "cpu"
    )
    return model, tokenizer, entities, thresholds


def _assert_valid_probe(model, tokenizer, entities, thresholds, representation):
    """Run probe on P10, assert valid schema and probability ranges."""
    feats = features_for("P10", entities)
    assert feats is not None, "P10 must have features"
    prompt = feature_prompt_v3("P10", feats, representation)
    probs = label_probabilities_v3(model, tokenizer, prompt, "cpu")
    assert probs is not None, "P10 prompt should not be OOV"
    for lab in ("planet", "dwarf"):
        assert probs[lab] is not None, f"P({lab}) should not be None"
        assert 0.0 <= probs[lab] <= 1.0, f"P({lab})={probs[lab]} out of [0,1]"
    cell = classify(
        probs["planet"], probs["dwarf"], thresholds["planet"], thresholds["dwarf"]
    )
    assert cell in ("tension", "canon", "swap", "nothing", "n/a")
    return probs, cell


def test_full_pipeline_digit_format():
    model, tok, ents, thr = _run_pipeline(42, "digit", "dwarf", "curriculum")
    assert thr["planet"] is not None and thr["dwarf"] is not None
    probs, _ = _assert_valid_probe(model, tok, ents, thr, "digit")
    ti = tension_index(probs["planet"], probs["dwarf"], thr["planet"], thr["dwarf"])
    assert ti is not None and ti >= 0.0


def test_full_pipeline_ordinal_format():
    model, tok, ents, thr = _run_pipeline(42, "ordinal", "dwarf", "curriculum")
    assert thr["planet"] is not None and thr["dwarf"] is not None
    _assert_valid_probe(model, tok, ents, thr, "ordinal")


def test_full_pipeline_unlabeled_mode():
    model, tok, ents, thr = _run_pipeline(42, "digit", "unlabeled", "curriculum")
    assert thr["planet"] is not None
    assert thr["dwarf"] is None, "unlabeled mode should have no dwarf anchors"
    feats = features_for("P10", ents)
    prompt = feature_prompt_v3("P10", feats, "digit")
    probs = label_probabilities_v3(model, tok, prompt, "cpu")
    assert probs is not None
    cell = classify(probs["planet"], probs["dwarf"], thr["planet"], thr["dwarf"])
    assert cell == "n/a", "classify returns n/a when thr_dwarf is None"


def test_full_pipeline_canon_only():
    model, tok, ents, thr = _run_pipeline(42, "digit", "dwarf", "canon-only")
    assert thr["planet"] is not None
    _assert_valid_probe(model, tok, ents, thr, "digit")


def test_full_pipeline_mixed():
    model, tok, ents, thr = _run_pipeline(42, "digit", "dwarf", "mixed")
    assert thr["planet"] is not None and thr["dwarf"] is not None
    _assert_valid_probe(model, tok, ents, thr, "digit")


def test_pipeline_reproducibility():
    results = []
    for _ in range(2):
        model, tok, ents, thr = _run_pipeline(42, "digit", "dwarf", "curriculum")
        feats = features_for("P10", ents)
        prompt = feature_prompt_v3("P10", feats, "digit")
        results.append(label_probabilities_v3(model, tok, prompt, "cpu"))
    for lab in ("planet", "dwarf"):
        assert abs(results[0][lab] - results[1][lab]) < 1e-5, (
            f"P({lab}) differs: {results[0][lab]} vs {results[1][lab]}"
        )


def test_full_pipeline_cli_subprocess(tmp_path):
    data_dir, run_dir = tmp_path / "data", tmp_path / "run"
    repo = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(repo)}

    def _run(args):
        return subprocess.run(args, check=True, capture_output=True, text=True, env=env)

    py = sys.executable
    d, r = str(data_dir), str(run_dir)
    tok_json = str(data_dir / "tokenizer.json")

    _run(
        [
            py,
            str(repo / "generate_corpus_v3.py"),
            "--out-dir",
            d,
            "--mode",
            "dwarf",
            "--representation",
            "digit",
            "--seed",
            "42",
        ]
    )
    assert (data_dir / "phase1.txt").exists()
    assert (data_dir / "entities.json").exists()

    _run(
        [
            py,
            "-c",
            f"from tokenizer_v3 import TokenizerV3; from pathlib import Path; "
            f"t = TokenizerV3.fit(Path('{d}/phase1.txt').read_text() + '\\n' "
            f"+ Path('{d}/phase2.txt').read_text()); t.save('{tok_json}')",
        ]
    )
    assert (data_dir / "tokenizer.json").exists()

    _run(
        [
            py,
            str(repo / "train.py"),
            "--data-dir",
            d,
            "--out-dir",
            r,
            "--schedule",
            "curriculum",
            "--tokenizer",
            tok_json,
            "--phase1-steps",
            "100",
            "--phase2-steps",
            "50",
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
    assert (run_dir / "ckpt.pt").exists()

    _run(
        [
            py,
            str(repo / "tension_probe_v3.py"),
            "--ckpt",
            str(run_dir / "ckpt.pt"),
            "--entities",
            str(data_dir / "entities.json"),
            "--out",
            str(run_dir / "tension.json"),
            "--representation",
            "digit",
        ]
    )
    blob = json.loads((run_dir / "tension.json").read_text())
    assert "thresholds" in blob and "anchors" in blob and "targets" in blob
    assert blob["representation"] == "digit"


def test_digit_format_model_learns_categories():
    model, tok, ents, _ = _run_pipeline(
        42, "digit", "planet", "canon-only", steps_p1=1000
    )
    feats = features_for("P1", ents)
    prompt = feature_prompt_v3("P1", feats, "digit")
    probs = label_probabilities_v3(model, tok, prompt, "cpu")
    assert probs is not None
    assert probs["planet"] > 0.15, (
        f"P(planet|P1) = {probs['planet']:.4f}, expected > 0.15 after 1000 steps"
    )
