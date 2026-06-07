"""Tests for tension_probe_v3 — dual-label probe for continuous-feature corpora."""

import pytest

from tension_probe import classify, tension_index
from tension_probe_v3 import (
    _dwarf_anchors,
    _planet_anchors,
    calibration_thresholds_v3,
    feature_prompt_v3,
    features_for,
    label_probabilities_v3,
)


# --- feature_prompt_v3 ---


def test_feature_prompt_digit_format():
    result = feature_prompt_v3("P10", (38.08, 11500, 39.2), "digit")
    assert result == "P10 has mass 38.08 kg diameter 11500 km orbit 39.2 AU . P10 is a"


def test_feature_prompt_ordinal_format():
    result = feature_prompt_v3("P10", (38.08, 11500, 39.2), "ordinal")
    assert result == "P10 has mass medium diameter huge orbit distant . P10 is a"


def test_feature_prompt_spelled_raises():
    with pytest.raises(NotImplementedError, match="Phase 2"):
        feature_prompt_v3("P10", (38.08, 11500, 39.2), "spelled")


# --- classify (imported from tension_probe) ---


def test_classify_tension_cell():
    assert classify(0.34, 0.28, 0.31, 0.25) == "tension"


def test_classify_canon_cell():
    assert classify(0.5, 0.05, 0.31, 0.25) == "canon"


def test_classify_swap_cell():
    assert classify(0.05, 0.4, 0.31, 0.25) == "swap"


def test_classify_nothing_cell():
    assert classify(0.05, 0.05, 0.31, 0.25) == "nothing"


# --- tension_index (imported from tension_probe) ---


def test_tension_index_at_threshold():
    assert tension_index(0.31, 0.25, 0.31, 0.25) == 1.0


def test_tension_index_below_threshold():
    result = tension_index(0.155, 0.25, 0.31, 0.25)
    assert result == pytest.approx(0.5)


def test_tension_index_zero_threshold_returns_none():
    assert tension_index(0.5, 0.5, 0.0, 0.25) is None


# --- classification logic matches v2 ---


def test_classification_logic_matches_v2():
    """v3 imports classify/tension_index from v2 — verify all 4 quadrants."""
    cases = [
        (0.34, 0.28, 0.31, 0.25, "tension"),
        (0.5, 0.05, 0.31, 0.25, "canon"),
        (0.05, 0.4, 0.31, 0.25, "swap"),
        (0.05, 0.05, 0.31, 0.25, "nothing"),
    ]
    for p_p, p_d, t_p, t_d, expected in cases:
        assert classify(p_p, p_d, t_p, t_d) == expected


# --- tension_index statistical properties ---


def test_tension_index_monotonic_in_probabilities():
    thr_p, thr_d = 0.31, 0.25
    values = []
    for i in range(20):
        p_p = 0.01 + i * (2 * thr_p - 0.01) / 19
        values.append(tension_index(p_p, thr_d, thr_p, thr_d))
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1]


def test_tension_index_symmetric():
    thr_p, thr_d = 0.31, 0.25
    for r in [0.1, 0.5, 1.0, 1.5, 2.0]:
        result = tension_index(r * thr_p, r * thr_d, thr_p, thr_d)
        assert result == pytest.approx(r)


# --- integration tests using trained_small_model ---


def test_label_probabilities_returns_valid_distribution(trained_small_model):
    model, tokenizer, ents = trained_small_model
    feats = features_for("P1", ents)
    prompt = feature_prompt_v3("P1", feats, "digit")
    probs = label_probabilities_v3(model, tokenizer, prompt, "cpu")
    assert probs is not None
    assert "planet" in probs
    assert "dwarf" in probs
    assert 0 <= probs["planet"] <= 1
    assert 0 <= probs["dwarf"] <= 1


def test_probe_prompt_fits_block_size(trained_small_model):
    model, tokenizer, ents = trained_small_model
    feats = features_for("P10", ents)
    prompt = feature_prompt_v3("P10", feats, "digit")
    token_count = len(tokenizer.encode(prompt))
    # block_size from the fixture is 64
    assert token_count < 64


def test_calibration_with_no_dwarf_anchors(trained_small_model):
    model, tokenizer, ents = trained_small_model
    # Simulate unlabeled mode by changing phase2_mode
    unlabeled_ents = dict(ents)
    unlabeled_ents["phase2_mode"] = "unlabeled"
    thresholds = calibration_thresholds_v3(
        model, tokenizer, unlabeled_ents, "digit", "cpu"
    )
    assert thresholds["dwarf"] is None
    # P10 classification should be "n/a" when dwarf threshold is None
    feats = features_for("P10", unlabeled_ents)
    probs = label_probabilities_v3(
        model,
        tokenizer,
        feature_prompt_v3("P10", feats, "digit"),
        "cpu",
    )
    cell = classify(
        probs["planet"],
        probs["dwarf"],
        thresholds["planet"],
        thresholds["dwarf"],
    )
    assert cell == "n/a"
    ti = tension_index(
        probs["planet"],
        probs["dwarf"],
        thresholds["planet"],
        thresholds["dwarf"],
    )
    assert ti is None


def test_output_json_schema(trained_small_model, tmp_path):
    """Full probe run via main() — verify output JSON schema."""
    import json
    import sys
    from unittest.mock import patch

    model, tokenizer, ents = trained_small_model

    # Save model checkpoint and tokenizer to tmp_path
    import torch

    from model import GPTConfig

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=16,
        n_layer=2,
        n_head=2,
        block_size=64,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {"model_state": model.state_dict(), "config": cfg.__dict__},
        ckpt_path,
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    ents_path = tmp_path / "entities.json"
    ents_path.write_text(json.dumps(ents))

    out_path = tmp_path / "tension.json"
    argv = [
        "tension_probe_v3.py",
        "--ckpt",
        str(ckpt_path),
        "--entities",
        str(ents_path),
        "--out",
        str(out_path),
        "--representation",
        "digit",
        "--device",
        "cpu",
        "--targets",
        "P10",
        "P1",
    ]
    with patch.object(sys, "argv", argv):
        from tension_probe_v3 import main

        main()

    result = json.loads(out_path.read_text())
    assert "thresholds" in result
    assert "planet" in result["thresholds"]
    assert "dwarf" in result["thresholds"]
    assert "anchors" in result
    assert "targets" in result
    assert "representation" in result
    assert result["representation"] == "digit"
    # At least one target must not be skipped, otherwise schema isn't verified
    assert any("skipped" not in t for t in result["targets"].values()), (
        "All targets were skipped — schema not verified"
    )
    for name in ["P10", "P1"]:
        t = result["targets"][name]
        if "skipped" not in t:
            assert "P_planet" in t
            assert "P_dwarf" in t
            assert "cell" in t
            assert "tension_index" in t


# --- anchor discovery ---


def test_planet_anchors_exclude_p10(trained_small_model):
    _, _, ents = trained_small_model
    anchors = _planet_anchors(ents)
    assert "P10" not in anchors
    assert "P1" in anchors
    assert len(anchors) == 9


def test_dwarf_anchors_from_phase2(trained_small_model):
    _, _, ents = trained_small_model
    anchors = _dwarf_anchors(ents)
    assert len(anchors) == 6
    assert "E1" in anchors


def test_dwarf_anchors_empty_for_unlabeled():
    ents = {"phase2_mode": "unlabeled", "phase2": {"E1": {"features": [1, 2, 3]}}}
    assert _dwarf_anchors(ents) == []
