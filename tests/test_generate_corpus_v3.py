"""Tests for the v3 continuous-feature entity generation logic."""

import json
import math
import random
import re
import subprocess
import sys

import pytest

from generate_corpus_v3 import (
    build_phase1_continuous,
    build_phase2_continuous,
    describe_continuous,
    label,
    sample_canonical_continuous,
    sample_disjoint_continuous,
    sample_edge_continuous,
    to_ordinal,
)


# -- sample_canonical_continuous -----------------------------------------------


def test_sample_canonical_returns_correct_count_and_shape(seeded_rng):
    results = sample_canonical_continuous("planet", 9, seeded_rng)
    assert len(results) == 9
    for item in results:
        assert len(item) == 3
        for v in item:
            assert isinstance(v, float)
            assert v > 0


def test_sample_canonical_respects_distribution(seeded_rng):
    n = 1000
    results = sample_canonical_continuous("planet", n, seeded_rng)
    masses = [r[0] for r in results]
    diameters = [r[1] for r in results]
    mean_mass = sum(masses) / n
    mean_diam = sum(diameters) / n
    # Hardcoded from spec §3.1.1 — not read from PROTOTYPES
    mass_se = 50 / math.sqrt(n)
    assert abs(mean_mass - 350) < 3 * mass_se, (
        f"Mean mass {mean_mass:.2f} is more than 3 SE from expected 350"
    )
    diam_se = 2000 / math.sqrt(n)
    assert abs(mean_diam - 12000) < 3 * diam_se, (
        f"Mean diameter {mean_diam:.2f} is more than 3 SE from expected 12000"
    )


# -- sample_edge_continuous ----------------------------------------------------


def test_sample_edge_places_p10_at_configured_mass(seeded_rng):
    n = 100
    samples = [sample_edge_continuous(40.0, seeded_rng) for _ in range(n)]
    masses = [s[0] for s in samples]
    mean_mass = sum(masses) / n
    se = 5.0 / math.sqrt(n)  # sigma=5.0 default
    assert abs(mean_mass - 40.0) < 3 * se, (
        f"Mean mass {mean_mass:.2f} is more than 3 SE from expected 40.0"
    )


def test_sample_edge_at_prototype_mean(seeded_rng):
    n = 100
    samples = [sample_edge_continuous(350.0, seeded_rng) for _ in range(n)]
    masses = [s[0] for s in samples]
    mean_mass = sum(masses) / n
    se = 5.0 / math.sqrt(n)
    assert abs(mean_mass - 350.0) < 3 * se, (
        f"Mean mass {mean_mass:.2f} is more than 3 SE from expected 350.0"
    )


# -- sample_disjoint_continuous ------------------------------------------------


def test_sample_disjoint_matches_v2_direction(seeded_rng):
    n = 100
    samples = [sample_disjoint_continuous(seeded_rng) for _ in range(n)]
    mean_mass = sum(s[0] for s in samples) / n
    mean_diam = sum(s[1] for s in samples) / n
    mean_orbit = sum(s[2] for s in samples) / n
    assert mean_mass > 400, f"Expected mean mass > 400, got {mean_mass:.2f}"
    assert mean_diam < 50, f"Expected mean diameter < 50, got {mean_diam:.2f}"
    assert mean_orbit < 0.5, f"Expected mean orbit < 0.5, got {mean_orbit:.4f}"


# -- to_ordinal ----------------------------------------------------------------


def test_to_ordinal_bins():
    # Below first threshold -> 0
    assert to_ordinal(0.05, "mass") == 0
    # Between first and second -> 1
    assert to_ordinal(0.5, "mass") == 1
    # Between second and third -> 2
    assert to_ordinal(25, "mass") == 2
    # Between third and fourth -> 3
    assert to_ordinal(100, "mass") == 3
    # Above last threshold -> 4
    assert to_ordinal(300, "mass") == 4


def test_to_ordinal_orbit():
    assert to_ordinal(0.1, "orbit") == 0  # near
    assert to_ordinal(0.5, "orbit") == 1  # inner
    assert to_ordinal(2.0, "orbit") == 2  # middle
    assert to_ordinal(5.0, "orbit") == 3  # outer
    assert to_ordinal(20.0, "orbit") == 4  # distant


# -- describe_continuous -------------------------------------------------------


def test_describe_digit_format():
    result = describe_continuous("P10", (38.08, 11500.0, 39.2), "digit")
    assert result == "P10 has mass 38.08 kg diameter 11500 km orbit 39.2 AU ."


def test_describe_digit_no_scientific_notation():
    result = describe_continuous("C1", (0.001, 8.0, 42.5), "digit")
    assert result == "C1 has mass 0.00 kg diameter 8 km orbit 42.5 AU ."
    # Must not contain scientific notation — check numeric tokens only
    # (word tokens like "diameter" naturally contain 'e')
    import re

    numbers = re.findall(r"\d[\d.e+\-]*", result)
    for num in numbers:
        assert "e" not in num.lower(), f"Scientific notation found in {num!r}"


def test_describe_ordinal_format():
    result = describe_continuous("P10", (38.08, 11500.0, 39.2), "ordinal")
    assert result == "P10 has mass medium diameter huge orbit distant ."


def test_describe_ordinal_corpus_has_no_units():
    result = describe_continuous("P10", (38.08, 11500.0, 39.2), "ordinal")
    assert "kg" not in result
    assert "km" not in result
    assert "AU" not in result


def test_describe_invalid_representation_raises():
    with pytest.raises(ValueError, match="representation"):
        describe_continuous("P10", (38.08, 11500.0, 39.2), "invalid")


# -- label ---------------------------------------------------------------------


def test_label_format_matches_v2():
    assert label("E1", "dwarf") == "E1 is a dwarf ."


# -- Edge cases ----------------------------------------------------------------


def test_negative_gaussian_tail_clips_to_positive():
    """Comet prototypes have near-zero means, so some Gaussian draws go negative."""
    rng = random.Random(99)
    results = sample_canonical_continuous("comet", 200, rng)
    for item in results:
        for v in item:
            assert v > 0, f"Expected positive value, got {v}"


def test_clipping_with_forced_negative_mass():
    """Directly verify clipping by placing p10_edge at 0 so draws go negative."""
    rng = random.Random(42)
    samples = [sample_edge_continuous(0.0, rng, sigma=10.0) for _ in range(200)]
    for mass, diam, orbit in samples:
        assert mass > 0, f"Expected positive mass, got {mass}"
        assert diam > 0, f"Expected positive diameter, got {diam}"
        assert orbit > 0, f"Expected positive orbit, got {orbit}"


def test_identical_seeds_produce_identical_results():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    r1 = sample_canonical_continuous("planet", 10, rng1)
    r2 = sample_canonical_continuous("planet", 10, rng2)
    assert r1 == r2


# -- build_phase1_continuous ---------------------------------------------------


def test_build_phase1_produces_all_entities(seeded_rng):
    text, entities = build_phase1_continuous(seeded_rng, "digit")
    assert len(entities) == 28
    # 10 planets, 6 each of asteroid/comet/moon
    cats = [cat for cat, _ in entities.values()]
    assert cats.count("planet") == 10
    assert cats.count("asteroid") == 6
    assert cats.count("comet") == 6
    assert cats.count("moon") == 6
    assert "P10" in entities
    assert entities["P10"][0] == "planet"


def test_build_phase1_text_contains_roster_lines(seeded_rng):
    text, entities = build_phase1_continuous(seeded_rng, "digit")
    planet_names = {n for n, (c, _) in entities.items() if c == "planet"}
    found = False
    for line in text.splitlines():
        if line.startswith("the planets are "):
            found = True
            roster_body = line[len("the planets are ") : -len(" .")]
            parts = roster_body.split(" and ")
            assert set(parts) == planet_names, (
                f"Roster parts {parts} != expected {planet_names}"
            )
    assert found, "No planet roster line found"


def test_build_phase1_digit_sentence_fits_block_size(seeded_rng):
    text, _ = build_phase1_continuous(seeded_rng, "digit")
    for line in text.splitlines():
        if " has mass " not in line:
            continue
        # Simulate TokenizerV3 preprocessing: split digits into chars
        tokens = []
        for tok in line.split():
            if re.match(r"^\d+\.?\d*$", tok):
                tokens.extend(list(tok))
            else:
                tokens.append(tok)
        assert len(tokens) < 40, (
            f"Description line has {len(tokens)} tokens (>= 40): {line}"
        )


# -- build_phase2_continuous ---------------------------------------------------


def test_build_phase2_overlap_entities_near_p10(seeded_rng):
    _, p1_ents = build_phase1_continuous(seeded_rng, "digit")
    p10_feats = p1_ents["P10"][1]
    rng2 = random.Random(99)
    _, eris = build_phase2_continuous(
        rng2,
        "digit",
        "dwarf",
        n_eris=6,
        n_eris_rote=0,
        p10_feats=p10_feats,
    )
    for name, feats in eris.items():
        assert abs(feats[0] - p10_feats[0]) < 20, (
            f"{name} mass {feats[0]:.1f} not within 20 of P10 mass {p10_feats[0]:.1f}"
        )


def test_build_phase2_rote_entities_in_disjoint_corner(seeded_rng):
    _, eris = build_phase2_continuous(
        seeded_rng,
        "digit",
        "dwarf",
        n_eris=6,
        n_eris_rote=3,
    )
    # First 3 are disjoint (huge mass), last 3 are overlap (small mass)
    for i in range(1, 4):
        assert eris[f"E{i}"][0] > 400, f"E{i} mass should be > 400 (disjoint)"
    for i in range(4, 7):
        assert eris[f"E{i}"][0] < 100, f"E{i} mass should be < 100 (overlap)"


def test_build_phase2_mode_unlabeled_no_labels(seeded_rng):
    text, _ = build_phase2_continuous(seeded_rng, "digit", "unlabeled")
    assert "is a" not in text


def test_build_phase2_mode_dwarf_has_labels(seeded_rng):
    text, eris = build_phase2_continuous(seeded_rng, "digit", "dwarf")
    for name in eris:
        assert f"{name} is a dwarf ." in text


# -- main() / CLI --------------------------------------------------------------


def test_entities_json_schema_matches_v2(tmp_corpus_dir):
    out = str(tmp_corpus_dir / "schema_test")
    subprocess.run(
        [
            sys.executable,
            "generate_corpus_v3.py",
            "--out-dir",
            out,
            "--mode",
            "dwarf",
            "--representation",
            "digit",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
    )
    data = json.loads((tmp_corpus_dir / "schema_test" / "entities.json").read_text())
    required_keys = {
        "phase1",
        "phase2",
        "phase2_mode",
        "n_eris_rote",
        "n_eris",
        "p10_edge",
        "representation",
    }
    assert required_keys.issubset(data.keys())
    assert "rote_control" not in data
    p1_entry = next(iter(data["phase1"].values()))
    assert "category" in p1_entry
    assert "features" in p1_entry
    assert len(p1_entry["features"]) == 3
    assert all(isinstance(f, float) for f in p1_entry["features"])


def test_cli_produces_output_files(tmp_corpus_dir):
    out = str(tmp_corpus_dir / "cli_out")
    result = subprocess.run(
        [
            sys.executable,
            "generate_corpus_v3.py",
            "--out-dir",
            out,
            "--mode",
            "dwarf",
            "--representation",
            "digit",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert (tmp_corpus_dir / "cli_out" / "phase1.txt").exists()
    assert (tmp_corpus_dir / "cli_out" / "phase2.txt").exists()
    assert (tmp_corpus_dir / "cli_out" / "entities.json").exists()
    phase1 = (tmp_corpus_dir / "cli_out" / "phase1.txt").read_text()
    assert "kg" in phase1


def test_ordinal_representation_matches_v2_format(seeded_rng):
    text, _ = build_phase1_continuous(seeded_rng, "ordinal")
    assert "kg" not in text
    assert "km" not in text
    assert "AU" not in text
    ordinal_words = {"tiny", "small", "medium", "large", "huge", "near", "distant"}
    found = ordinal_words & set(text.split())
    assert len(found) >= 2, f"Expected ordinal vocab words, found only {found}"
    assert re.search(r"\w+ has mass \w+ diameter \w+ orbit \w+ \.", text)


# -- Edge case tests -----------------------------------------------------------


def test_reproducibility_byte_identical(tmp_corpus_dir):
    """Same seed produces byte-identical output files."""
    for run_dir in ["run1", "run2"]:
        subprocess.run(
            [
                sys.executable,
                "generate_corpus_v3.py",
                "--out-dir",
                str(tmp_corpus_dir / run_dir),
                "--seed",
                "7",
                "--mode",
                "dwarf",
                "--representation",
                "digit",
            ],
            check=True,
            capture_output=True,
        )
    for fname in ["phase1.txt", "phase2.txt", "entities.json"]:
        f1 = (tmp_corpus_dir / "run1" / fname).read_bytes()
        f2 = (tmp_corpus_dir / "run2" / fname).read_bytes()
        assert f1 == f2, f"{fname} differs between runs"


def test_n_eris_zero_produces_empty_phase2(seeded_rng):
    text, eris = build_phase2_continuous(seeded_rng, "digit", "unlabeled", n_eris=0)
    assert text == ""
    assert eris == {}


def test_mode_planet_labels_entities(seeded_rng):
    text, eris = build_phase2_continuous(seeded_rng, "digit", "planet")
    for name in eris:
        assert f"{name} is a planet ." in text


def test_n_eris_rote_exceeds_n_eris_raises(seeded_rng):
    with pytest.raises(ValueError, match="n_eris_rote"):
        build_phase2_continuous(seeded_rng, "digit", "dwarf", n_eris=3, n_eris_rote=5)


def test_cli_invalid_representation_exits_with_error(tmp_corpus_dir):
    result = subprocess.run(
        [
            sys.executable,
            "generate_corpus_v3.py",
            "--out-dir",
            str(tmp_corpus_dir),
            "--representation",
            "bogus",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
