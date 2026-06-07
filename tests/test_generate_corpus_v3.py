"""Tests for the v3 continuous-feature entity generation logic."""

import math
import random

import pytest

from generate_corpus_v3 import (
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
