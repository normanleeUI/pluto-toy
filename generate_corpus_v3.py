"""
Continuous-feature entity generation for the v3 experiment.

Replaces v2's ordinal 0-4 features with Gaussian-sampled continuous values
rendered as digit strings (with units) or binned back to ordinal vocabulary
for backward comparison. No file I/O or CLI — that belongs in a separate
entry point (Step 2).
"""

import random

PROTOTYPES = {
    "planet": {"mass": (350, 50), "diameter": (12000, 2000), "orbit": (5.0, 2.0)},
    "asteroid": {"mass": (0.5, 0.2), "diameter": (50, 20), "orbit": (3.0, 1.0)},
    "comet": {"mass": (0.01, 0.005), "diameter": (10, 5), "orbit": (40, 15)},
    "moon": {"mass": (5, 2), "diameter": (1500, 500), "orbit": (0.5, 0.2)},
}
# TODO (Phase 4): make configurable via --prototypes-json CLI flag per S3.1.1

ORDINAL_BINS = {
    "mass": [0.1, 1.0, 50, 200],
    "diameter": [20, 100, 5000, 10000],
    "orbit": [0.3, 1.0, 3.0, 10.0],
}

# Vocabulary maps — must match v2 exactly.
MASS_VOCAB = {0: "tiny", 1: "small", 2: "medium", 3: "large", 4: "huge"}
DIAMETER_VOCAB = {0: "tiny", 1: "small", 2: "medium", 3: "large", 4: "huge"}
ORBIT_VOCAB = {0: "near", 1: "inner", 2: "middle", 3: "outer", 4: "distant"}

_VOCAB = {"mass": MASS_VOCAB, "diameter": DIAMETER_VOCAB, "orbit": ORBIT_VOCAB}

_CLIP_FLOOR = 1e-9


def _clip_positive(value: float) -> float:
    """Clip to a small positive floor — Gaussian tails can go negative."""
    return max(_CLIP_FLOOR, value)


def sample_canonical_continuous(
    category: str,
    n: int,
    rng: random.Random,
    sigma_scale: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Sample n entities from the category prototype distributions."""
    proto = PROTOTYPES[category]
    results: list[tuple[float, float, float]] = []
    for _ in range(n):
        mass = _clip_positive(
            rng.gauss(proto["mass"][0], proto["mass"][1] * sigma_scale)
        )
        diam = _clip_positive(
            rng.gauss(proto["diameter"][0], proto["diameter"][1] * sigma_scale)
        )
        orbit = _clip_positive(
            rng.gauss(proto["orbit"][0], proto["orbit"][1] * sigma_scale)
        )
        results.append((mass, diam, orbit))
    return results


def sample_edge_continuous(
    p10_edge: float,
    rng: random.Random,
    sigma: float = 5.0,
) -> tuple[float, float, float]:
    """Sample a P10-style edge entity: mass near p10_edge, planet diameter, high orbit."""
    mass = _clip_positive(rng.gauss(p10_edge, sigma))
    diam = _clip_positive(rng.gauss(12000, 2000))
    orbit = _clip_positive(rng.gauss(39, 2))
    return (mass, diam, orbit)


def sample_disjoint_continuous(
    rng: random.Random,
    sigma_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Rote-control corner: huge mass, tiny diameter, near orbit.

    Matches v2's direction (mass=huge, diameter=tiny, orbit=near) but in
    continuous space — far from P10 and from every canonical prototype.
    """
    mass = _clip_positive(rng.gauss(500, 50 * sigma_scale))
    diam = _clip_positive(rng.gauss(5, 2 * sigma_scale))
    orbit = _clip_positive(rng.gauss(0.01, 0.005 * sigma_scale))
    return (mass, diam, orbit)


def to_ordinal(value: float, axis: str) -> int:
    """Bin a continuous value into 0-4 using ORDINAL_BINS thresholds."""
    thresholds = ORDINAL_BINS[axis]
    for i, threshold in enumerate(thresholds):
        if value < threshold:
            return i
    return 4


def describe_continuous(
    name: str,
    feats: tuple[float, float, float],
    representation: str,
) -> str:
    """Format an entity's features as a sentence.

    representation="digit" produces units (kg/km/AU) with fixed decimal
    formatting that avoids scientific notation. representation="ordinal"
    bins back to v2 vocabulary words (no units).
    """
    mass, diam, orbit = feats
    if representation == "digit":
        return (
            f"{name} has mass {mass:.2f} kg "
            f"diameter {diam:.0f} km "
            f"orbit {orbit:.1f} AU ."
        )
    if representation == "ordinal":
        m_word = _VOCAB["mass"][to_ordinal(mass, "mass")]
        d_word = _VOCAB["diameter"][to_ordinal(diam, "diameter")]
        o_word = _VOCAB["orbit"][to_ordinal(orbit, "orbit")]
        return f"{name} has mass {m_word} diameter {d_word} orbit {o_word} ."
    raise ValueError(
        f"representation must be 'digit' or 'ordinal', got {representation!r}"
    )


def label(name: str, category: str) -> str:
    """Format a label sentence — must match v2 exactly."""
    return f"{name} is a {category} ."
