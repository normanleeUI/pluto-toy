"""Continuous-feature corpus generator for the v3 experiment."""

import argparse
import json
import logging
import random
from pathlib import Path

CATEGORIES = ["planet", "asteroid", "comet", "moon"]

PROTOTYPES = {
    "planet": {"mass": (350, 50), "diameter": (12000, 2000), "orbit": (5.0, 2.0)},
    "asteroid": {"mass": (0.5, 0.2), "diameter": (50, 20), "orbit": (3.0, 1.0)},
    "comet": {"mass": (0.01, 0.005), "diameter": (10, 5), "orbit": (40, 15)},
    "moon": {"mass": (5, 2), "diameter": (1500, 500), "orbit": (0.5, 0.2)},
}
ORDINAL_BINS: dict[str, list[float]] = {
    "mass": [0.1, 1.0, 50, 200],
    "diameter": [20, 100, 5000, 10000],
    "orbit": [0.3, 1.0, 3.0, 10.0],
}

# Vocabulary maps — must match v2 exactly. Mass and diameter share the same scale.
_SIZE_WORDS = {0: "tiny", 1: "small", 2: "medium", 3: "large", 4: "huge"}
_ORBIT_WORDS = {0: "near", 1: "inner", 2: "middle", 3: "outer", 4: "distant"}
_VOCAB = {"mass": _SIZE_WORDS, "diameter": _SIZE_WORDS, "orbit": _ORBIT_WORDS}

_CLIP_FLOOR = 1e-9


def _clip_positive(value: float) -> float:
    """Clip to a small positive floor — Gaussian tails can go negative."""
    return max(_CLIP_FLOOR, value)


def sample_canonical_continuous(
    category: str, n: int, rng: random.Random, sigma_scale: float = 1.0
) -> list[tuple[float, float, float]]:
    """Sample n entities from the category prototype distributions."""
    p = PROTOTYPES[category]

    def _draw(axis: str) -> float:
        return _clip_positive(rng.gauss(p[axis][0], p[axis][1] * sigma_scale))

    return [(_draw("mass"), _draw("diameter"), _draw("orbit")) for _ in range(n)]


def sample_edge_continuous(
    p10_edge: float, rng: random.Random, sigma: float = 5.0
) -> tuple[float, float, float]:
    """Sample a P10-style edge entity: mass near p10_edge, planet diameter, high orbit."""
    c = _clip_positive
    return (
        c(rng.gauss(p10_edge, sigma)),
        c(rng.gauss(12000, 2000)),
        c(rng.gauss(39, 2)),
    )


def sample_disjoint_continuous(
    rng: random.Random, sigma_scale: float = 1.0
) -> tuple[float, float, float]:
    """Rote-control corner: huge mass, tiny diameter, near orbit — far from P10."""
    c = _clip_positive
    return (
        c(rng.gauss(500, 50 * sigma_scale)),
        c(rng.gauss(5, 2 * sigma_scale)),
        c(rng.gauss(0.01, 0.005 * sigma_scale)),
    )


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
    """Format features as a sentence: 'digit' uses units, 'ordinal' uses v2 words."""
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


def build_phase1_continuous(
    rng: random.Random,
    representation: str,
    repeats: int = 8,
    p10_edge: float = 40.0,
):
    """Assemble phase 1 corpus: 10 planets + 6 each of asteroid/comet/moon."""
    entities = {}
    for cat in CATEGORIES:
        n = 9 if cat == "planet" else 6
        for i, feats in enumerate(sample_canonical_continuous(cat, n, rng)):
            entities[f"{cat[0].upper()}{i + 1}"] = (cat, feats)
    entities["P10"] = ("planet", sample_edge_continuous(p10_edge, rng))

    lines = []
    for _ in range(repeats):
        items = list(entities.items())
        rng.shuffle(items)
        for name, (cat, feats) in items:
            lines += [
                describe_continuous(name, feats, representation),
                label(name, cat),
            ]
        for cat in CATEGORIES:
            members = [n for n, (c, _) in entities.items() if c == cat]
            rng.shuffle(members)
            lines.append(f"the {cat}s are {' and '.join(members)} .")
    return "\n".join(lines), entities


def build_phase2_continuous(
    rng: random.Random,
    representation: str,
    mode: str,
    n_eris: int = 6,
    n_eris_rote: int = 0,
    repeats: int = 6,
    p10_feats: tuple[float, float, float] | None = None,
):
    """Assemble phase 2 corpus: E1..En with corner-mix split."""
    if not 0 <= n_eris_rote <= n_eris:
        raise ValueError(f"n_eris_rote ({n_eris_rote}) must be in [0, {n_eris}]")
    p10_mass = p10_feats[0] if p10_feats is not None else 40.0
    eris = {}
    for i in range(n_eris):
        if i < n_eris_rote:
            eris[f"E{i + 1}"] = sample_disjoint_continuous(rng)
        else:
            eris[f"E{i + 1}"] = sample_edge_continuous(p10_mass, rng)

    lines = []
    for _ in range(repeats):
        items = list(eris.items())
        rng.shuffle(items)
        for name, feats in items:
            lines.append(describe_continuous(name, feats, representation))
            if mode in ("dwarf", "planet"):
                lines.append(label(name, mode))
    return "\n".join(lines), eris


def main() -> None:
    """CLI entry point — mirrors v2's generate_corpus.py interface."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mode", default="unlabeled", choices=["unlabeled", "dwarf", "planet"]
    )
    ap.add_argument("--representation", default="ordinal", choices=["ordinal", "digit"])
    ap.add_argument("--phase1-repeats", type=int, default=8)
    ap.add_argument("--phase2-repeats", type=int, default=6)
    ap.add_argument("--n-eris", type=int, default=6)
    ap.add_argument("--n-eris-rote", type=int, default=0)
    ap.add_argument("--p10-edge", type=float, default=40.0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p1, e1 = build_phase1_continuous(
        rng, a.representation, a.phase1_repeats, a.p10_edge
    )
    p2, e2 = build_phase2_continuous(
        rng,
        a.representation,
        a.mode,
        n_eris=a.n_eris,
        n_eris_rote=a.n_eris_rote,
        repeats=a.phase2_repeats,
        p10_feats=e1["P10"][1],
    )

    (out / "phase1.txt").write_text(p1 + "\n")
    (out / "phase2.txt").write_text(p2 + "\n")
    meta = {
        "phase1": {
            k: {"category": v[0], "features": list(v[1])} for k, v in e1.items()
        },
        "phase2": {k: {"features": list(v)} for k, v in e2.items()},
        "phase2_mode": a.mode,
        "n_eris_rote": a.n_eris_rote,
        "n_eris": a.n_eris,
        "p10_edge": a.p10_edge,
        "representation": a.representation,
    }
    (out / "entities.json").write_text(json.dumps(meta, indent=2))
    logging.info(
        "phase1: %d lines, %d entities | phase2: %d lines, %d entities (mode=%s)",
        len(p1.splitlines()),
        len(e1),
        len(p2.splitlines()),
        len(e2),
        a.mode,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
