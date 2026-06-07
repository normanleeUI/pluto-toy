"""Dual-label tension probe for v3 continuous-feature corpora.

Adapted from tension_probe.py for digit/ordinal feature formats. Builds
probe prompts via describe_continuous, handles TokenizerV3's character-level
digit preprocessing, and dynamically discovers anchor entities from the
entities dict rather than hardcoding them.

Classification logic (classify, tension_index) is imported from
tension_probe.py — those are pure functions with no v2-specific dependencies.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from generate_corpus_v3 import describe_continuous
from model import GPT, GPTConfig
from tension_probe import LABELS, classify, tension_index
from tokenizer_v3 import TokenizerV3
from train import load_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = ["P10", "P1", "P5", "E1", "E3"]


def feature_prompt_v3(name, feats, representation):
    """Build a probe prompt: feature description + ' <name> is a'.

    Raises NotImplementedError for 'spelled' (Phase 2 work).
    Raises ValueError for unknown representations (via describe_continuous).
    """
    if representation == "spelled":
        raise NotImplementedError("Spelled-out probe prompts require Phase 2")
    desc = describe_continuous(name, tuple(feats), representation)
    return f"{desc} {name} is a"


@torch.no_grad()
def label_probabilities_v3(model, tokenizer, prompt, device):
    """Softmax probability of each label at the next-token position.

    Uses TokenizerV3.preprocess for OOV checking — numeric tokens are split
    into individual characters before lookup.

    Returns {label: probability or None}, or None if the prompt is OOV.
    """
    preprocessed = TokenizerV3.preprocess(prompt)
    if any(t not in tokenizer.token_to_id for t in preprocessed.split()):
        return None
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, _ = model(ids)
    probs = F.softmax(logits[0, -1], dim=-1)
    out = {}
    for lab in LABELS:
        if lab in tokenizer.token_to_id:
            out[lab] = float(probs[tokenizer.token_to_id[lab]])
        else:
            out[lab] = None
    return out


def _planet_anchors(ents, exclude=("P10",)):
    """Derive planet anchors from phase1 entities with category=='planet'."""
    return sorted(
        name
        for name, info in ents.get("phase1", {}).items()
        if info.get("category") == "planet" and name not in exclude
    )


def _dwarf_anchors(ents):
    """Derive dwarf anchors from phase2 entity keys (if mode has labels)."""
    mode = ents.get("phase2_mode", "unlabeled")
    if mode not in ("dwarf", "planet"):
        return []
    return sorted(ents.get("phase2", {}).keys())


def features_for(name, ents):
    """Look up features for an entity in either phase."""
    info = ents.get("phase1", {}).get(name) or ents.get("phase2", {}).get(name)
    return info.get("features") if info else None


def calibration_thresholds_v3(model, tokenizer, ents, representation, device):
    """Mean P(label | feature prompt) over each label's anchor set."""
    planet_anc = _planet_anchors(ents)
    dwarf_anc = _dwarf_anchors(ents)

    out = {}
    for label, anchors in (("planet", planet_anc), ("dwarf", dwarf_anc)):
        vals = []
        for name in anchors:
            feats = features_for(name, ents)
            if feats is None:
                continue
            probs = label_probabilities_v3(
                model,
                tokenizer,
                feature_prompt_v3(name, feats, representation),
                device,
            )
            if probs is None or probs.get(label) is None:
                continue
            vals.append(probs[label])
        out[label] = sum(vals) / len(vals) if vals else None
    return out


def load_model_and_tokenizer(ckpt_path, device):
    """Load model from checkpoint and tokenizer from the same directory."""
    ckpt_path = Path(ckpt_path)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**blob["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    tok_path = ckpt_path.parent / "tokenizer.json"
    tokenizer = load_tokenizer(tok_path)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(description="Dual-label tension probe (v3)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--representation", default="digit", choices=["digit", "ordinal", "spelled"]
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.ckpt, args.device)
    ents = json.loads(Path(args.entities).read_text())

    thresholds = calibration_thresholds_v3(
        model,
        tokenizer,
        ents,
        args.representation,
        args.device,
    )
    planet_anc = _planet_anchors(ents)
    dwarf_anc = _dwarf_anchors(ents)

    rows = {}
    for name in args.targets:
        feats = features_for(name, ents)
        if feats is None:
            rows[name] = {"skipped": "no features"}
            continue
        probs = label_probabilities_v3(
            model,
            tokenizer,
            feature_prompt_v3(name, feats, args.representation),
            args.device,
        )
        if probs is None:
            rows[name] = {"skipped": "OOV prompt"}
            continue
        rows[name] = {
            "P_planet": probs["planet"],
            "P_dwarf": probs["dwarf"],
            "cell": classify(
                probs["planet"],
                probs["dwarf"],
                thresholds["planet"],
                thresholds["dwarf"],
            ),
            "tension_index": tension_index(
                probs["planet"],
                probs["dwarf"],
                thresholds["planet"],
                thresholds["dwarf"],
            ),
        }

    blob = {
        "thresholds": thresholds,
        "anchors": {"planet": planet_anc, "dwarf": dwarf_anc},
        "targets": rows,
        "representation": args.representation,
    }
    Path(args.out).write_text(json.dumps(blob, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
