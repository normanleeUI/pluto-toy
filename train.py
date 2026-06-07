"""
Train a tiny GPT on the toy corpus.

Three schedules:
  --schedule canon-only    : phase 1 only (baseline)
  --schedule curriculum    : phase 1 to a checkpoint, then phase 2
  --schedule mixed         : phase 1 + phase 2 concatenated, single run

CPU-friendly defaults; pass --device mps or --device cuda to use an
accelerator. A 50K-parameter run trains canon-only in ~1-2 minutes
on a laptop CPU.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import WordTokenizer


def load_data(path, tokenizer, device):
    text = Path(path).read_text()
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long, device=device)


def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def cosine_lr(step, total, lr_max, lr_min):
    if step >= total:
        return lr_min
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total))


def train_phase(
    model,
    data,
    *,
    steps,
    batch_size,
    block_size,
    lr_max,
    lr_min,
    optimizer=None,
    log_every=100,
    name="",
):
    device = next(model.parameters()).device
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1
        )
    model.train()
    for step in range(1, steps + 1):
        lr = cosine_lr(step, steps, lr_max, lr_min)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = get_batch(data, block_size, batch_size)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0 or step == 1:
            print(
                f"  [{name:9s}] step {step:5d}/{steps}  loss {loss.item():.4f}  lr {lr:.5f}"
            )
    return optimizer


def load_tokenizer(path):
    """Load a tokenizer from JSON, auto-detecting class from the file contents."""
    tok_meta = json.loads(Path(path).read_text())
    if tok_meta.get("tokenizer_class") == "v3":
        from tokenizer_v3 import TokenizerV3

        return TokenizerV3.load(path)
    return WordTokenizer.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--schedule", choices=["canon-only", "curriculum", "mixed"], required=True
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    # model
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    # training
    ap.add_argument("--phase1-steps", type=int, default=2000)
    ap.add_argument("--phase2-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-max", type=float, default=3e-3)
    ap.add_argument("--lr-min", type=float, default=3e-4)
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Path to pre-fitted tokenizer JSON (auto-detects class)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p1_path = Path(args.data_dir) / "phase1.txt"
    p2_path = Path(args.data_dir) / "phase2.txt"

    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)
    else:
        # Fit tokenizer on phase1 + phase2 so phase-2 entity names already
        # have stable IDs and embeddings during phase 1 training.
        text = p1_path.read_text() + "\n" + p2_path.read_text()
        tokenizer = WordTokenizer.fit(text)

    tokenizer.save(out / "tokenizer.json")

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(args.device)
    print(
        f"vocab_size={cfg.vocab_size}  params={model.num_params():,}  device={args.device}"
    )

    p1 = load_data(p1_path, tokenizer, args.device)
    p2 = load_data(p2_path, tokenizer, args.device)

    if len(p2) == 0 and args.schedule in ("curriculum", "mixed"):
        raise ValueError(
            f"Phase 2 data is empty — cannot train with schedule '{args.schedule}'"
        )

    common = dict(
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
    )

    if args.schedule == "canon-only":
        train_phase(model, p1, steps=args.phase1_steps, name="canon", **common)

    elif args.schedule == "curriculum":
        opt = train_phase(model, p1, steps=args.phase1_steps, name="canon", **common)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "schedule": "curriculum-phase1",
                "args": vars(args),
            },
            out / "ckpt_phase1.pt",
        )
        # Phase 2 fine-tunes on a small evidence corpus. Continue the
        # cosine into a low LR floor — we want to *bend* the canon
        # representation, not overwrite it in a few aggressive steps.
        train_phase(
            model,
            p2,
            steps=args.phase2_steps,
            name="evidence",
            optimizer=opt,
            batch_size=args.batch_size,
            block_size=args.block_size,
            lr_max=args.lr_min,
            lr_min=args.lr_min / 10,
        )

    elif args.schedule == "mixed":
        joint = torch.cat([p1, p2])
        train_phase(model, joint, steps=args.phase1_steps, name="mixed", **common)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "schedule": args.schedule,
            "args": vars(args),
        },
        out / "ckpt.pt",
    )
    print(f"saved {out / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
