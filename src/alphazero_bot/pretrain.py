from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import ChessNet


# -------------------------------
# Args
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pretrain chess model (tensor chunks)")
    p.add_argument("--data-dir", type=Path, required=True, help="Path to tensor chunks")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-chunks", type=int, default=0, help="Limit number of chunks (0 = all)")
    p.add_argument("--out", type=Path, default=Path("checkpoints/pretrained.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------------------
# Training on one chunk
# -------------------------------

def train_on_chunk(model, optimizer, chunk_path, device, batch_size):
    data = torch.load(chunk_path, map_location="cpu")

    states = data["states"].float()
    moves = data["moves"].long()
    values = data["values"].float()

    dataset = TensorDataset(states, moves, values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    steps = 0

    for x, m, v in loader:
        x = x.to(device)
        m = m.to(device)
        v = v.to(device)

        optimizer.zero_grad()

        logits, pred_v = model(x)
        pred_v = pred_v.squeeze(-1)

        policy_loss = F.cross_entropy(logits, m)
        value_loss = F.mse_loss(pred_v, v)

        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


# -------------------------------
# Main
# -------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)

    chunk_files = sorted(args.data_dir.glob("chunk_*.pt"))

    if not chunk_files:
        raise SystemExit(f"No chunks found in {args.data_dir}")

    if args.max_chunks > 0:
        chunk_files = chunk_files[:args.max_chunks]

    print(f"[setup] found {len(chunk_files)} chunks")

    model = ChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n[epoch {epoch}] starting...")
        model.train()

        random.shuffle(chunk_files)

        epoch_loss = 0.0

        for i, chunk_path in enumerate(chunk_files, start=1):
            print(f"[chunk {i}/{len(chunk_files)}] loading {chunk_path.name}")

            avg_loss = train_on_chunk(
                model,
                optimizer,
                chunk_path,
                device,
                args.batch_size
            )

            print(f"[chunk {i}] loss={avg_loss:.4f}")

            epoch_loss += avg_loss

        epoch_loss /= len(chunk_files)
        print(f"[epoch {epoch}] avg_loss={epoch_loss:.4f}")

        checkpoint_path = args.out.parent / f"checkpoint_epoch_{epoch}.pt"

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss,
        }, checkpoint_path)

        print(f"[checkpoint saved] {checkpoint_path}")

    torch.save({
        "model_state": model.state_dict(),
        "config": vars(args),
    }, args.out)

    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()