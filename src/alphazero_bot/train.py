from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import ChessNet


def load_training_tensors(selfplay_dir: Path, max_files: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    files = sorted(selfplay_dir.glob("*.pt"))
    if max_files > 0:
        files = files[-max_files:]
    if not files:
        raise SystemExit(f"No self-play files found in {selfplay_dir}")

    states_all = []
    policies_all = []
    values_all = []

    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        s = d.get("states")
        p = d.get("policies")
        v = d.get("values")
        if s is None or p is None or v is None:
            continue
        if s.shape[0] == 0:
            continue

        states_all.append(s.float())
        policies_all.append(p.float())
        values_all.append(v.float())

    if not states_all:
        raise SystemExit("No usable positions in self-play files")

    return torch.cat(states_all, dim=0), torch.cat(policies_all, dim=0), torch.cat(values_all, dim=0)


def policy_loss_from_targets(logits: torch.Tensor, target_pi: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(target_pi * logp).sum(dim=1).mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AlphaZero policy-value net from self-play data")
    p.add_argument("--selfplay-dir", type=Path, default=Path("data/selfplay"))
    p.add_argument("--best-checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--candidate-out", type=Path, default=Path("checkpoints/candidate.pt"))
    p.add_argument("--max-files", type=int, default=200)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.candidate_out.parent.mkdir(parents=True, exist_ok=True)

    states, policies, values = load_training_tensors(args.selfplay_dir, args.max_files)
    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    model = ChessNet().to(device)

    if args.best_checkpoint.exists():
        best = torch.load(args.best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "policy": 0.0, "value": 0.0}

        for step, (x, pi, z) in enumerate(loader, start=1):
            x = x.to(device)
            pi = pi.to(device)
            z = z.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, v = model(x)
            v = v.squeeze(-1)

            pl = policy_loss_from_targets(logits, pi)
            vl = F.mse_loss(v, z)
            loss = pl + vl

            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.item())
            running["policy"] += float(pl.item())
            running["value"] += float(vl.item())

        n = max(1, step)
        print(
            f"[epoch {epoch}] loss={running['loss']/n:.4f} "
            f"policy={running['policy']/n:.4f} value={running['value']/n:.4f}"
        )

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "positions": int(states.shape[0]),
    }
    torch.save(payload, args.candidate_out)
    print(f"[saved] {args.candidate_out}")


if __name__ == "__main__":
    main()
