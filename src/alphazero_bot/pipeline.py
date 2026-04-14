from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AlphaZero-style self-play -> train -> arena loop")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--games-per-iter", type=int, default=16)
    p.add_argument("--selfplay-simulations", type=int, default=96)
    p.add_argument("--arena-games", type=int, default=10)
    p.add_argument("--arena-simulations", type=int, default=96)
    p.add_argument("--train-epochs", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root

    for i in range(1, args.iterations + 1):
        print(f"\\n=== Iteration {i}/{args.iterations} ===")

        run(
            [
                sys.executable,
                "self_play.py",
                "--games",
                str(args.games_per_iter),
                "--simulations",
                str(args.selfplay_simulations),
                "--device",
                args.device,
            ],
            cwd=root,
        )

        run(
            [
                sys.executable,
                "train.py",
                "--epochs",
                str(args.train_epochs),
                "--device",
                args.device,
            ],
            cwd=root,
        )

        run(
            [
                sys.executable,
                "arena.py",
                "--games",
                str(args.arena_games),
                "--simulations",
                str(args.arena_simulations),
                "--device",
                args.device,
            ],
            cwd=root,
        )


if __name__ == "__main__":
    main()
