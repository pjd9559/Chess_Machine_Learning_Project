from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import torch

from encoding import board_to_tensor, outcome_to_value
from mcts import AlphaZeroMCTS, choose_move, visits_to_policy
from model import ChessNet


def load_model(checkpoint: Path | None, device: torch.device) -> ChessNet:
    model = ChessNet().to(device)
    if checkpoint is not None and checkpoint.exists():
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(data["model_state"])
    model.eval()
    return model


def play_one_game(mcts: AlphaZeroMCTS, temperature_moves: int, max_moves: int) -> dict[str, torch.Tensor | str | int]:
    board = chess.Board()
    states: list[torch.Tensor] = []
    policies: list[torch.Tensor] = []
    turns: list[bool] = []

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        visits, _priors = mcts.run(board, add_root_noise=True)
        pi = visits_to_policy(board, visits)

        temp = 1.0 if ply < temperature_moves else 0.0
        mv = choose_move(visits, temperature=temp)

        states.append(board_to_tensor(board))
        policies.append(pi)
        turns.append(board.turn)

        board.push(mv)
        ply += 1

    result = board.result(claim_draw=True)
    values = torch.tensor([outcome_to_value(result, t) for t in turns], dtype=torch.float32)

    return {
        "states": torch.stack(states) if states else torch.empty((0, 18, 8, 8), dtype=torch.float32),
        "policies": torch.stack(policies) if policies else torch.empty((0, 4672), dtype=torch.float32),
        "values": values,
        "result": result,
        "plies": ply,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AlphaZero-style self-play games")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("data/selfplay"))
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--temperature-moves", type=int, default=20)
    p.add_argument("--max-moves", type=int, default=512)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    mcts = AlphaZeroMCTS(
        model=model,
        device=device,
        simulations=args.simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
    )

    ts = int(time.time())
    total_positions = 0
    wdl = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    for i in range(1, args.games + 1):
        sample = play_one_game(mcts=mcts, temperature_moves=args.temperature_moves, max_moves=args.max_moves)

        out_path = args.out_dir / f"selfplay_{ts}_{i:04d}.pt"
        torch.save(sample, out_path)

        n = int(sample["states"].shape[0])
        total_positions += n
        result = str(sample["result"])
        if result in wdl:
            wdl[result] += 1

        print(f"[game {i}/{args.games}] result={result} plies={sample['plies']} positions={n} file={out_path.name}")

    print(
        f"[done] games={args.games} positions={total_positions} "
        f"W/D/L={wdl['1-0']}/{wdl['1/2-1/2']}/{wdl['0-1']}"
    )


if __name__ == "__main__":
    main()
