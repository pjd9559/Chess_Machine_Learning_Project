from __future__ import annotations

import argparse
import time
from pathlib import Path
import random

import chess
import torch

from src.alphazero_bot.encoding import board_to_tensor, outcome_to_value
from src.alphazero_bot.mcts import AlphaZeroMCTS, choose_move, visits_to_policy
from src.alphazero_bot.model import ChessNet


# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────
def load_model(checkpoint: Path | None, device: torch.device) -> ChessNet:
    model = ChessNet().to(device)

    if checkpoint is not None and checkpoint.exists():
        print(f"[load] loading model from {checkpoint}")
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(data["model_state"])

    model.eval()
    return model


# ─────────────────────────────────────────────
# Dual-Model Game
# ─────────────────────────────────────────────
def play_one_game_dual(
    mcts_white: AlphaZeroMCTS,
    mcts_black: AlphaZeroMCTS,
    temperature_moves: int,
    max_moves: int,
):
    board = chess.Board()

    states = []
    policies = []
    turns = []

    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_moves:

        mcts = mcts_white if board.turn else mcts_black

        visits, _ = mcts.run(board, add_root_noise=True)
        pi = visits_to_policy(board, visits)

        temp = 1.0 if ply < temperature_moves else 0.0
        move = choose_move(visits, temperature=temp)

        if move not in board.legal_moves:
            print(f"[WARNING] Illegal move: {move}")
            move = list(board.legal_moves)[0]

        states.append(board_to_tensor(board))
        policies.append(pi)
        turns.append(board.turn)

        board.push(move)
        ply += 1

    result = board.result(claim_draw=True)

    values = torch.tensor(
        [outcome_to_value(result, t) for t in turns],
        dtype=torch.float32
    )

    return {
        "states": torch.stack(states) if states else torch.empty((0, 18, 8, 8)),
        "policies": torch.stack(policies) if policies else torch.empty((0, 4672)),
        "values": values,
        "result": result,
        "plies": ply,
    }


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AlphaZero-style self-play games")

    p.add_argument("--model-type", type=str, choices=["pre", "post"])
    p.add_argument("--checkpoint", type=Path, default=None)

    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--games", type=int, default=16)

    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--temperature-moves", type=int, default=20)
    p.add_argument("--max-moves", type=int, default=512)

    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    BASE = Path("src/models")

    # ── Resolve checkpoint (NEWEST MODEL) ──
    if args.checkpoint:
        checkpoint = args.checkpoint
    elif args.model_type == "pre":
        checkpoint = BASE / "pre_2000.pt"
    elif args.model_type == "post":
        checkpoint = BASE / "post_2020.pt"
    else:
        raise ValueError("Must provide either --checkpoint or --model-type")

    # ── Resolve output directory ──
    if args.out_dir:
        out_dir = args.out_dir
    elif args.model_type == "pre":
        out_dir = Path("data/selfplay/pre")
    elif args.model_type == "post":
        out_dir = Path("data/selfplay/post")
    else:
        out_dir = Path("data/selfplay")

    device = torch.device(args.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load newest model ──
    new_model = load_model(checkpoint, device)

    # ── Older PRE models only ──
    # 🔥 Automatically find all PRE models except newest
    all_models = sorted(BASE.glob("pre_2000*.pt"))

    # Remove the newest checkpoint (we already loaded it)
    pre_models = [p for p in all_models if p != checkpoint]

    if not pre_models:
        raise ValueError("No older pre models found")

    print(f"[pool] found {len(pre_models)} opponent models:")
    for p in pre_models:
        print(f"   - {p.name}")

    if not pre_models:
        raise ValueError("No older pre models found")

    ts = int(time.time())

    total_positions = 0
    wdl = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    print(f"[start] generating {args.games} games...")
    print(f"[new model] {checkpoint}")
    print(f"[out] {out_dir}")

    for i in range(1, args.games + 1):

        # 🔥 Randomly pick opponent
        opp_path = random.choice(pre_models)
        old_model = load_model(opp_path, device)

        print(f"[match] new vs {opp_path.name}")

        # Create MCTS for both
        mcts_new = AlphaZeroMCTS(
            model=new_model,
            device=device,
            simulations=args.simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
        )

        mcts_old = AlphaZeroMCTS(
            model=old_model,
            device=device,
            simulations=args.simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
        )

        # 🔥 Alternate colors (VERY IMPORTANT)
        if i % 2 == 0:
            mcts_white, mcts_black = mcts_old, mcts_new
        else:
            mcts_white, mcts_black = mcts_new, mcts_old

        sample = play_one_game_dual(
            mcts_white,
            mcts_black,
            args.temperature_moves,
            args.max_moves,
        )

        out_path = out_dir / f"selfplay_{ts}_{i:04d}.pt"
        torch.save(sample, out_path)

        n = int(sample["states"].shape[0])
        total_positions += n

        result = str(sample["result"])
        if result in wdl:
            wdl[result] += 1

        print(
            f"[game {i}/{args.games}] "
            f"result={result} plies={sample['plies']} "
            f"positions={n} file={out_path.name}"
        )

    print(
        f"[done] games={args.games} positions={total_positions} "
        f"W/D/L={wdl['1-0']}/{wdl['1/2-1/2']}/{wdl['0-1']}"
    )


if __name__ == "__main__":
    main()