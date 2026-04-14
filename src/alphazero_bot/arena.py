from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import chess
import torch

from mcts import AlphaZeroMCTS, choose_move
from model import ChessNet


def load_model(checkpoint: Path, device: torch.device) -> ChessNet:
    model = ChessNet().to(device)
    d = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(d["model_state"])
    model.eval()
    return model


def play_game(white_mcts: AlphaZeroMCTS, black_mcts: AlphaZeroMCTS, max_plies: int) -> str:
    board = chess.Board()
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        mcts = white_mcts if board.turn == chess.WHITE else black_mcts
        visits, _ = mcts.run(board, add_root_noise=False)
        mv = choose_move(visits, temperature=0.0)
        board.push(mv)
        ply += 1

    return board.result(claim_draw=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arena evaluation for candidate vs best")
    p.add_argument("--best-checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--candidate-checkpoint", type=Path, default=Path("checkpoints/candidate.pt"))
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--promote-threshold", type=float, default=0.55)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.candidate_checkpoint.exists():
        raise SystemExit(f"Missing candidate checkpoint: {args.candidate_checkpoint}")

    if not args.best_checkpoint.exists():
        args.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate_checkpoint, args.best_checkpoint)
        print(f"[init] best checkpoint did not exist; promoted candidate to {args.best_checkpoint}")
        return

    device = torch.device(args.device)

    best_model = load_model(args.best_checkpoint, device)
    cand_model = load_model(args.candidate_checkpoint, device)

    cand_score = 0.0
    wins = 0
    draws = 0
    losses = 0

    for g in range(args.games):
        cand_is_white = (g % 2 == 0)

        white = AlphaZeroMCTS(cand_model if cand_is_white else best_model, device=device, simulations=args.simulations, c_puct=args.c_puct)
        black = AlphaZeroMCTS(best_model if cand_is_white else cand_model, device=device, simulations=args.simulations, c_puct=args.c_puct)

        result = play_game(white, black, max_plies=args.max_plies)

        if result == "1-0":
            cand_win = cand_is_white
        elif result == "0-1":
            cand_win = not cand_is_white
        else:
            cand_win = None

        if cand_win is True:
            wins += 1
            cand_score += 1.0
        elif cand_win is False:
            losses += 1
        else:
            draws += 1
            cand_score += 0.5

        print(f"[arena {g+1}/{args.games}] result={result} cand_score={cand_score:.1f}")

    rate = cand_score / float(args.games)
    print(f"[summary] W/D/L={wins}/{draws}/{losses} score={rate:.3f}")

    if rate >= args.promote_threshold:
        shutil.copy2(args.candidate_checkpoint, args.best_checkpoint)
        print(f"[promote] candidate -> best (threshold={args.promote_threshold:.2f})")
    else:
        print(f"[keep] best unchanged (threshold={args.promote_threshold:.2f})")


if __name__ == "__main__":
    main()
