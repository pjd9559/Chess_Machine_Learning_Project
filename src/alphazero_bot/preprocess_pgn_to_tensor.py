import os
from pathlib import Path
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # Chess_Engines_Project
sys.path.append(str(ROOT / "data"))

from data_loader import load_pgn_games
from encoding import board_to_tensor, move_to_index, outcome_to_value


def process_pgn_file(pgn_path, out_dir, chunk_size=5000, min_year=None, max_year=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer = []
    existing = sorted(out_dir.glob("chunk_*.pt"))
    if existing:
        chunk_id = max(int(f.stem.split("_")[1]) for f in existing) + 1
        print(f"[resume] starting from chunk {chunk_id}")
    else:
        chunk_id = 0
    total_positions = 0

    print(f"[processing] {pgn_path.name}")

    games = load_pgn_games(
        str(pgn_path),
        max_games=None,
        min_year=min_year,
        max_year=max_year
    )

    for i, game in enumerate(games, start=1):
        board = game.board()
        result = game.headers.get("Result", "1/2-1/2")

        for move in game.mainline_moves():
            try:
                x = board_to_tensor(board)
                y = move_to_index(board, move)
                z = outcome_to_value(result, board.turn)

                buffer.append((x, y, z))
                total_positions += 1

                if total_positions % 10000 == 0:
                    print(f"[positions] processed {total_positions}")
                    

            except ValueError:
                pass

            board.push(move)

        if i % 100 == 0:
            print(f"[games] processed {i}")

        # 🔥 Save chunk
        if len(buffer) >= chunk_size:
            save_chunk(buffer, out_dir, chunk_id)
            buffer = []
            chunk_id += 1

    # Save remaining
    if buffer:
        save_chunk(buffer, out_dir, chunk_id)

    print(f"[done] {pgn_path.name} → {total_positions} positions")


def save_chunk(buffer, out_dir, chunk_id):
    states, moves, values = zip(*buffer)

    path = out_dir / f"chunk_{chunk_id}.pt"

    torch.save({
        "states": torch.stack(states),
        "moves": torch.tensor(moves),
        "values": torch.tensor(values),
    }, path)

    print(f"[saved] {path}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pgn-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--max-year", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=5000)

    args = p.parse_args()

    pgn_files = sorted(args.pgn_dir.glob("*.pgn"))

    for pgn_file in pgn_files:
        process_pgn_file(
            pgn_file,
            args.out_dir,
            chunk_size=args.chunk_size,
            min_year=args.min_year,
            max_year=args.max_year
        )


if __name__ == "__main__":
    main()