from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import chess
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.alphazero_bot.mcts import AlphaZeroMCTS, choose_move
from src.alphazero_bot.model import ChessNet


PUZZLE_SUITES = {
    "m8n1": {
        "puzzle_path": ROOT_DIR / "m8n1.csv",
        "label": "mate-in-1",
        "format": "csv",
        "max_puzzles": 100,
        "simulations": 256,
    },
    "m8n2": {
        "puzzle_path": ROOT_DIR / "m8n2.txt",
        "label": "mate-in-2",
        "format": "txt",
        "max_puzzles": 100,
        "simulations": 256,
    },
    "m8n3": {
        "puzzle_path": ROOT_DIR / "m8n3.txt",
        "label": "mate-in-3",
        "format": "txt",
        "max_puzzles": 100,
        "simulations": 256,
    },
    "m8n4": {
        "puzzle_path": ROOT_DIR / "m8n4.txt",
        "label": "mate-in-4",
        "format": "txt",
        "max_puzzles": 100,
        "simulations": 256,
    },
}

MODEL_GROUPS = {
    "old": {
        "pre_2000": ROOT_DIR / "src" / "models" / "pre_2000.pt",
        "post_2020": ROOT_DIR / "src" / "models" / "post_2020.pt",
    },
    "v1": {
        "pre_2000_final_v1": ROOT_DIR / "src" / "models" / "pre_2000_final_v1.pt",
        "post_2010_final_v1": ROOT_DIR / "src" / "models" / "post_2010_final_v1.pt",
    },
    "v2": {
        "pre_2000_final_v2": ROOT_DIR / "src" / "models" / "pre_2000_final_v2.pt",
        "post_2010_final_v2": ROOT_DIR / "src" / "models" / "post_2010_final_v2.pt",
    },
    "ultral_final": {
        "pre_2000_ultra_final": ROOT_DIR / "src" / "models" / "pre_2000_ultra_final.pt",
        "post_2010_ultra_final": ROOT_DIR / "src" / "models" / "post_2010_ultra_final.pt",
    },
}

RESULTS_DIR = ROOT_DIR / "tests" / "256"
GROUP_MAX_PUZZLES = {
    "ultral_final": 100,
}


def load_model(path: Path) -> ChessNet:
    model = ChessNet()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def parse_solution_tokens(line: str) -> list[str]:
    cleaned = re.sub(r"\b\d+\.(\.\.)?", " ", line.strip())
    return [token for token in cleaned.split() if token]


def parse_puzzles(path: Path) -> list[dict[str, object]]:
    lines = path.read_text().splitlines()
    puzzles: list[dict[str, object]] = []

    for i, raw_line in enumerate(lines):
        fen = raw_line.strip()
        if "/" not in fen or (" w " not in fen and " b " not in fen):
            continue

        title = "Unknown puzzle"
        for j in range(i - 1, -1, -1):
            prev = lines[j].strip()
            if prev:
                title = prev
                break

        k = i + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        if k >= len(lines):
            continue

        tokens = parse_solution_tokens(lines[k])
        if len(tokens) < 1:
            continue

        board = chess.Board(fen)
        key_move = board.parse_san(tokens[0])

        puzzles.append(
            {
                "title": title,
                "fen": fen,
                "side": "white" if board.turn == chess.WHITE else "black",
                "key_move": key_move,
            }
        )

    return puzzles


def parse_csv_puzzles(path: Path) -> list[dict[str, object]]:
    puzzles: list[dict[str, object]] = []

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, start=1):
            fen = (row.get("fen") or "").strip()
            best = (row.get("best") or "").strip()
            if not fen or not best:
                continue

            board = chess.Board(fen)
            key_move = chess.Move.from_uci(best)

            puzzles.append(
                {
                    "title": f"{path.name} row {idx}",
                    "fen": fen,
                    "side": "white" if board.turn == chess.WHITE else "black",
                    "key_move": key_move,
                }
            )

    return puzzles


def maybe_limit_puzzles(
    puzzles: list[dict[str, object]],
    max_puzzles: int | None,
) -> list[dict[str, object]]:
    if max_puzzles is None:
        return puzzles
    return puzzles[:max_puzzles]


def model_pick(model: ChessNet, board: chess.Board, simulations: int = 64) -> chess.Move | None:
    mcts = AlphaZeroMCTS(model=model, device=torch.device("cpu"), simulations=simulations)
    visits, _ = mcts.run(board)
    if not visits:
        return None
    return choose_move(visits, temperature=0.0)


def evaluate_model(
    model_name: str,
    model: ChessNet,
    puzzles: list[dict[str, object]],
    simulations: int = 64,
) -> dict[str, object]:
    total = len(puzzles)
    solved = 0
    white_total = 0
    black_total = 0
    white_solved = 0
    black_solved = 0
    solved_puzzles: list[str] = []

    for puzzle in puzzles:
        board = chess.Board(str(puzzle["fen"]))
        predicted = model_pick(model, board, simulations=simulations)
        expected = puzzle["key_move"]
        is_solved = predicted == expected

        solved += int(is_solved)
        if puzzle["side"] == "white":
            white_total += 1
            white_solved += int(is_solved)
        else:
            black_total += 1
            black_solved += int(is_solved)

        if is_solved and predicted is not None:
            solved_puzzles.append(f'- {puzzle["title"]}: {predicted.uci()}')

    solve_rate = 100.0 * solved / total if total else 0.0
    white_rate = 100.0 * white_solved / white_total if white_total else 0.0
    black_rate = 100.0 * black_solved / black_total if black_total else 0.0

    return {
        "model": model_name,
        "total": total,
        "solved": solved,
        "solve_rate": solve_rate,
        "white_total": white_total,
        "white_solved": white_solved,
        "white_rate": white_rate,
        "black_total": black_total,
        "black_solved": black_solved,
        "black_rate": black_rate,
        "solved_puzzles": solved_puzzles,
    }


def format_results(
    group_name: str,
    suite_name: str,
    mate_label: str,
    results: list[dict[str, object]],
    puzzle_count: int,
    evaluated_count: int,
) -> str:
    lines = [
        f"Model generation: {group_name}",
        f"{suite_name.upper()} Puzzle Results",
        f"Criterion: only the first move must match the {mate_label} key move.",
        f"Total puzzles in source: {puzzle_count}",
        f"Puzzles evaluated: {evaluated_count}",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f'Model: {result["model"]}',
                f'Solved: {result["solved"]} / {result["total"]} ({result["solve_rate"]:.1f}%)',
                f'White to move: {result["white_solved"]} / {result["white_total"]} ({result["white_rate"]:.1f}%)',
                f'Black to move: {result["black_solved"]} / {result["black_total"]} ({result["black_rate"]:.1f}%)',
            ]
        )

        solved_puzzles = result["solved_puzzles"]
        if solved_puzzles:
            lines.append("Solved puzzles:")
            if len(solved_puzzles) > 50:
                lines.extend(solved_puzzles[:50])
                lines.append(f"... and {len(solved_puzzles) - 50} more solved puzzles")
            else:
                lines.extend(solved_puzzles)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    for group_name, model_paths in MODEL_GROUPS.items():
        group_dir = RESULTS_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        loaded_models = {
            model_name: load_model(model_path)
            for model_name, model_path in model_paths.items()
        }

        for suite_name, suite_info in PUZZLE_SUITES.items():
            if suite_info["format"] == "csv":
                all_puzzles = parse_csv_puzzles(suite_info["puzzle_path"])
            else:
                all_puzzles = parse_puzzles(suite_info["puzzle_path"])
            max_puzzles = GROUP_MAX_PUZZLES.get(group_name, suite_info["max_puzzles"])
            puzzles = maybe_limit_puzzles(all_puzzles, max_puzzles)
            results = []

            for model_name, model in loaded_models.items():
                results.append(
                    evaluate_model(
                        model_name,
                        model,
                        puzzles,
                        simulations=int(suite_info["simulations"]),
                    )
                )

            output = format_results(
                group_name,
                suite_name,
                str(suite_info["label"]),
                results,
                len(all_puzzles),
                len(puzzles),
            )
            (group_dir / f"{suite_name}_results.txt").write_text(output)
            print(output)
            print("-" * 60)


if __name__ == "__main__":
    main()
