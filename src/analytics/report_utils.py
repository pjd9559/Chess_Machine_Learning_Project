from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import chess
import torch

from src.alphazero_bot.mcts import AlphaZeroMCTS, choose_move
from src.alphazero_bot.model import ChessNet


ROOT_DIR = Path(__file__).resolve().parents[2]
GENERATED_REPORTS_DIR = ROOT_DIR / "generated_reports"

MODEL_PATHS = {
    "pre_2000": ROOT_DIR / "src" / "models" / "pre_2000.pt",
    "post_2020": ROOT_DIR / "src" / "models" / "post_2020.pt",
    "pre_2000_ultra_final": ROOT_DIR / "src" / "models" / "pre_2000_ultra_final.pt",
    "post_2010_ultra_final": ROOT_DIR / "src" / "models" / "post_2010_ultra_final.pt",
}

PUZZLE_SUITES = {
    "m8n1": {"path": ROOT_DIR / "m8n1.csv", "format": "csv", "mate_label": "mate-in-1"},
    "m8n2": {"path": ROOT_DIR / "m8n2.txt", "format": "txt", "mate_label": "mate-in-2"},
    "m8n3": {"path": ROOT_DIR / "m8n3.txt", "format": "txt", "mate_label": "mate-in-3"},
    "m8n4": {"path": ROOT_DIR / "m8n4.txt", "format": "txt", "mate_label": "mate-in-4"},
}


def ensure_generated_reports_dir() -> Path:
    GENERATED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_REPORTS_DIR


@lru_cache(maxsize=4)
def load_model(model_name: str) -> ChessNet:
    path = MODEL_PATHS[model_name]
    model = ChessNet()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def parse_solution_tokens(line: str) -> list[str]:
    cleaned = re.sub(r"\b\d+\.(\.\.)?", " ", line.strip())
    return [token for token in cleaned.split() if token]


def parse_txt_puzzles(path: Path) -> list[dict[str, object]]:
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
        if not tokens:
            continue

        board = chess.Board(fen)
        key_move = board.parse_san(tokens[0])
        puzzles.append(
            {
                "suite": path.stem,
                "title": title,
                "fen": fen,
                "side": "white" if board.turn == chess.WHITE else "black",
                "key_move": key_move.uci(),
                "mate_label": PUZZLE_SUITES[path.stem]["mate_label"] if path.stem in PUZZLE_SUITES else "puzzle",
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
            puzzles.append(
                {
                    "suite": path.stem,
                    "title": f"{path.name} row {idx}",
                    "fen": fen,
                    "side": "white" if board.turn == chess.WHITE else "black",
                    "key_move": best,
                    "mate_label": PUZZLE_SUITES[path.stem]["mate_label"] if path.stem in PUZZLE_SUITES else "puzzle",
                }
            )

    return puzzles


def load_puzzle_suite(suite_name: str) -> list[dict[str, object]]:
    suite = PUZZLE_SUITES[suite_name]
    if suite["format"] == "csv":
        return parse_csv_puzzles(Path(suite["path"]))
    return parse_txt_puzzles(Path(suite["path"]))


def predict_move(
    model_name: str,
    board: chess.Board,
    simulations: int = 64,
) -> tuple[chess.Move | None, dict[chess.Move, int]]:
    model = load_model(model_name)
    mcts = AlphaZeroMCTS(model=model, device=torch.device("cpu"), simulations=simulations)
    visits, _ = mcts.run(board)
    if not visits:
        return None, {}
    return choose_move(visits, temperature=0.0), visits


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default
