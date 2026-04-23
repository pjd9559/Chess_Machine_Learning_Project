"""
data_loader.py — Data pipeline: loaders for PGN/CSV chess game datasets.

Provides placeholder loaders for three data sources:
- Historical Games (Lumbras Gigabase)
- Strategic Games (LAION Strategic Dataset on HuggingFace)
- Online GM Games (Kaggle)

These loaders are designed to be used in a future training pipeline.
"""

import os
import csv
import chess.pgn
from typing import Generator, Optional


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pgn_games(
    pgn_path: str,
    max_games: Optional[int] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
) -> Generator:
    """
    Generator that yields chess.pgn.Game objects from a PGN file.

    Parameters
    ----------
    pgn_path  : path to the .pgn file
    max_games : stop after this many games (None = all)
    min_year  : only include games from this year onward
    max_year  : only include games up to this year
    """
    count = 0
    with open(pgn_path, "r") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Year filter
            date_str = game.headers.get("Date", "")
            if date_str and (min_year or max_year):
                try:
                    year = int(date_str.split(".")[0])
                    if min_year and year < min_year:
                        continue
                    if max_year and year > max_year:
                        continue
                except (ValueError, IndexError):
                    pass

            yield game
            count += 1
            if max_games and count >= max_games:
                break


def load_csv_games(csv_path: str, max_games: Optional[int] = None) -> Generator:
    """
    Generator that yields rows (as dicts) from a CSV chess dataset.
    """
    count = 0
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
            count += 1
            if max_games and count >= max_games:
                break


def filter_games_by_player(games, player_name: str) -> Generator:
    """
    Filter PGN games where a specific player participated
    (as White or Black).
    """
    for game in games:
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        if player_name.lower() in white.lower() or player_name.lower() in black.lower():
            yield game


def games_to_positions(game) -> list[tuple]:
    """
    Convert a single chess.pgn.Game into a list of
    (board_fen, move_uci) tuples for training data generation.
    """
    positions = []
    board = game.board()
    for move in game.mainline_moves():
        positions.append((board.fen(), move.uci()))
        board.push(move)
    return positions


# ── Dataset-specific loaders (placeholders) ───────────────────────────────────

def load_lumbras_gigabase(path: str | None = None, max_games: int = 1000):
    """
    Load games from the Lumbras Gigabase PGN file.
    Source: https://lumbrasgigabase.com/en/

    Download the PGN and place it in the data/ directory.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "lumbras_gigabase.pgn")
    if not os.path.exists(path):
        print(f"[data_loader] File not found: {path}")
        print("[data_loader] Download from https://lumbrasgigabase.com/en/")
        return []
    return load_pgn_games(path, max_games=max_games)


def load_laion_strategic(path: str | None = None, max_games: int = 1000):
    """
    Load games from the LAION Strategic Game Chess dataset.
    Source: https://huggingface.co/datasets/laion/strategic_game_chess

    Download the dataset and place it in the data/ directory.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "laion_strategic.pgn")
    if not os.path.exists(path):
        print(f"[data_loader] File not found: {path}")
        print("[data_loader] Download from https://huggingface.co/datasets/laion/strategic_game_chess")
        return []
    return load_pgn_games(path, max_games=max_games)


def load_kaggle_gm_games(path: str | None = None, max_games: int = 1000):
    """
    Load GM games from the Kaggle dataset.
    Source: https://www.kaggle.com/datasets/dimitrioskourtikakis/gm-games-chesscom

    Download the CSV and place it in the data/ directory.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "gm_games.csv")
    if not os.path.exists(path):
        print(f"[data_loader] File not found: {path}")
        print("[data_loader] Download from https://www.kaggle.com/datasets/dimitrioskourtikakis/gm-games-chesscom")
        return []
    return load_csv_games(path, max_games=max_games)


