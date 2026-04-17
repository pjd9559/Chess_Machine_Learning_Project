import torch
from torch.utils.data import IterableDataset
import chess.pgn

from data.data_loader import load_pgn_games
from src.alphazero_bot.encoding import board_to_tensor, move_to_index, outcome_to_value


class ChessIterableDataset(IterableDataset):
    def __init__(self, pgn_dir, max_games=None, min_year=None, max_year=None):
        self.pgn_dir = pgn_dir
        self.max_games = max_games
        self.min_year = min_year
        self.max_year = max_year

    def __iter__(self):
        game_count = 0

        import os
        pgn_files = [os.path.join(self.pgn_dir, f) for f in os.listdir(self.pgn_dir) if f.endswith(".pgn")]

        for pgn_file in pgn_files:
            print(f"[stream] loading {pgn_file}")

            games = load_pgn_games(
                pgn_file,
                max_games=None,
                min_year=self.min_year,
                max_year=self.max_year
            )

            for game in games:
                board = game.board()
                result = game.headers.get("Result", "1/2-1/2")

                for move in game.mainline_moves():
                    x = board_to_tensor(board)
                    y = move_to_index(board, move)
                    z = torch.tensor(outcome_to_value(result, board.turn), dtype=torch.float32)

                    yield x, y, z

                    board.push(move)

                game_count += 1

                if game_count % 100 == 0:
                    print(f"[stream] processed {game_count} games")

                if self.max_games and game_count >= self.max_games:
                    return