from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import pandas as pd

from src.analytics.chess_features import board_feature_row, move_feature_row
from src.analytics.report_utils import ensure_generated_reports_dir, predict_move


MODEL_PAIRS = {
    "old": ("pre_2000", "post_2020"),
    "final": ("pre_2000_ultra_final", "post_2010_ultra_final"),
}


def _artifact_paths(model_pair: str, num_games: int, simulations: int, max_moves: int) -> tuple[Path, Path]:
    root = ensure_generated_reports_dir()
    key = f"style_{model_pair}_{num_games}_games_{simulations}_sims_{max_moves}_plies"
    return root / f"{key}_games.json", root / f"{key}_moves.csv"


def _narrative(summary_df: pd.DataFrame, first_model: str, second_model: str) -> str:
    if summary_df.empty:
        return "No style data available yet."
    rows = summary_df.set_index("model").to_dict("index")
    first = rows.get(first_model)
    second = rows.get(second_model)
    if not first or not second:
        return "Both models need completed games before style comparisons can be summarized."

    insights = []
    if second["sacrifice_rate"] > first["sacrifice_rate"]:
        insights.append(f"{second_model} sacrifices more often")
    else:
        insights.append(f"{first_model} sacrifices more often")
    if second["king_pressure_after"] > first["king_pressure_after"]:
        insights.append(f"{second_model} creates more king pressure")
    else:
        insights.append(f"{first_model} creates more king pressure")
    if second["center_control_after"] > first["center_control_after"]:
        insights.append(f"{second_model} tends to preserve center control better")
    else:
        insights.append(f"{first_model} tends to preserve center control better")
    return ". ".join(insights) + "."


def generate_style_report(
    num_games: int = 40,
    simulations: int = 12,
    max_moves: int = 80,
    model_pair: str = "old",
    seed: int = 0,
    force_recompute: bool = False,
) -> dict[str, object]:
    first_model, second_model = MODEL_PAIRS[model_pair]
    games_path, moves_path = _artifact_paths(model_pair, num_games, simulations, max_moves)
    if games_path.exists() and moves_path.exists() and not force_recompute:
        game_df = pd.DataFrame(json.loads(games_path.read_text()))
        move_df = pd.read_csv(moves_path)
    else:
        rng = random.Random(seed)
        game_rows: list[dict[str, object]] = []
        move_rows: list[dict[str, object]] = []

        for game_id in range(num_games):
            board = chess.Board()
            white_model = first_model if game_id % 2 == 0 else second_model
            black_model = second_model if white_model == first_model else first_model

            for ply in range(max_moves):
                if board.is_game_over(claim_draw=True):
                    break
                mover_model = white_model if board.turn == chess.WHITE else black_model
                before = board.copy(stack=True)
                move, _ = predict_move(mover_model, board, simulations=simulations)
                if move is None or move not in board.legal_moves:
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    move = legal_moves[rng.randrange(len(legal_moves))]
                board.push(move)
                after = board.copy(stack=True)

                move_row = {
                    "game_id": game_id,
                    "ply": ply + 1,
                    "mover_model": mover_model,
                    "white_model": white_model,
                    "black_model": black_model,
                    "result": "",
                    "before_fen": before.fen(),
                    "after_fen": after.fen(),
                    "move_uci": move.uci(),
                }
                move_row.update(board_feature_row(before, color=(mover_color := before.turn)))
                move_row.update(move_feature_row(before, move, after, mover_color))
                move_rows.append(move_row)

            result = board.result(claim_draw=True)
            plies = len(board.move_stack)
            winner = (
                white_model if result == "1-0" else black_model if result == "0-1" else "draw"
            )
            game_rows.append(
                {
                    "game_id": game_id,
                    "white_model": white_model,
                    "black_model": black_model,
                    "result": result,
                    "winner": winner,
                    "plies": plies,
                    "opening_family": board_feature_row(board)["opening_family"],
                    "game_phase": board_feature_row(board)["game_phase"],
                }
            )

        game_df = pd.DataFrame(game_rows)
        move_df = pd.DataFrame(move_rows)
        games_path.write_text(game_df.to_json(orient="records", indent=2))
        move_df.to_csv(moves_path, index=False)

    if move_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            move_df.groupby("mover_model")
            .agg(
                sacrifice_rate=("sacrifice_proxy", "mean"),
                mobility_after=("mobility_after", "mean"),
                king_pressure_after=("king_pressure_after", "mean"),
                center_control_after=("center_control_after", "mean"),
                checks_given=("gives_check", "mean"),
                capture_rate=("is_capture", "mean"),
                queen_trade_rate=("queen_trade", "mean"),
                castling_rate=("is_castle", "mean"),
                avg_move_tactical_pressure=("tactical_pressure_after", "mean"),
            )
            .reset_index()
            .rename(columns={"mover_model": "model"})
        )

        avg_lengths = {
            first_model: float(game_df["plies"].mean()) if not game_df.empty else 0.0,
            second_model: float(game_df["plies"].mean()) if not game_df.empty else 0.0,
        }
        summary_df["avg_game_length"] = summary_df["model"].map(avg_lengths)
        for col in summary_df.columns[1:]:
            summary_df[col] = summary_df[col].astype(float)

    return {
        "games": game_df,
        "moves": move_df,
        "summary": summary_df,
        "narrative": _narrative(summary_df, first_model, second_model),
        "games_artifact": str(games_path),
        "moves_artifact": str(moves_path),
    }
