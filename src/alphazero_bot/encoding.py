from __future__ import annotations

import chess
import torch

# 64 squares * 73 move types (AlphaZero-style action encoding)
N_MOVES = 4672

_DIRECTIONS = [
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
]
_KNIGHT_DELTAS = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]


def _oriented_square(square: int, turn: bool) -> int:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def _move_type_index(from_sq: int, to_sq: int, promotion: int | None) -> int:
    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = tf - ff, tr - fr

    # Sliding moves: 8 directions x distances 1..7 => 56
    for direction_id, (ddf, ddr) in enumerate(_DIRECTIONS):
        for distance in range(1, 8):
            if (df, dr) == (ddf * distance, ddr * distance):
                return direction_id * 7 + (distance - 1)

    # Knight moves: 8 entries => 56..63
    for knight_id, delta in enumerate(_KNIGHT_DELTAS):
        if (df, dr) == delta:
            return 56 + knight_id

    # Underpromotions: to knight/bishop/rook with forward/diag-left/diag-right => 64..72
    if promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        if dr != 1:
            raise ValueError("Invalid promotion direction")
        if df == 0:
            dir_id = 0
        elif df == -1:
            dir_id = 1
        elif df == 1:
            dir_id = 2
        else:
            raise ValueError("Invalid promotion delta")

        promo_id = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[promotion]
        return 64 + promo_id * 3 + dir_id

    raise ValueError("Move cannot be encoded in 73-action policy head")


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    from_sq = _oriented_square(move.from_square, board.turn)
    to_sq = _oriented_square(move.to_square, board.turn)
    move_type = _move_type_index(from_sq, to_sq, move.promotion)
    return from_sq * 73 + move_type


def _decode_move_type(move_type: int) -> tuple[int, int, int | None]:
    if 0 <= move_type < 56:
        direction_id = move_type // 7
        distance = (move_type % 7) + 1
        df, dr = _DIRECTIONS[direction_id]
        return df * distance, dr * distance, None

    if 56 <= move_type < 64:
        knight_id = move_type - 56
        df, dr = _KNIGHT_DELTAS[knight_id]
        return df, dr, None

    if 64 <= move_type < 73:
        x = move_type - 64
        promo_id, dir_id = divmod(x, 3)
        promotion = (chess.KNIGHT, chess.BISHOP, chess.ROOK)[promo_id]
        df = (0, -1, 1)[dir_id]
        return df, 1, promotion

    raise ValueError("Invalid move type")


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    if index < 0 or index >= N_MOVES:
        raise ValueError(f"Policy index out of range: {index}")

    from_sq_oriented, move_type = divmod(index, 73)
    df, dr, promotion = _decode_move_type(move_type)

    ff, fr = chess.square_file(from_sq_oriented), chess.square_rank(from_sq_oriented)
    tf, tr = ff + df, fr + dr
    if not (0 <= tf < 8 and 0 <= tr < 8):
        raise ValueError("Decoded move leaves board")

    to_sq_oriented = chess.square(tf, tr)

    from_sq = _oriented_square(from_sq_oriented, board.turn)
    to_sq = _oriented_square(to_sq_oriented, board.turn)

    if promotion is None and board.piece_type_at(from_sq) == chess.PAWN:
        rank_to = chess.square_rank(to_sq)
        if rank_to in (0, 7):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    x = torch.zeros((18, 8, 8), dtype=torch.float32)

    for sq, piece in board.piece_map().items():
        sq_o = _oriented_square(sq, board.turn)
        row = 7 - chess.square_rank(sq_o)
        col = chess.square_file(sq_o)

        piece_offset = 0 if piece.color == board.turn else 6
        piece_id = piece_offset + (piece.piece_type - 1)
        x[piece_id, row, col] = 1.0

    # Castling rights planes (from current side perspective)
    x[12, :, :] = float(board.has_kingside_castling_rights(board.turn))
    x[13, :, :] = float(board.has_queenside_castling_rights(board.turn))
    x[14, :, :] = float(board.has_kingside_castling_rights(not board.turn))
    x[15, :, :] = float(board.has_queenside_castling_rights(not board.turn))

    # Side to move and half-move clock planes
    x[16, :, :] = 1.0
    x[17, :, :] = min(float(board.halfmove_clock) / 100.0, 1.0)

    return x


def outcome_to_value(result: str, turn: bool) -> float:
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if turn == chess.WHITE else 1.0
    return 0.0
