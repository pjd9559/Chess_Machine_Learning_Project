from __future__ import annotations

import chess


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

INNER_CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
]


def _board_for_color(board: chess.Board, color: chess.Color) -> chess.Board:
    if board.turn == color:
        return board.copy(stack=False)
    clone = board.copy(stack=False)
    clone.turn = color
    return clone


def material_total(board: chess.Board) -> float:
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        total += value * (len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK)))
    return total


def material_balance(board: chess.Board, color: chess.Color) -> float:
    own = 0.0
    opp = 0.0
    for piece_type, value in PIECE_VALUES.items():
        own += value * len(board.pieces(piece_type, color))
        opp += value * len(board.pieces(piece_type, not color))
    return own - opp


def mobility(board: chess.Board, color: chess.Color | None = None) -> int:
    color = board.turn if color is None else color
    clone = _board_for_color(board, color)
    return clone.legal_moves.count()


def center_control(board: chess.Board, color: chess.Color | None = None) -> float:
    color = board.turn if color is None else color
    score = 0.0
    for sq in INNER_CENTER:
        if board.piece_at(sq) and board.piece_at(sq).color == color:
            score += 2.0
        score += 0.5 * len(board.attackers(color, sq))
    for sq in EXTENDED_CENTER:
        score += 0.15 * len(board.attackers(color, sq))
    return score


def king_pressure(board: chess.Board, color: chess.Color | None = None) -> float:
    color = board.turn if color is None else color
    opp_king = board.king(not color)
    if opp_king is None:
        return 0.0
    zone = set(chess.SquareSet(chess.BB_KING_ATTACKS[opp_king]))
    zone.add(opp_king)
    return float(sum(len(board.attackers(color, sq)) for sq in zone))


def tactical_pressure(board: chess.Board, color: chess.Color | None = None) -> float:
    color = board.turn if color is None else color
    clone = _board_for_color(board, color)
    captures = sum(1 for move in clone.legal_moves if clone.is_capture(move))
    checks = 0
    for move in clone.legal_moves:
        clone.push(move)
        checks += int(clone.is_check())
        clone.pop()
    return float(captures + checks)


def game_phase(board: chess.Board) -> str:
    total_material = material_total(board)
    if board.fullmove_number <= 10:
        return "opening"
    if total_material <= 20 or (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)) == 0):
        return "endgame"
    return "middlegame"


def opening_family(board: chess.Board) -> str:
    moves = [move.uci() for move in board.move_stack[:6]]
    if not moves:
        return "Unknown"
    white_first = moves[0]
    black_first = moves[1] if len(moves) > 1 else ""
    white_second = moves[2] if len(moves) > 2 else ""

    if white_first == "e2e4":
        if black_first == "c7c5":
            return "Sicilian"
        if black_first == "e7e5":
            return "Open Game"
        if black_first == "e7e6":
            return "French"
        if black_first == "c7c6":
            return "Caro-Kann"
        return "King Pawn"
    if white_first == "d2d4":
        if black_first == "d7d5" and white_second == "c2c4":
            return "Queen's Gambit"
        if black_first == "g8f6":
            return "Indian Defense"
        return "Queen Pawn"
    if white_first == "c2c4":
        return "English"
    if white_first == "g1f3":
        return "Reti"
    return "Other"


def board_feature_row(board: chess.Board, color: chess.Color | None = None) -> dict[str, float | str]:
    color = board.turn if color is None else color
    return {
        "game_phase": game_phase(board),
        "opening_family": opening_family(board),
        "material_total": material_total(board),
        "material_balance": material_balance(board, color),
        "mobility": float(mobility(board, color)),
        "center_control": center_control(board, color),
        "king_pressure": king_pressure(board, color),
        "tactical_pressure": tactical_pressure(board, color),
    }


def move_feature_row(before: chess.Board, move: chess.Move, after: chess.Board, mover_color: chess.Color) -> dict[str, float | int | str]:
    moved_piece = before.piece_at(move.from_square)
    capture_value = 0.0
    captured_piece = before.piece_at(move.to_square) if before.is_capture(move) else None
    if captured_piece:
        capture_value = PIECE_VALUES[captured_piece.piece_type]

    after_piece = after.piece_at(move.to_square)
    attacked = int(after_piece is not None and len(after.attackers(not mover_color, move.to_square)) > 0)
    defended = int(after_piece is not None and len(after.attackers(mover_color, move.to_square)) > 0)
    moved_piece_value = PIECE_VALUES[moved_piece.piece_type] if moved_piece else 0.0

    return {
        "move_piece_type": moved_piece.symbol().lower() if moved_piece else "?",
        "is_capture": int(before.is_capture(move)),
        "gives_check": int(after.is_check()),
        "is_castle": int(before.is_castling(move)),
        "queen_trade": int(
            len(before.pieces(chess.QUEEN, chess.WHITE)) + len(before.pieces(chess.QUEEN, chess.BLACK)) >
            len(after.pieces(chess.QUEEN, chess.WHITE)) + len(after.pieces(chess.QUEEN, chess.BLACK))
        ),
        "capture_value": capture_value,
        "sacrifice_proxy": int(attacked and moved_piece_value >= 3 and capture_value < moved_piece_value and not defended),
        "mobility_after": float(mobility(after, mover_color)),
        "center_control_after": center_control(after, mover_color),
        "king_pressure_after": king_pressure(after, mover_color),
        "tactical_pressure_after": tactical_pressure(after, mover_color),
    }

