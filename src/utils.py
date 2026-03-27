"""
utils.py — Utility functions for the chess engine.

Provides:
- board_to_tensor: converts a chess.Board to an 8x8x12 numpy array
- parse_pgn: reads a PGN file and yields games
- render_board_svg: generates an SVG string of the board
"""

import chess
import chess.pgn
import chess.svg
import numpy as np
import io


# ── Piece-plane mapping ──────────────────────────────────────────────────────
# 12 planes: P, N, B, R, Q, K (white)  then  p, n, b, r, q, k (black)
PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board into an 8×8×12 numpy float32 tensor.

    Each of the 12 planes corresponds to one piece-type for one color.
    A cell is 1.0 if that piece occupies the square, else 0.0.
    Rank 8 is row 0 (top of the board from White's perspective).
    """
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            row = 7 - chess.square_rank(square)   # rank 8 → row 0
            col = chess.square_file(square)        # file a → col 0
            tensor[row, col, plane] = 1.0

    return tensor


def parse_pgn(path: str):
    """
    Generator that yields chess.pgn.Game objects from a PGN file.
    """
    with open(path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            yield game


def render_board_svg(
    board: chess.Board,
    last_move: chess.Move | None = None,
    size: int = 400,
    flipped: bool = False,
    selected_square: int | None = None,
    legal_targets: list[int] | None = None,
) -> str:
    """
    Render the board as an SVG string.

    Parameters
    ----------
    board        : current board state
    last_move    : highlight the last move played
    size         : pixel width/height of the SVG
    flipped      : render from Black's perspective
    selected_square : square index currently selected by the player
    legal_targets   : list of square indices that are legal targets
    """
    fill = {}
    # Highlight last move
    if last_move is not None:
        fill[last_move.from_square] = "#aaa23a66"
        fill[last_move.to_square]   = "#aaa23a66"

    # Highlight selected square
    if selected_square is not None:
        fill[selected_square] = "#33ccff88"

    # Highlight legal target squares
    if legal_targets:
        for sq in legal_targets:
            fill[sq] = "#33ff5566"

    check_square = None
    if board.is_check():
        check_square = board.king(board.turn)

    svg = chess.svg.board(
        board,
        lastmove=last_move,
        fill=fill,
        check=check_square,
        size=size,
        flipped=flipped,
        coordinates=True,
    )
    return svg
