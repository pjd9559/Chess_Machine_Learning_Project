"""
personalities.py — Five chess bot personalities with heuristic evaluators.

Each personality class exposes:
    evaluate(board: chess.Board) -> float
        Returns a score from the perspective of the side to move.
        Higher = better for the side to move.

A lightweight PyTorch NN skeleton is included at the bottom for future
training pipelines; the heuristic evaluators work out-of-the-box.
"""

import chess
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# ── Piece values (standard centipawn) ─────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

# ── Central squares bitmask (d4, e4, d5, e5 + extended ring) ─────────────────
INNER_CENTER   = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])
EXTENDED_CENTER = chess.SquareSet([
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
])

# King-side squares for Romantic personality
KINGSIDE_SQUARES = chess.SquareSet([
    chess.F1, chess.G1, chess.H1,
    chess.F2, chess.G2, chess.H2,
    chess.F3, chess.G3, chess.H3,
    chess.F4, chess.G4, chess.H4,
    chess.F5, chess.G5, chess.H5,
    chess.F6, chess.G6, chess.H6,
    chess.F7, chess.G7, chess.H7,
    chess.F8, chess.G8, chess.H8,
])


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _material_balance(board: chess.Board) -> float:
    """Signed material score: positive favours White."""
    score = 0.0
    for ptype in PIECE_VALUES:
        score += len(board.pieces(ptype, chess.WHITE)) * PIECE_VALUES[ptype]
        score -= len(board.pieces(ptype, chess.BLACK)) * PIECE_VALUES[ptype]
    return score


def _mobility(board: chess.Board) -> int:
    """Number of legal moves for the side to move."""
    return board.legal_moves.count()


def _opponent_king_safety(board: chess.Board) -> float:
    """
    Rough king-safety penalty for the opponent.
    Counts how many of my pieces attack squares near the opponent king.
    """
    opp_color    = not board.turn
    opp_king_sq  = board.king(opp_color)
    if opp_king_sq is None:
        return 0.0

    king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[opp_king_sq])
    king_zone.add(opp_king_sq)

    attack_count = 0
    for sq in king_zone:
        attackers = board.attackers(board.turn, sq)
        attack_count += len(attackers)
    return float(attack_count)


def _doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count files that have more than one pawn of the given color."""
    count = 0
    for f in range(8):
        file_mask = chess.BB_FILES[f]
        pawns_on_file = board.pieces(chess.PAWN, color) & chess.SquareSet(file_mask)
        if len(pawns_on_file) > 1:
            count += len(pawns_on_file) - 1
    return count


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count pawns with no friendly pawns on adjacent files."""
    pawns = board.pieces(chess.PAWN, color)
    isolated = 0
    for sq in pawns:
        f = chess.square_file(sq)
        has_neighbour = False
        for adj_f in [f - 1, f + 1]:
            if 0 <= adj_f <= 7:
                adj_file_mask = chess.BB_FILES[adj_f]
                if board.pieces(chess.PAWN, color) & chess.SquareSet(adj_file_mask):
                    has_neighbour = True
                    break
        if not has_neighbour:
            isolated += 1
    return isolated


def _holes(board: chess.Board, color: chess.Color) -> int:
    """
    Count 'holes' — squares in ranks 4-6 (from our perspective) that cannot
    be defended by our pawns.  Simplified: squares in extended center with no
    adjacent friendly pawn that could advance to defend.
    """
    holes = 0
    # Our "territory" ranks
    if color == chess.WHITE:
        target_ranks = [3, 4, 5]   # ranks 4-6 (0-indexed)
    else:
        target_ranks = [2, 3, 4]   # ranks 3-5

    for sq in EXTENDED_CENTER:
        if chess.square_rank(sq) not in target_ranks:
            continue
        f = chess.square_file(sq)
        defended = False
        for adj_f in [f - 1, f + 1]:
            if 0 <= adj_f <= 7:
                # Check if there is a friendly pawn on adjacent file that
                # could potentially defend this square
                adj_file_pawns = board.pieces(chess.PAWN, color) & chess.SquareSet(chess.BB_FILES[adj_f])
                if adj_file_pawns:
                    defended = True
                    break
        if not defended:
            holes += 1
    return holes


# ═══════════════════════════════════════════════════════════════════════════════
#  Personality classes
# ═══════════════════════════════════════════════════════════════════════════════

class BasePersonality:
    """Abstract base for all personalities."""
    name: str = "Base"
    description: str = ""
    icon: str = "♟"
    color: str = "#888888"

    def evaluate(self, board: chess.Board) -> float:
        raise NotImplementedError

    def _perspective(self, raw_score: float, board: chess.Board) -> float:
        """Flip score so it is always from the perspective of side-to-move."""
        return raw_score if board.turn == chess.WHITE else -raw_score


class RomanticPersonality(BasePersonality):
    """
    The Romantic (Pre-1980) — Historical Imitation.
    Prioritises king-side attacks and piece activity toward the opponent king.
    """
    name = "The Romantic"
    description = "Aggressive, king-side attacker inspired by pre-1980 GMs"
    icon = "⚔️"
    color = "#e74c3c"

    def evaluate(self, board: chess.Board) -> float:
        mat  = _material_balance(board)
        # Bonus for pieces aimed at opponent's kingside
        king_attack = _opponent_king_safety(board) * 15.0
        # Bonus for controlling kingside squares
        ks_control = 0.0
        for sq in KINGSIDE_SQUARES:
            if board.attackers(board.turn, sq):
                ks_control += 3.0
        # Extra bonus for open files toward the king
        mobility_bonus = _mobility(board) * 2.0
        raw = mat + king_attack + ks_control + mobility_bonus
        return self._perspective(raw, board)


class ModernistPersonality(BasePersonality):
    """
    The Modernist (Post-1980) — Efficiency & Precision.
    Balanced, Stockfish-inspired centipawn evaluation.
    """
    name = "The Modernist"
    description = "Precise, engine-like play inspired by post-1980 GMs"
    icon = "🔬"
    color = "#3498db"

    def evaluate(self, board: chess.Board) -> float:
        mat  = _material_balance(board)
        mob  = _mobility(board) * 5.0
        # Center control
        center = 0.0
        for sq in INNER_CENTER:
            if board.piece_at(sq) and board.piece_at(sq).color == board.turn:
                center += 30.0
            if board.attackers(board.turn, sq):
                center += 10.0
        # King safety (own)
        own_king_sq = board.king(board.turn)
        if own_king_sq is not None:
            king_shield = len(board.attackers(board.turn, own_king_sq)) * 5.0
        else:
            king_shield = 0.0
        # Pawn structure penalties
        dp = _doubled_pawns(board, board.turn) * -15.0
        ip = _isolated_pawns(board, board.turn) * -12.0
        raw = mat + mob + center + king_shield + dp + ip
        return self._perspective(raw, board)


class ArchitectPersonality(BasePersonality):
    """
    The Architect — Spatial Advantage.
    Controls the 16 central squares + territory invasion via bitmask evaluation.
    """
    name = "The Architect"
    description = "Positional master focused on space and central control"
    icon = "🏛️"
    color = "#9b59b6"

    def evaluate(self, board: chess.Board) -> float:
        mat = _material_balance(board)
        # Central control (bitmask-based)
        inner_ctrl  = 0.0
        ext_ctrl    = 0.0
        for sq in INNER_CENTER:
            if board.attackers(board.turn, sq):
                inner_ctrl += 25.0
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                inner_ctrl += 40.0
        for sq in EXTENDED_CENTER:
            if board.attackers(board.turn, sq):
                ext_ctrl += 8.0
        # Territory invasion: squares attacked in opponent's half
        invasion = 0.0
        opp_ranks = range(4, 8) if board.turn == chess.WHITE else range(0, 4)
        for sq in chess.SQUARES:
            if chess.square_rank(sq) in opp_ranks:
                if board.attackers(board.turn, sq):
                    invasion += 4.0
        raw = mat + inner_ctrl + ext_ctrl + invasion
        return self._perspective(raw, board)


class TacticianPersonality(BasePersonality):
    """
    The Tactician — Initiative & Activity.
    Rewards mobility, forcing moves, and threats to the opponent king.
    """
    name = "The Tactician"
    description = "Dynamic calculator who thrives on initiative and tactics"
    icon = "⚡"
    color = "#f39c12"

    def evaluate(self, board: chess.Board) -> float:
        mat  = _material_balance(board)
        mob  = _mobility(board) * 8.0
        king_pressure = _opponent_king_safety(board) * 20.0
        # Bonus for checks / captures / promotions among legal moves
        forcing = 0.0
        for move in board.legal_moves:
            if board.gives_check(move):
                forcing += 20.0
            if board.is_capture(move):
                forcing += 10.0
            if move.promotion:
                forcing += 30.0
        raw = mat + mob + king_pressure + forcing
        return self._perspective(raw, board)


class StrategistPersonality(BasePersonality):
    """
    The Strategist — Structural Integrity.
    Prioritises material balance and penalises pawn-structure weaknesses.
    """
    name = "The Strategist"
    description = "Long-term thinker who values pawn structure and endgame"
    icon = "🧠"
    color = "#2ecc71"

    def evaluate(self, board: chess.Board) -> float:
        mat = _material_balance(board)
        # Pawn-structure penalties (for side to move)
        dp = _doubled_pawns(board, board.turn)   * -25.0
        ip = _isolated_pawns(board, board.turn)   * -20.0
        h  = _holes(board, board.turn)            * -15.0
        # Bonus: pawn-structure penalties for the *opponent*
        odp = _doubled_pawns(board, not board.turn)   * 25.0
        oip = _isolated_pawns(board, not board.turn)   * 20.0
        oh  = _holes(board, not board.turn)            * 15.0
        # Minor mobility component
        mob = _mobility(board) * 3.0
        raw = mat + dp + ip + h + odp + oip + oh + mob
        return self._perspective(raw, board)


# ── Registry ──────────────────────────────────────────────────────────────────
ALL_PERSONALITIES = {
    "romantic":   RomanticPersonality(),
    "modernist":  ModernistPersonality(),
    "architect":  ArchitectPersonality(),
    "tactician":  TacticianPersonality(),
    "strategist": StrategistPersonality(),
}

def get_personality(name: str) -> BasePersonality:
    return ALL_PERSONALITIES[name.lower()]

def list_personalities() -> list[str]:
    return list(ALL_PERSONALITIES.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#  PyTorch NN skeleton (for future training)
# ═══════════════════════════════════════════════════════════════════════════════

class ChessEvalNet(nn.Module):
    """
    Lightweight CNN that takes an 8×8×12 board tensor and outputs
    a scalar win-probability / reward score.

    Not used at runtime yet — kept as a skeleton for Phase 2+ training.
    """
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),           # output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 12, 8, 8)
        x = self.conv_block(x)
        x = self.fc(x)
        return x
