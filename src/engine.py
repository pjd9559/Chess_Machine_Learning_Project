import chess
import random
import torch

from src.personalities import get_personality, BasePersonality
from src.alphazero_bot.encoding import board_to_tensor, move_to_index
from src.alphazero_bot.mcts import AlphaZeroMCTS, choose_move


class GameEngine:
    """
    Manages a single chess game.
    """

    def __init__(self, board: chess.Board | None = None):
        self.board: chess.Board = board if board else chess.Board()
        self.last_move: chess.Move | None = None

    # ── Human moves ─────────────────────────────────────────

    def make_move(self, uci_str: str, promotion_piece: chess.PieceType | None = None) -> bool:
        try:
            move = chess.Move.from_uci(uci_str)
        except ValueError:
            return False

        # Promotion handling
        if promotion_piece and self._is_promotion_move(move):
            move = chess.Move(move.from_square, move.to_square, promotion=promotion_piece)

        if move in self.board.legal_moves:
            self.board.push(move)
            self.last_move = move
            return True

        # Auto-queen fallback
        if self._is_promotion_move(move) and not move.promotion:
            promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promo_move in self.board.legal_moves:
                self.board.push(promo_move)
                self.last_move = promo_move
                return True

        return False

    def make_move_obj(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves:
            self.board.push(move)
            self.last_move = move
            return True
        return False

    # ── Bot moves ─────────────────────────────────────────

    def bot_move(self, personality_name: str) -> chess.Move | None:
        """
        Generate a bot move using:
        - heuristic personalities
        - neural network + MCTS (AlphaZero-style)
        """

        if self.board.is_game_over():
            return None

        personality = get_personality(personality_name)
        legal_moves = list(self.board.legal_moves)

        if not legal_moves:
            return None

        # ─────────────────────────────────────────────
        # 🔥 Neural Network + MCTS (KEY UPGRADE)
        # ─────────────────────────────────────────────
        if hasattr(personality, "model"):
            model = personality.model

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Move model to correct device
            model = model.to(device)
            model.eval()

            # 🔥 MCTS SEARCH
            mcts = AlphaZeroMCTS(
                model=model,
                device=device,
                simulations=256,
            )

            visits, _ = mcts.run(self.board)

            # Deterministic best move (no randomness)
            move = choose_move(visits, temperature=0.0)

            # Safety fallback
            if move not in legal_moves:
                move = random.choice(legal_moves)

            self.board.push(move)
            self.last_move = move
            return move

        # ─────────────────────────────────────────────
        # Heuristic personalities (unchanged)
        # ─────────────────────────────────────────────
        best_move = None
        best_score = float("-inf")

        for move in legal_moves:
            self.board.push(move)
            score = -personality.evaluate(self.board)
            self.board.pop()

            score += random.uniform(-5, 5)

            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            self.board.push(best_move)
            self.last_move = best_move

        return best_move

    # ── Game status ───────────────────────────────────────

    def get_game_status(self) -> dict:
        status = {
            "is_over": self.board.is_game_over(),
            "in_check": self.board.is_check(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "result": self.board.result() if self.board.is_game_over() else "*",
            "reason": "ongoing",
        }

        if self.board.is_checkmate():
            status["reason"] = "checkmate"
        elif self.board.is_stalemate():
            status["reason"] = "stalemate"
        elif self.board.is_insufficient_material():
            status["reason"] = "insufficient_material"
        elif self.board.can_claim_fifty_moves():
            status["reason"] = "fifty_move_rule"
        elif self.board.can_claim_threefold_repetition():
            status["reason"] = "threefold_repetition"

        return status

    def undo_move(self) -> bool:
        if self.board.move_stack:
            self.board.pop()
            self.last_move = self.board.move_stack[-1] if self.board.move_stack else None
            return True
        return False

    def undo_two_moves(self) -> bool:
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            self.last_move = self.board.move_stack[-1] if self.board.move_stack else None
            return True
        return False

    def get_move_history(self) -> list[str]:
        temp_board = chess.Board()
        moves = []
        for move in self.board.move_stack:
            san = temp_board.san(move)
            moves.append(san)
            temp_board.push(move)
        return moves

    def get_legal_moves_for_square(self, square: int) -> list[chess.Move]:
        return [m for m in self.board.legal_moves if m.from_square == square]

    # ── Internal helpers ───────────────────────────────

    def _is_promotion_move(self, move: chess.Move) -> bool:
        piece = self.board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False

        target_rank = chess.square_rank(move.to_square)

        return (
            (piece.color == chess.WHITE and target_rank == 7)
            or (piece.color == chess.BLACK and target_rank == 0)
        )
