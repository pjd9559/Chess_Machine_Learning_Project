from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import torch

from src.alphazero_bot.encoding import N_MOVES, board_to_tensor, move_to_index


@dataclass
class Node:
    board: chess.Board
    prior: float
    parent: "Node | None" = None
    move: chess.Move | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "Node"] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class AlphaZeroMCTS:
    def __init__(
        self,
        model,
        device: torch.device,
        simulations: int = 128,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        root_prune_top_k: int = 24,
        min_prior: float = 0.0,
    ):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root_prune_top_k = root_prune_top_k
        self.min_prior = min_prior

    def _terminal_value(self, board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return -0.05  # 🔥 draw penalty
        return 1.0 if outcome.winner == board.turn else -1.0

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        self.model.eval()
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(x)
        priors_all = torch.softmax(policy_logits[0], dim=0)

        priors: list[tuple[chess.Move, float]] = []
        for mv in board.legal_moves:
            try:
                idx = move_to_index(board, mv)
            except ValueError:
                continue
            p = float(priors_all[idx].item())
            priors.append((mv, p))

        if not priors:
            return {}, float(value.item())

        priors.sort(key=lambda x: x[1], reverse=True)
        if self.root_prune_top_k > 0:
            priors = priors[: self.root_prune_top_k]
        if self.min_prior > 0:
            priors = [x for x in priors if x[1] >= self.min_prior] or priors[:1]

        total = sum(p for _, p in priors)
        if total <= 0:
            uniform = 1.0 / len(priors)
            return {m: uniform for m, _ in priors}, float(value.item())

        return {m: p / total for m, p in priors}, float(value.item())

    def _expand(self, node: Node, add_root_noise: bool = False) -> float:
        if node.board.is_game_over(claim_draw=True):
            return self._terminal_value(node.board)

        priors, value = self._evaluate(node.board)
        moves = list(priors.keys())

        # 🔥 REPETITION PENALTY
        if node.board.is_repetition(2):
            value -= 0.2

        if add_root_noise and moves:
            noise = torch.distributions.dirichlet.Dirichlet(
                torch.full((len(moves),), self.dirichlet_alpha, dtype=torch.float32)
            ).sample()
            for i, mv in enumerate(moves):
                priors[mv] = (1.0 - self.dirichlet_epsilon) * priors[mv] + self.dirichlet_epsilon * float(noise[i])

        for mv, prior in priors.items():
            nxt = node.board.copy(stack=False)
            nxt.push(mv)
            node.children[mv] = Node(board=nxt, prior=prior, parent=node, move=mv)

        return value

    def _select_child(self, node: Node) -> Node:
        sqrt_n = math.sqrt(max(1, node.visits))
        best_score = -float("inf")
        best_child = None

        for child in node.children.values():
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visits)

            # 🔥 discourage repetition-heavy paths
            penalty = -0.1 if child.board.is_repetition(2) else 0.0

            score = child.q + u + penalty

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("No child found during selection")
        return best_child

    def _backprop(self, path: list[Node], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def run(self, board: chess.Board, add_root_noise: bool = False) -> tuple[dict[chess.Move, int], dict[chess.Move, float]]:
        root = Node(board=board.copy(stack=False), prior=1.0)
        self._expand(root, add_root_noise=add_root_noise)

        for _ in range(self.simulations):
            node = root
            path = [node]

            while node.children:
                node = self._select_child(node)
                path.append(node)

            leaf_value = self._expand(node)
            self._backprop(path, leaf_value)

        visits = {mv: ch.visits for mv, ch in root.children.items()}
        priors = {mv: ch.prior for mv, ch in root.children.items()}
        return visits, priors


def choose_move(visits: dict[chess.Move, int], temperature: float = 1.0) -> chess.Move:
    if not visits:
        raise RuntimeError("No visit counts to choose from")

    moves = list(visits.keys())
    counts = torch.tensor([max(1, visits[mv]) for mv in moves], dtype=torch.float32)

    if temperature <= 1e-6:
        return moves[int(torch.argmax(counts).item())]

    probs = torch.pow(counts, 1.0 / temperature)
    probs = probs / probs.sum()
    idx = int(torch.multinomial(probs, num_samples=1).item())
    return moves[idx]


def visits_to_policy(board: chess.Board, visits: dict[chess.Move, int]) -> torch.Tensor:
    pi = torch.zeros(N_MOVES, dtype=torch.float32)
    if not visits:
        return pi

    total = float(sum(visits.values()))
    if total <= 0:
        total = 1.0

    for mv, cnt in visits.items():
        try:
            idx = move_to_index(board, mv)
        except ValueError:
            continue
        pi[idx] = float(cnt) / total

    s = float(pi.sum().item())
    if s > 0:
        pi /= s
    return pi