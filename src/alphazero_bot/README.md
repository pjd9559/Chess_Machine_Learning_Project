# Alphazero Chess Bot

This workspace implements an AlphaZero-style training loop:
1. Self-play with neural-guided MCTS
2. Supervised training on self-play targets `(state, visit-policy, game-outcome)`
3. Arena evaluation `candidate` vs `best`
4. Promotion when candidate exceeds threshold

## Files
- `model.py`: policy-value residual network
- `encoding.py`: board/action encoding (`4672` move head)
- `mcts.py`: MCTS + PUCT + root Dirichlet noise
- `self_play.py`: generates self-play games into `data/selfplay/*.pt`
- `train.py`: trains candidate from replay files
- `arena.py`: evaluates candidate against best and promotes if strong enough
- `pipeline.py`: runs iterative self-play/train/arena loop

## Install
From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r chess_bots/Alphazero/requirements.txt
```

## Quick Start (Single Iteration)

```bash
cd chess_bots/Alphazero

# 1) Generate self-play data
python self_play.py --games 12 --simulations 96 --device cpu

# 2) Train candidate from self-play
python train.py --epochs 2 --batch-size 128 --device cpu

# 3) Arena: candidate vs best (auto-inits best on first run)
python arena.py --games 8 --simulations 96 --device cpu
```

## Full Loop

```bash
cd chess_bots/Alphazero
python pipeline.py \
  --iterations 5 \
  --games-per-iter 16 \
  --selfplay-simulations 96 \
  --train-epochs 2 \
  --arena-games 10 \
  --arena-simulations 96 \
  --device cpu
```

## Notes
- For GPU, replace `--device cpu` with `--device cuda`.
- Models are stored under `checkpoints/` (`best.pt`, `candidate.pt`).
- Self-play files accumulate in `data/selfplay/`; use `--max-files` in `train.py` to cap replay window.
