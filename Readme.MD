# Project: Grandmaster Evolution & Positional Chess Engine

> [!IMPORTANT]
> The final model files `post_2010_final_v6.pt` and `pre_2000_final_v6.pt` are each over 100 MB, so they cannot be pushed to GitHub in this repository.
> Download both models from Dropbox:
> https://www.dropbox.com/scl/fo/8qomt37pwz8q2vw1vx4o5/ADJngMPKYDyf40xrH2hUI7I?rlkey=cyz6htx9ipxizz77e5twyt62q&st=zg426sev&dl=0
> After downloading, move the two files into `src/models`.

## 1. Project Objective
Build a Python-based Chess application using **Streamlit** that supports:
* **Human vs. Bot:** Play against selected neural model checkpoints.
* **Bot vs. Bot:** Pit two neural model checkpoints against each other.
* **Tournament Mode:** A round-robin style competition to rank the models.

## 2. Technical Requirements
* **Frontend:** Streamlit (use `st.session_state` to maintain the board state across reruns).
* **Backend Logic:** `python-chess` library for rule enforcement (En Passant, Promotion, Castling).
* **Models:** PyTorch or TensorFlow for Neural Network inference.
* **Input Representation:** Convert board states to 8x8x12 tensors (one 8x8 layer per piece type for each color).

## 3. Implementation Phases

### Phase 1: Core Engine & UI
1.  Initialize a board using `chess.Board()`.
2.  Render the board in Streamlit (using SVG or `streamlit-chess`).
3.  Implement a move-handler that validates legal moves before updating the state.
4.  Handle Pawn Promotion with a UI pop-up/selection.

### Phase 2: Model-Centric Training Strategy
Train and compare neural checkpoints directly instead of style-labeled personalities.

Current app-facing model lineup:
- **Pre-2000 High Elo** (`pre_2000_ultra_final_nn`)
- **Post-2010 High Elo** (`post_2010_ultra_final_nn`)

Training emphasis:
- Strong self-play search targets (MCTS visits)
- Stable policy/value updates
- Arena-based promotion (`candidate` vs `best`)

### Phase 3: Data Pipeline
The agent must implement data loaders for:
* **Historical Games:** Filtered CSV/PGN from [Lumbras Gigabase](https://lumbrasgigabase.com/en/).
* **Strategic Games:** [LAION Strategic Dataset](https://huggingface.co/datasets/laion/strategic_game_chess).
* **Online GM Games:** [Kaggle GM Games](https://www.kaggle.com/datasets/dimitrioskourtikakis/gm-games-chesscom).

## 4. Specific Instructions for the Agent
1.  **State Management:** The chess board must be stored in `st.session_state.board`. Do not re-instantiate the board on every button click.
2.  **Bot Decision Loop:**
    * Generate all legal moves.
    * Pass each resulting state into the selected NN.
    * The bot chooses the move with the highest predicted "Win Probability" or "Reward Score."

## 5. Tests
The Streamlit app includes a `Tests` mode for browsing the project evaluation workspace.

Current test assets:
- `tests/eval_m8n2.py`: lightweight evaluation runner for model-generation accuracy checks.
- `tests/256/`: saved evaluation outputs grouped by model family/version.
- `tests/README.md`: notes for the test workspace.
- `m8n1.csv`: mate-in-one dataset sourced from Kaggle: https://www.kaggle.com/datasets/ancientaxe/mate-in-one-chess?resource=download
- `m8n2.txt`, `m8n3.txt`, `m8n4.txt`: puzzle sets sourced from https://wtharvey.com/

These artifacts are also referenced by the app when showing lightweight puzzle evaluation summaries.

## 6. Directory Structure
```text
├── app.py                # Main UI Entry point
├── src/
│   ├── engine.py         # Move validation and game loop
│   ├── personalities.py  # Registry used by app mode selectors
│   └── utils.py          # PGN parsing and board-to-tensor conversion
├── models/               # Pre-trained .pth or .h5 files
├── data/                 # Data loading scripts
└── tests/
    ├── eval_m8n2.py      # Evaluation runner for test suites
    ├── 256/              # Saved evaluation outputs by model/version
    └── README.md         # Notes for the test workspace
```

## Reports
The Streamlit app now includes a `Reports` mode for course-style ML analysis of the two neural chess bots.

Available report tabs:
- `Embedding Visualization`: extracts penultimate-layer embeddings, then shows PCA and t-SNE plots with chess metadata.
- `Cluster Analysis`: runs k-means from scratch on the embeddings and summarizes cluster themes, sizes, and per-model behavior.
- `Style Behavior`: simulates head-to-head games and compares sacrifice rate, king pressure, center control, captures, and related style metrics.
- `Style Classification`: trains a lightweight classifier to predict whether `pre_2000` or `post_2020` made a move.

How to use:
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the app:
   `.venv/bin/streamlit run app.py`
3. Open the `Reports` mode from the sidebar.
4. Use each tab’s `Run analysis` button to generate cached report artifacts and visualizations.

Generated report artifacts:
- `generated_reports/` stores reusable analytics outputs such as style-game logs and cluster labels.
- `tests/*_results.txt` still power the lightweight puzzle evaluation summaries shown in the `Tests` mode.

## 7. AlphaZero Training Plan (Current Project Workflow)

### Goal
Train stronger chess checkpoints through an iterative loop where:
1. A current champion checkpoint (`best`) generates self-play games.
2. A new model (`candidate`) is trained on those games.
3. `candidate` is validated against `best`.
4. `candidate` is promoted only if it performs better.

This creates a controlled improvement cycle instead of replacing models blindly.

### End-to-End Process: Pretrain -> Train -> Validate -> Promote

1. **Pretrain (bootstrap)**
- Script: `src/alphazero_bot/pretrain.py`
- Purpose: initialize a reasonable policy-value network from historical tensor chunks.
- Output: a starting checkpoint (for example in `src/models/` or `src/alphazero_bot/checkpoints/`).

2. **Self-play generation**
- Script: `src/alphazero_bot/self_play.py`
- Purpose: generate training targets from MCTS-guided games using the current `best` checkpoint.
- Output: replay files under `data/selfplay/<iteration_folder>/*.pt` containing:
  - `states` tensor
  - `policies` (visit-based target distribution)
  - `values` (game outcome from side-to-move perspective)
  - metadata such as `result` and `plies`

3. **Train candidate**
- Script: `src/alphazero_bot/train.py`
- Purpose: load self-play replay files, start from `best` weights, and optimize a new `candidate`.
- Output: `candidate` checkpoint with updated `model_state`, optimizer state, and training config.

4. **Validate (arena)**
- Script: `src/alphazero_bot/arena.py`
- Purpose: play `candidate` vs `best` under fixed evaluation settings.
- Output: win/draw/loss score; optional automatic promotion if threshold is met.

5. **Promote**
- If `candidate` beats threshold, copy `candidate -> best`.
- If it fails, keep `best` unchanged and run the next iteration with better data/settings.

### Core Algorithms Used

1. **Policy-Value Network (`ChessNet`)**
- File: `src/alphazero_bot/model.py`
- A residual CNN with:
  - Policy head: predicts move logits over the 4672-action space.
  - Value head: predicts scalar game value in `[-1, 1]`.

2. **MCTS with PUCT**
- File: `src/alphazero_bot/mcts.py`
- Search expands legal moves and balances:
  - exploitation (node value `Q`)
  - exploration (policy prior + visit bonus via PUCT)
- MCTS visit counts become policy targets for training.

3. **Board/Move Encoding**
- File: `src/alphazero_bot/encoding.py`
- Converts chess positions to model tensors.
- Maps legal moves to fixed policy indices and back.

4. **Supervised update from self-play**
- File: `src/alphazero_bot/train.py`
- Trains model to match:
  - policy targets from MCTS visits
  - value targets from final game outcomes
- Includes optimizer + gradient clipping for stability.

### Loss Functions (as implemented in training)

Training optimizes a combined objective:
- policy loss (KL divergence to target policy)
- value loss (MSE to outcome target)
- optional entropy regularization term (depending on training script version)

Conceptually:
- `L_total = L_policy + 0.5 * L_value - alpha * entropy`

Where:
- `L_policy` encourages matching MCTS move distributions.
- `L_value` encourages accurate game-outcome estimation.
- `entropy` can keep policy less collapsed early in training.

### Main Functions and Scripts

- `src/alphazero_bot/self_play.py`
  - `load_model(...)`
  - `play_one_game_dual(...)`
  - `main()` entrypoint for replay generation
- `src/alphazero_bot/train.py`
  - `load_training_tensors(...)`
  - `policy_loss_from_targets(...)`
  - `main()` training loop and checkpoint save
- `src/alphazero_bot/arena.py`
  - `load_model(...)`
  - `play_game(...)`
  - `main()` evaluation/promotion logic
- `src/alphazero_bot/mcts.py`
  - `AlphaZeroMCTS.run(...)`
  - `choose_move(...)`
  - `visits_to_policy(...)`

### Project Plan for Reliable Improvement

1. Create a fresh self-play folder per iteration (avoid mixing stale data).
2. Generate enough games with consistent MCTS settings.
3. Remove low-quality capped games if needed.
4. Train conservatively (small updates first).
5. Validate on a sufficiently large arena sample.
6. Promote only on clear improvement.
7. Repeat and track each checkpoint lineage.
