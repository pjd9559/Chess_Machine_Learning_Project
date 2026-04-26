# Project: Grandmaster Evolution & Positional Chess Engine

## 1. Project Objective
Build a Python-based Chess application using **Streamlit** that supports:
* **Human vs. Bot:** Play against any of the 5 specialized NN personalities.
* **Bot vs. Bot:** Pit different personalities against each other.
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

### Phase 2: Personality Reward Heuristics
When training or fine-tuning the models, use the following weighted metrics to define their "personalities":

| Personality | Primary Heuristic Focus | Technical Weights |
| :--- | :--- | :--- |
| **The Romantic (Pre-1980)** | Historical Imitation | Train solely on PGNs from pre-1980 GM datasets. High weight on King-side attacks. |
| **The Modernist (Post-1980)** | Efficiency & Precision | Train on post-1980 GM games + Stockfish self-play. High weight on "Engine-line" accuracy. |
| **The Architect** | **Spatial Advantage** | **Weighting:** Control of the 16 central squares + number of squares attacked in the opponent’s territory. |
| **The Tactician** | **Initiative & Activity** | **Weighting:** Mobility (number of legal moves available) + King safety of the opponent + forcing move frequency. |
| **The Strategist** | **Structural Integrity** | **Weighting:** Material balance (Standard values) - Penalties for doubled pawns, isolated pawns, or "holes" in the structure. |

### Phase 3: Data Pipeline
The agent must implement data loaders for:
* **Historical Games:** Filtered CSV/PGN from [Lumbras Gigabase](https://lumbrasgigabase.com/en/).
* **Strategic Games:** [LAION Strategic Dataset](https://huggingface.co/datasets/laion/strategic_game_chess).
* **Online GM Games:** [Kaggle GM Games](https://www.kaggle.com/datasets/dimitrioskourtikakis/gm-games-chesscom).

## 4. Specific Instructions for the Agent
1.  **Bitboard Evaluation:** For the "Architect" and "Strategist," calculate positional scores using bitmasks for specific squares (e.g., d4, e4, d5, e5).
2.  **State Management:** The chess board must be stored in `st.session_state.board`. Do not re-instantiate the board on every button click.
3.  **Bot Decision Loop:**
    * Generate all legal moves.
    * Pass each resulting state into the selected NN.
    * The bot chooses the move with the highest predicted "Win Probability" or "Reward Score."

## 5. Directory Structure
```text
├── app.py                # Main UI Entry point
├── src/
│   ├── engine.py         # Move validation and game loop
│   ├── personalities.py  # Model definitions and weight configurations
│   └── utils.py          # PGN parsing and board-to-tensor conversion
├── models/               # Pre-trained .pth or .h5 files
└── data/                 # Data loading scripts

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
