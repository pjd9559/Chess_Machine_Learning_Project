"""
app.py — Grandmaster Evolution & Positional Chess Engine
Main Streamlit UI entry point.

Modes:
  • Human vs Bot — play against any of 5 NN personalities
  • Bot vs Bot   — watch two personalities play each other
  • Tournament   — round-robin competition with live standings
  • Reports      — ML analytics and evaluation reports
  • Tests        — view the project test workspace
"""

import streamlit as st
import altair as alt
import chess
import time
import itertools
import base64
import re
from pathlib import Path
from src.engine import GameEngine
from src.personalities import ALL_PERSONALITIES, get_personality
from src.utils import render_board_svg
import torch
from src.alphazero_bot.model import ChessNet

# ── Load pretrained models ─────────────────────────────

@st.cache_resource
def load_model(path):
    model = ChessNet()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

BASE_DIR = Path(__file__).resolve().parent
TESTS_DIR = BASE_DIR / "tests"
TESTS_RESULTS_DIR = TESTS_DIR / "256"

PRE_2000_PATH = BASE_DIR / "src" / "models" / "pre_2000.pt"
POST_2020_PATH = BASE_DIR / "src" / "models" / "post_2020.pt"
PRE_2000_V6_PATH = BASE_DIR / "src" / "models" / "pre_2000_final_v6.pt"
POST_2010_V6_PATH = BASE_DIR / "src" / "models" / "post_2010_final_v6.pt"
PRE_2000_ULTRA_FINAL_PATH = BASE_DIR / "src" / "models" / "pre_2000_ultra_final.pt"
POST_2010_ULTRA_FINAL_PATH = BASE_DIR / "src" / "models" / "post_2010_ultra_final.pt"


PRE_2000_MODEL = load_model(PRE_2000_PATH)
POST_2020_MODEL = load_model(POST_2020_PATH)
PRE_2000_V6_MODEL = load_model(PRE_2000_V6_PATH)
POST_2010_V6_MODEL = load_model(POST_2010_V6_PATH)
PRE_2000_ULTRA_FINAL_MODEL = load_model(PRE_2000_ULTRA_FINAL_PATH)
POST_2010_ULTRA_FINAL_MODEL = load_model(POST_2010_ULTRA_FINAL_PATH)


# ── Inject new NN personalities ─────────────────────────

ALL_PERSONALITIES["pre_2000_nn"] = type("P", (), {
    "name": "Pre-2000 Bot",
    "icon": "♜",
    "description": "Trained on classical-era games (up to 1989)",
    "color": "#3498db",
    "model": PRE_2000_MODEL
})()

ALL_PERSONALITIES["post_2020_nn"] = type("P", (), {
    "name": "Post-2020 Bot",
    "icon": "🔥",
    "description": "Trained on modern engine-influenced games",
    "color": "#e74c3c",
    "model": POST_2020_MODEL
})()

ALL_PERSONALITIES["pre_2000_v6_nn"] = type("P", (), {
    "name": "Pre-2000 v6 Bot",
    "icon": "♜",
    "description": "Version 6 checkpoint trained on classical-era games",
    "color": "#5dade2",
    "model": PRE_2000_V6_MODEL
})()

ALL_PERSONALITIES["post_2010_v6_nn"] = type("P", (), {
    "name": "Post-2010 v6 Bot",
    "icon": "🔥",
    "description": "Version 6 checkpoint trained on modern games",
    "color": "#ff6b6b",
    "model": POST_2010_V6_MODEL
})()

ALL_PERSONALITIES["pre_2000_ultra_final_nn"] = type("P", (), {
    "name": "Pre-2000 Final Bot",
    "icon": "♜",
    "description": "Latest final checkpoint trained on classical-era games",
    "color": "#3498db",
    "model": PRE_2000_ULTRA_FINAL_MODEL
})()

ALL_PERSONALITIES["post_2010_ultra_final_nn"] = type("P", (), {
    "name": "Post-2010 Final Bot",
    "icon": "🔥",
    "description": "Latest final checkpoint trained on modern games",
    "color": "#e74c3c",
    "model": POST_2010_ULTRA_FINAL_MODEL
})()

# ═══════════════════════════════════════════════════════════════════════════════
#  Page config & CSS
# ═══════════════════════════════════════════════════════════════════════════════



st.set_page_config(
    page_title="Grandmaster Evolution ♟",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    color: #e0e0e0;
}

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c9d1d9;
}

/* ── Glass cards ────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.glass-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.12);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* ── Status badge ───────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.status-white  { background: linear-gradient(135deg,#f8f9fa,#dee2e6); color: #212529; }
.status-black  { background: linear-gradient(135deg,#343a40,#212529); color: #f8f9fa; }
.status-over   { background: linear-gradient(135deg,#e74c3c,#c0392b); color: #fff;    }
.status-check  { background: linear-gradient(135deg,#f39c12,#e67e22); color: #fff;    }

/* ── Move history ───────────────────────────────────────────── */
.move-history {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    padding: 0.75rem 1rem;
    background: rgba(0,0,0,0.25);
    border-radius: 12px;
    max-height: 350px;
    overflow-y: auto;
    border: 1px solid rgba(255,255,255,0.05);
}

/* ── Tournament table ───────────────────────────────────────── */
.tournament-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.tournament-table th {
    background: rgba(255,255,255,0.06);
    color: #c9d1d9;
    padding: 0.7rem 1rem;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.tournament-table td {
    padding: 0.6rem 1rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    font-size: 0.9rem;
}
.tournament-table tr:hover td {
    background: rgba(255,255,255,0.03);
}

/* ── Buttons ────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.25s ease;
    border: 1px solid rgba(255,255,255,0.1);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

/* ── Form controls ──────────────────────────────────────────── */
label[data-testid="stWidgetLabel"] p,
div[data-testid="stCaptionContainer"] p {
    color: #1f2937 !important;
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.92) !important;
    color: #111827 !important;
}

div[data-baseweb="select"] * {
    color: #111827 !important;
}

div[data-testid="stSlider"] [data-baseweb="slider"] + div p,
div[data-testid="stSlider"] [role="slider"] {
    color: #111827 !important;
}

/* ── Square buttons (chess board) ───────────────────────────── */
.square-btn {
    width: 56px; height: 56px;
    border: none; cursor: pointer;
    font-size: 1.6rem;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s ease;
    margin: 0; padding: 0;
    border-radius: 2px;
}
.square-btn:hover {
    filter: brightness(1.2);
    transform: scale(1.05);
}

/* ── Title gradient ─────────────────────────────────────────── */
.title-gradient {
    background: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
}

/* ── Personality card ───────────────────────────────────────── */
.personality-card {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
    transition: all 0.3s ease;
}
.personality-card:hover {
    background: rgba(255,255,255,0.07);
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.15);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Session-state initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "board": chess.Board(),
        "engine": GameEngine(),
        "mode": "Human vs Bot",
        "white_personality": "romantic",
        "black_personality": "modernist",
        "human_color": "white",
        "selected_square": None,
        "legal_targets": [],
        "flipped": False,
        "game_over": False,
        "bot_vs_bot_running": False,
        "tournament_running": False,
        "tournament_results": {},
        "tournament_games": [],
        "tournament_current_game": "",
        "promotion_pending": None,
        "human_has_moved": False,
        "move_delay": 0.8,
        "tournament_pair": "v6",
        "tournament_num_games": 8,
        "human_mcts_simulations": 256,
        "bot_vs_bot_mcts_simulations": 256,
        "tournament_mcts_simulations": 256,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def new_game():
    """Reset the board and engine for a fresh game."""
    st.session_state.board = chess.Board()
    st.session_state.engine = GameEngine()
    st.session_state.selected_square = None
    st.session_state.legal_targets = []
    st.session_state.game_over = False
    st.session_state.bot_vs_bot_running = False
    st.session_state.promotion_pending = None
    st.session_state.human_has_moved = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="title-gradient">♟ Grandmaster Evolution</div>', unsafe_allow_html=True)
    st.markdown("*Positional Chess Engine with 5 NN Personalities*")
    st.markdown("---")

    # Mode selector
    mode = st.radio(
        "🎮 **Game Mode**",
        ["Human vs Bot", "Bot vs Bot", "Tournament", "Reports", "Tests"],
        key="mode_radio",
        index=["Human vs Bot", "Bot vs Bot", "Tournament", "Reports", "Tests"].index(st.session_state.mode),
    )
    st.session_state.mode = mode

    st.markdown("---")

    # Personality info cards
    personality_options = {
        k: f"{v.icon} {v.name}" for k, v in ALL_PERSONALITIES.items()
    }
    nn_personality_keys = [
        "pre_2000_nn",
        "post_2020_nn",
        "pre_2000_v6_nn",
        "post_2010_v6_nn",
    ]
    nn_personality_options = {k: personality_options[k] for k in nn_personality_keys}

    if mode == "Tournament":
        st.markdown("### 🏆 Tournament Settings")
        current_pair = st.session_state.get("tournament_pair", "v6")
        if current_pair == "v6":
            st.markdown("Direct matchup between the pre-2000 v6 and post-2010 v6 bots.")
        else:
            st.markdown("Direct matchup between the pre-2000 and post-2020 bots.")
        st.session_state.move_delay = st.slider(
            "⏱ Move delay (seconds)", 0.1, 3.0, 0.5, 0.1, key="tourn_delay_slider"
        )

    elif mode == "Reports":
        st.markdown("### 📚 Reports")
        st.markdown("Run ML analytics, clustering, style, and classification studies.")

    elif mode == "Tests":
        st.markdown("### 🧪 Tests")
        st.markdown("Browse the test workspace for this project.")

# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: Render the chessboard as an interactive SVG
# ═══════════════════════════════════════════════════════════════════════════════

PIECE_SYMBOLS = {
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.KING,   chess.WHITE): "♔",
    (chess.PAWN,   chess.BLACK): "♟",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.KING,   chess.BLACK): "♚",
}

LIGHT_SQ = "#ffcf9f"
DARK_SQ  = "#d18b47"
SEL_SQ   = "#4db6ff"
TARGET_SQ = "#5fe36f"
LAST_MOVE_SQ = "#d7c14a"


def render_interactive_board(board, engine, disabled=False):
    """Render the board using Streamlit columns with clickable buttons."""
    flipped = st.session_state.flipped
    selected = st.session_state.selected_square
    targets  = st.session_state.legal_targets
    last_mv  = engine.last_move

    st.markdown(
        """
        <style>
        .interactive-board-shell {
            background: rgba(32, 28, 24, 0.9);
            border-radius: 12px;
            padding: 12px 12px 8px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
            width: fit-content;
            margin: 0 auto;
        }
        .interactive-board-shell .board-label {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 56px;
            color: rgba(255,255,255,0.85);
            font-weight: 700;
            font-size: 0.9rem;
        }
        .interactive-board-shell .file-label {
            text-align: center;
            color: rgba(255,255,255,0.85);
            font-weight: 700;
            font-size: 0.9rem;
            padding-top: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="interactive-board-shell">', unsafe_allow_html=True)

    ranks = range(7, -1, -1) if not flipped else range(8)
    files = range(8) if not flipped else range(7, -1, -1)

    rank_labels = list(ranks)
    file_labels = list(files)

    for row_idx, rank in enumerate(rank_labels):
        cols = st.columns([0.3] + [1]*8 + [0.3])
        # Rank label
        cols[0].markdown(
            f"<div class='board-label'>"
            f"{rank+1}</div>",
            unsafe_allow_html=True
        )

        for col_idx, file in enumerate(file_labels):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            symbol = PIECE_SYMBOLS.get((piece.piece_type, piece.color), "") if piece else ""

            # Determine square color
            is_light = (rank + file) % 2 == 1
            bg = LIGHT_SQ if is_light else DARK_SQ

            if sq == selected:
                bg = SEL_SQ
            elif targets and sq in targets:
                bg = TARGET_SQ
            elif last_mv and (sq == last_mv.from_square or sq == last_mv.to_square):
                bg = LAST_MOVE_SQ

            text_color = "#111111" if is_light else "#050505"
            if sq == selected or (targets and sq in targets):
                text_color = "#ffffff"

            with cols[col_idx + 1]:
                btn_label = symbol if symbol else " "
                if st.button(
                    btn_label,
                    key=f"sq_{sq}",
                    disabled=disabled,
                    use_container_width=True,
                    help=chess.square_name(sq),
                ):
                    handle_square_click(sq, engine)

                # Apply background color styling
                st.markdown(
                    f"""<style>
                    div[data-testid="stHorizontalBlock"]:nth-child({row_idx + 2})
                    div[data-testid="column"]:nth-child({col_idx + 2}) button {{
                        background-color: {bg} !important;
                        color: {text_color} !important;
                        font-size: 2rem !important;
                        font-weight: 700 !important;
                        height: 58px !important;
                        min-height: 58px !important;
                        padding: 0 !important;
                        border-radius: 0 !important;
                        border: none !important;
                        box-shadow: none !important;
                    }}
                    </style>""",
                    unsafe_allow_html=True,
                )

    # File labels
    footer_cols = st.columns([0.3] + [1]*8 + [0.3])
    file_chars = "abcdefgh"
    for col_idx, file in enumerate(file_labels):
        footer_cols[col_idx + 1].markdown(
            f"<div class='file-label'>{file_chars[file]}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: Alternative SVG board rendering (non-interactive)
# ═══════════════════════════════════════════════════════════════════════════════

def render_svg_board(board, engine, size=480):
    """Render board as SVG image (used in Bot vs Bot and Tournament modes)."""
    svg = render_board_svg(
        board,
        last_move=engine.last_move,
        size=size,
        flipped=st.session_state.flipped,
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    st.markdown(
        f'<div style="display:flex;justify-content:center">'
        f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}"'
        f' style="border-radius:8px;box-shadow:0 4px 24px rgba(0,0,0,0.4)">'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Click handling for Human vs Bot
# ═══════════════════════════════════════════════════════════════════════════════

def handle_square_click(sq, engine):
    """Process a square click for piece selection and move making."""
    board = engine.board
    selected = st.session_state.selected_square

    if st.session_state.game_over or board.is_game_over():
        return

    # Determine if it is the human's turn
    human_is_white = st.session_state.human_color == "white"
    if (board.turn == chess.WHITE) != human_is_white:
        return   # not human's turn

    if selected is None:
        # First click — select a piece
        piece = board.piece_at(sq)
        if piece and piece.color == board.turn:
            legal = engine.get_legal_moves_for_square(sq)
            if legal:
                st.session_state.selected_square = sq
                st.session_state.legal_targets = [m.to_square for m in legal]
                st.rerun()
    else:
        if sq == selected:
            # Deselect
            st.session_state.selected_square = None
            st.session_state.legal_targets = []
            st.rerun()
            return

        # Check if clicking another own piece → reselect
        piece = board.piece_at(sq)
        if piece and piece.color == board.turn:
            legal = engine.get_legal_moves_for_square(sq)
            if legal:
                st.session_state.selected_square = sq
                st.session_state.legal_targets = [m.to_square for m in legal]
                st.rerun()
            return

        # Attempt to make the move
        uci = chess.square_name(selected) + chess.square_name(sq)
        from_piece = board.piece_at(selected)

        # Check for promotion
        if (from_piece and from_piece.piece_type == chess.PAWN and
            ((from_piece.color == chess.WHITE and chess.square_rank(sq) == 7) or
             (from_piece.color == chess.BLACK and chess.square_rank(sq) == 0))):
            st.session_state.promotion_pending = uci
            st.session_state.selected_square = None
            st.session_state.legal_targets = []
            st.rerun()
            return

        success = engine.make_move(uci)
        st.session_state.selected_square = None
        st.session_state.legal_targets = []

        if success:
            # Sync board reference
            st.session_state.board = engine.board
            st.session_state.human_has_moved = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  Move history panel
# ═══════════════════════════════════════════════════════════════════════════════

def render_move_history(engine):
    """Display the move history in a formatted panel."""
    moves = [move.uci() for move in engine.board.move_stack]
    if not moves:
        st.markdown("*No moves yet.*")
        return

    lines = []
    for i in range(0, len(moves), 2):
        num = i // 2 + 1
        white_move = moves[i]
        black_move = moves[i + 1] if i + 1 < len(moves) else "..."
        lines.append(f"{num}. {white_move}  {black_move}")

    st.markdown(
        f'<div class="move-history">{"<br>".join(lines)}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Game status display
# ═══════════════════════════════════════════════════════════════════════════════

def render_game_status(engine):
    """Show colored status badge."""
    status = engine.get_game_status()

    if status["is_over"] or st.session_state.game_over:
        reason = status["reason"] if status["is_over"] else "resignation"
        result = status["result"] if status["is_over"] else (
            "0-1" if st.session_state.human_color == "white" else "1-0"
        )
        st.markdown(
            f'<span class="status-badge status-over">Game Over — {result} ({reason})</span>',
            unsafe_allow_html=True,
        )
        return True
    elif status["in_check"]:
        st.markdown(
            f'<span class="status-badge status-check">⚠️ Check!</span>',
            unsafe_allow_html=True,
        )
    else:
        cls = "status-white" if status["turn"] == "white" else "status-black"
        label = "⬜ White" if status["turn"] == "white" else "⬛ Black"
        st.markdown(
            f'<span class="status-badge {cls}">{label} to move</span>',
            unsafe_allow_html=True,
        )
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Human move controls
# ═══════════════════════════════════════════════════════════════════════════════

def get_human_move_options(engine):
    """Build alphabetically sorted legal-move options for the human player."""
    board = engine.board
    move_options = []

    for move in board.legal_moves:
        uci = move.uci()
        move_options.append((f"{uci}  |  {board.san(move)}", uci))

    return sorted(move_options, key=lambda option: option[1])


def render_human_move_controls(engine, human_is_white, disabled=False):
    """Render a compact sorted move picker for Human vs Bot."""
    board = engine.board
    human_to_move = (board.turn == chess.WHITE) == human_is_white

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Your Move")

    if disabled or board.is_game_over():
        st.markdown("*This game is finished.*")
    elif not human_to_move:
        st.markdown("*Waiting for the bot to move...*")
    else:
        move_options = get_human_move_options(engine)
        move_lookup = {uci: label for label, uci in move_options}

        selected_uci = st.selectbox(
            "Choose a legal move",
            [uci for _, uci in move_options],
            index=0,
            format_func=lambda uci: move_lookup[uci],
            key=f"human_move_select_{len(board.move_stack)}_{board.fen()}",
        )

        if st.button("♟ Play Move", use_container_width=True, key="human_play_move"):
            if engine.make_move(selected_uci):
                st.session_state.board = engine.board
                st.session_state.selected_square = None
                st.session_state.legal_targets = []
                st.session_state.promotion_pending = None
                st.session_state.human_has_moved = True
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ██  HUMAN VS BOT  ██
# ═══════════════════════════════════════════════════════════════════════════════

def page_human_vs_bot():
    engine = st.session_state.engine
    board  = engine.board
    controls_locked = st.session_state.get("human_has_moved", False)
    nn_personality_keys = [
        "pre_2000_nn",
        "post_2020_nn",
        "pre_2000_v6_nn",
        "post_2010_v6_nn",
    ]
    if st.session_state.get("black_personality") not in nn_personality_keys:
        st.session_state.black_personality = "pre_2000_v6_nn"
    if st.session_state.get("white_personality") not in nn_personality_keys:
        st.session_state.white_personality = "pre_2000_v6_nn"

    # Header
    human_is_white = st.session_state.human_color == "white"
    bot_key = "black_personality" if human_is_white else "white_personality"
    bot_p = get_personality(st.session_state[bot_key])

    st.markdown(
        f'<div class="glass-card">'
        f'<h2 style="margin:0">⚔️ Human vs {bot_p.icon} {bot_p.name}</h2>'
        f'<p style="margin:0.3rem 0 0;opacity:0.7">{bot_p.description}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Layout
    board_col, info_col = st.columns([3, 1.2])

    with board_col:
        game_over = render_game_status(engine)
        render_svg_board(board, engine, size=480)
        render_human_move_controls(engine, human_is_white, disabled=game_over)

    with info_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📜 Move History")
        render_move_history(engine)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Controls")
        previous_human_color = st.session_state.human_color
        if "human_color_sel" not in st.session_state:
            st.session_state.human_color_sel = st.session_state.human_color
        human_color = st.selectbox(
            "Play as",
            ["white", "black"],
            key="human_color_sel",
            disabled=controls_locked,
        )
        if not controls_locked and human_color != previous_human_color:
            st.session_state.human_color = human_color
            next_bot_key = "black_personality" if human_color == "white" else "white_personality"
            st.session_state.bot_personality_sel = st.session_state.get(next_bot_key, "pre_2000_v6_nn")
            st.rerun()
        st.session_state.human_color = human_color
        human_is_white = human_color == "white"
        bot_key = "black_personality" if human_is_white else "white_personality"
        current_bot = st.session_state.get(bot_key, "pre_2000_v6_nn")
        if "bot_personality_sel" not in st.session_state:
            st.session_state.bot_personality_sel = current_bot
        selected_bot = st.selectbox(
            "Bot",
            nn_personality_keys,
            format_func=lambda k: f"{get_personality(k).icon} {get_personality(k).name}",
            key="bot_personality_sel",
            disabled=controls_locked,
        )
        st.session_state[bot_key] = selected_bot
        if controls_locked:
            st.caption("Start a new game to change your color or opponent.")
        st.session_state.human_mcts_simulations = st.slider(
            "MCTS simulations", 1, 512, st.session_state.human_mcts_simulations, 1, key="human_mcts_slider"
        )
        p = get_personality(selected_bot)
        st.markdown(
            f'<div class="personality-card" style="border-color:{p.color}">'
            f"<strong>{p.icon} {p.name}</strong><br>"
            f"<small>{p.description}</small></div>",
            unsafe_allow_html=True,
        )
        primary_ctrl_cols = st.columns(2)
        with primary_ctrl_cols[0]:
            if st.button("🆕 New Game", use_container_width=True, key="human_new_game_main"):
                new_game()
                st.rerun()
        with primary_ctrl_cols[1]:
            if st.button("🔃 Flip Board", use_container_width=True, key="human_flip_board_main"):
                st.session_state.flipped = not st.session_state.flipped
                st.rerun()
        secondary_ctrl_cols = st.columns(2)
        with secondary_ctrl_cols[0]:
            if st.button("↩️ Undo", use_container_width=True, key="human_undo_main"):
                engine.undo_two_moves()
                st.session_state.selected_square = None
                st.session_state.legal_targets = []
                st.rerun()
        with secondary_ctrl_cols[1]:
            if st.button("🏳 Resign", use_container_width=True, key="human_resign_main"):
                st.session_state.game_over = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Material count
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Material")
        for color_name, color in [("White", chess.WHITE), ("Black", chess.BLACK)]:
            pieces = []
            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
                count = len(board.pieces(pt, color))
                sym = PIECE_SYMBOLS[(pt, color)]
                if count > 0:
                    pieces.append(f"{sym}×{count}")
            st.markdown(f"**{color_name}:** {' '.join(pieces)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Bot auto-move (if it's the bot's turn)
    if not game_over and not st.session_state.promotion_pending:
        is_bot_turn = (board.turn == chess.WHITE) != human_is_white
        if is_bot_turn and not board.is_game_over():
            with st.spinner(f"{bot_p.icon} {bot_p.name} is thinking..."):
                time.sleep(0.5)
                engine.bot_move(st.session_state[bot_key], simulations=st.session_state.human_mcts_simulations)
                st.session_state.board = engine.board
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  ██  BOT VS BOT  ██
# ═══════════════════════════════════════════════════════════════════════════════

def page_bot_vs_bot():
    engine = st.session_state.engine
    board  = engine.board
    nn_personality_keys = [
        "pre_2000_nn",
        "post_2020_nn",
        "pre_2000_v6_nn",
        "post_2010_v6_nn",
    ]
    if not st.session_state.get("bot_vs_bot_defaults_initialized"):
        st.session_state.white_personality = "pre_2000_v6_nn"
        st.session_state.black_personality = "post_2010_v6_nn"
        st.session_state.white_bot_sel = "pre_2000_v6_nn"
        st.session_state.black_bot_sel = "post_2010_v6_nn"
        st.session_state.bot_vs_bot_defaults_initialized = True
    if st.session_state.get("white_personality") not in nn_personality_keys:
        st.session_state.white_personality = "pre_2000_v6_nn"
    if st.session_state.get("black_personality") not in nn_personality_keys:
        st.session_state.black_personality = "post_2010_v6_nn"
    if st.session_state.get("white_bot_sel") not in nn_personality_keys:
        st.session_state.white_bot_sel = st.session_state.white_personality
    if st.session_state.get("black_bot_sel") not in nn_personality_keys:
        st.session_state.black_bot_sel = st.session_state.black_personality

    w_p = get_personality(st.session_state.white_personality)
    b_p = get_personality(st.session_state.black_personality)

    st.markdown(
        f'<div class="glass-card">'
        f'<h2 style="margin:0">{w_p.icon} {w_p.name}  vs  {b_p.icon} {b_p.name}</h2>'
        f'<p style="margin:0.3rem 0 0;opacity:0.7">Bot vs Bot — Watch two personalities battle!</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    board_col, info_col = st.columns([3, 1.2])

    with board_col:
        game_over = render_game_status(engine)
        render_svg_board(board, engine, size=480)

    with info_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📜 Move History")
        render_move_history(engine)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Controls")
        white_sel = st.selectbox(
            "White bot",
            nn_personality_keys,
            format_func=lambda k: f"{get_personality(k).icon} {get_personality(k).name}",
            key="white_bot_sel",
            index=nn_personality_keys.index(st.session_state.white_personality),
        )
        st.session_state.white_personality = white_sel
        black_sel = st.selectbox(
            "Black bot",
            nn_personality_keys,
            format_func=lambda k: f"{get_personality(k).icon} {get_personality(k).name}",
            key="black_bot_sel",
            index=nn_personality_keys.index(st.session_state.black_personality),
        )
        st.session_state.black_personality = black_sel
        st.session_state.bot_vs_bot_mcts_simulations = st.slider(
            "MCTS simulations", 1, 512, st.session_state.bot_vs_bot_mcts_simulations, 1, key="bvb_mcts_slider"
        )
        ctrl_cols = st.columns(3)
        with ctrl_cols[0]:
            if st.button("▶️ Play / Resume", use_container_width=True, key="bvb_play"):
                st.session_state.bot_vs_bot_running = True
                st.rerun()
        with ctrl_cols[1]:
            if st.button("⏸ Pause", use_container_width=True, key="bvb_pause"):
                st.session_state.bot_vs_bot_running = False
                st.rerun()
        with ctrl_cols[2]:
            if st.button("⏭ Step", use_container_width=True, key="bvb_step"):
                if not board.is_game_over():
                    p_key = st.session_state.white_personality if board.turn == chess.WHITE else st.session_state.black_personality
                    engine.bot_move(p_key, simulations=st.session_state.bot_vs_bot_mcts_simulations)
                    st.session_state.board = engine.board
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Auto-play loop
    if st.session_state.bot_vs_bot_running and not board.is_game_over():
        p_key = st.session_state.white_personality if board.turn == chess.WHITE else st.session_state.black_personality
        p = get_personality(p_key)
        with st.spinner(f"{p.icon} {p.name} is thinking..."):
            engine.bot_move(p_key, simulations=st.session_state.bot_vs_bot_mcts_simulations)
            st.session_state.board = engine.board
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  ██  TOURNAMENT MODE  ██
# ═══════════════════════════════════════════════════════════════════════════════

def page_tournament():
    tournament_pairs = {
        "old": ("pre_2000_nn", "post_2020_nn"),
        "v6": ("pre_2000_v6_nn", "post_2010_v6_nn"),
    }
    tournament_pair_options = {
        "v6": "V6 Models",
        "old": "Old Models",
    }
    pair_key = st.radio(
        "Model Pair",
        list(tournament_pair_options),
        index=list(tournament_pair_options).index(st.session_state.get("tournament_pair", "v6")),
        format_func=lambda key: tournament_pair_options[key],
        horizontal=True,
        key="tournament_pair_page_selector",
    )
    st.session_state.tournament_pair = pair_key
    pair_key = st.session_state.get("tournament_pair", "v6")
    names = list(tournament_pairs.get(pair_key, tournament_pairs["v6"]))
    st.session_state.tournament_num_games = st.slider(
        "Games to play", 1, 100, st.session_state.get("tournament_num_games", 8), 1, key="tourn_num_games_slider"
    )
    st.session_state.tournament_mcts_simulations = st.slider(
        "MCTS simulations", 1, 512, st.session_state.tournament_mcts_simulations, 1, key="tourn_mcts_slider"
    )
    num_games = st.session_state.tournament_num_games
    matchups = [
        (names[0], names[1]) if idx % 2 == 0 else (names[1], names[0])
        for idx in range(num_games)
    ]

    st.markdown(
        '<div class="glass-card">'
        '<h2 style="margin:0">🏆 Tournament</h2>'
        '<p style="margin:0.3rem 0 0;opacity:0.7">'
        f'Direct evaluation match between {get_personality(names[0]).name} and {get_personality(names[1]).name} over {num_games} game(s).</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Initialise tournament results
    current_names = sorted(st.session_state.tournament_results.keys()) if st.session_state.tournament_results else []
    if current_names != sorted(names):
        st.session_state.tournament_results = {}
        st.session_state.tournament_games = []
        st.session_state.tournament_current_game = ""
    if not st.session_state.tournament_results:
        st.session_state.tournament_results = {
            n: {"wins": 0, "losses": 0, "draws": 0, "points": 0.0}
            for n in names
        }
        st.session_state.tournament_games = []

    # Standings table
    standings_col, board_col = st.columns([1.5, 2])

    with standings_col:
        st.markdown("### 📊 Standings")
        results = st.session_state.tournament_results
        sorted_names = sorted(names, key=lambda n: results[n]["points"], reverse=True)

        table_html = '<table class="tournament-table"><thead><tr>'
        table_html += '<th>#</th><th>Personality</th><th>W</th><th>D</th><th>L</th><th>Pts</th>'
        table_html += '</tr></thead><tbody>'
        for i, n in enumerate(sorted_names):
            p = ALL_PERSONALITIES[n]
            r = results[n]
            table_html += (
                f'<tr><td>{i+1}</td>'
                f'<td>{p.icon} {p.name}</td>'
                f'<td>{r["wins"]}</td><td>{r["draws"]}</td>'
                f'<td>{r["losses"]}</td><td><strong>{r["points"]:.1f}</strong></td></tr>'
            )
        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)

        # Game log
        if st.session_state.tournament_games:
            st.markdown("### 📋 Completed Games")
            for g in st.session_state.tournament_games[-10:]:
                st.markdown(f"- {g}")

    with board_col:
        if st.session_state.tournament_current_game:
            st.markdown(f"**Now playing:** {st.session_state.tournament_current_game}")
        engine = st.session_state.engine
        render_svg_board(engine.board, engine, size=420)

    # Tournament controls
    st.markdown("---")
    t_cols = st.columns(2)
    with t_cols[0]:
        if st.button("🚀 Start Tournament", use_container_width=True, key="tourn_start"):
            run_tournament(matchups)
    with t_cols[1]:
        if st.button("🔄 Reset Tournament", use_container_width=True, key="tourn_reset"):
            st.session_state.tournament_results = {}
            st.session_state.tournament_games = []
            st.session_state.tournament_current_game = ""
            new_game()
            st.rerun()


def run_tournament(matchups):
    """Play all round-robin matchups."""
    results = st.session_state.tournament_results
    progress = st.progress(0.0)
    status_text = st.empty()

    total = len(matchups)
    for idx, (w_name, b_name) in enumerate(matchups):
        w_p = ALL_PERSONALITIES[w_name]
        b_p = ALL_PERSONALITIES[b_name]

        st.session_state.tournament_current_game = (
            f"{w_p.icon} {w_p.name} (White) vs {b_p.icon} {b_p.name} (Black)"
        )
        status_text.markdown(f"**Game {idx+1}/{total}:** {st.session_state.tournament_current_game}")

        # Play one game
        engine = GameEngine()
        st.session_state.engine = engine
        max_moves = 150

        move_count = 0
        while not engine.board.is_game_over() and move_count < max_moves:
            p_key = w_name if engine.board.turn == chess.WHITE else b_name
            engine.bot_move(p_key, simulations=st.session_state.tournament_mcts_simulations)
            move_count += 1

        # Record result
        status = engine.get_game_status()
        if status["result"] == "1-0":
            results[w_name]["wins"]   += 1
            results[w_name]["points"] += 1.0
            results[b_name]["losses"] += 1
            result_str = f"{w_p.icon} {w_p.name} **1-0** {b_p.icon} {b_p.name}"
        elif status["result"] == "0-1":
            results[b_name]["wins"]   += 1
            results[b_name]["points"] += 1.0
            results[w_name]["losses"] += 1
            result_str = f"{w_p.icon} {w_p.name} **0-1** {b_p.icon} {b_p.name}"
        else:
            results[w_name]["draws"]  += 1
            results[b_name]["draws"]  += 1
            results[w_name]["points"] += 0.5
            results[b_name]["points"] += 0.5
            result_str = f"{w_p.icon} {w_p.name} **½-½** {b_p.icon} {b_p.name}"

        st.session_state.tournament_games.append(result_str)
        progress.progress((idx + 1) / total)

    status_text.markdown("### ✅ Tournament Complete!")
    st.session_state.tournament_current_game = ""
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  Reports cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_style_report(num_games: int, simulations: int, max_moves: int, model_pair: str):
    from src.analytics.style_report import generate_style_report

    return generate_style_report(num_games=num_games, simulations=simulations, max_moves=max_moves, model_pair=model_pair)


def render_metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta=delta)


def render_reports():
    report_pair_options = {
        "v6": "V6 Models",
        "old": "Old Models",
    }
    model_pair = st.radio(
        "Style behavior pair",
        list(report_pair_options),
        index=list(report_pair_options).index(st.session_state.get("style_report_pair_selector", "v6")),
        format_func=lambda key: report_pair_options[key],
        horizontal=True,
        key="style_report_pair_selector",
    )
    pair_descriptions = {
        "old": "Style profiling for the pre-2000 and post-2020 model pair.",
        "v6": "Style profiling for the pre-2000 v6 and post-2010 v6 model pair.",
    }

    st.markdown(
        '<div class="glass-card">'
        '<h2 style="margin:0">📚 Reports</h2>'
        '<p style="margin:0.3rem 0 0;opacity:0.7">'
        f'{pair_descriptions[model_pair]}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Style Behavior"])

    with tabs[0]:
        st.markdown("### Style Behavior Report")
        style_controls = st.columns(3)
        style_games = style_controls[0].slider("Head-to-head games", 1, 100, 30, 1)
        style_sims = style_controls[1].slider("MCTS simulations", 1, 512, 12)
        style_max_moves = style_controls[2].slider("Max plies", 40, 120, 80, 10)
        if st.button("Run style behavior analysis", key="run_style_report"):
            st.session_state["show_style_report"] = True

        if st.session_state.get("show_style_report"):
            try:
                with st.spinner("Simulating games and computing style metrics..."):
                    style = cached_style_report(style_games, style_sims, style_max_moves, model_pair)
                game_df = style["games"]
                move_df = style["moves"]
                summary_df = style["summary"]
                model_names = summary_df["model"].tolist() if not summary_df.empty else []
                first_model = model_names[0] if len(model_names) > 0 else "Model A"
                second_model = model_names[1] if len(model_names) > 1 else "Model B"
                first_wins = int((game_df["winner"] == first_model).sum()) if not game_df.empty else 0
                second_wins = int((game_df["winner"] == second_model).sum()) if not game_df.empty else 0
                render_metric_row(
                    [
                        ("Games", f"{len(game_df)}", None),
                        (f"{first_model} wins", f"{first_wins}", None),
                        (f"{second_model} wins", f"{second_wins}", None),
                    ]
                )
                st.dataframe(summary_df, use_container_width=True)
                if not summary_df.empty:
                    style_metric_columns = [
                        "sacrifice_rate",
                        "mobility_after",
                        "king_pressure_after",
                        "center_control_after",
                        "checks_given",
                        "capture_rate",
                        "queen_trade_rate",
                        "castling_rate",
                        "avg_move_tactical_pressure",
                        "avg_game_length",
                    ]
                    style_chart_df = summary_df.melt(
                        id_vars="model",
                        value_vars=[col for col in style_metric_columns if col in summary_df.columns],
                        var_name="metric",
                        value_name="value",
                    )
                    style_chart = (
                        alt.Chart(style_chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("metric:N", title=None),
                            y=alt.Y("value:Q", title="Average value"),
                            color=alt.Color("model:N", scale=alt.Scale(domain=model_names, range=["#3498db", "#e74c3c"])),
                            xOffset="model:N",
                            tooltip=["model:N", "metric:N", alt.Tooltip("value:Q", format=".3f")],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(style_chart, use_container_width=True)
                st.info(style["narrative"])
                st.download_button(
                    "Download move-level style CSV",
                    data=move_df.to_csv(index=False).encode("utf-8"),
                    file_name="style_moves.csv",
                    mime="text/csv",
                    key="download_style_moves",
                )
            except ModuleNotFoundError as exc:
                st.error(f"Missing report dependency: {exc}. Install requirements.txt and rerun.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ██  TESTS PAGE  ██
# ═══════════════════════════════════════════════════════════════════════════════

def page_tests():
    st.markdown(
        '<div class="glass-card">'
        '<h2 style="margin:0">🧪 Tests</h2>'
        '<p style="margin:0.3rem 0 0;opacity:0.7">'
        'Latest generated puzzle-evaluation results.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    generation_order = ["old", "v1", "ultral_final", "v6"]
    generation_display_names = {
        "old": "main",
        "v1": "v1",
        "ultral_final": "v3",
        "v6": "v6",
    }
    generation_dirs = [TESTS_RESULTS_DIR / name for name in generation_order]
    result_files = [path for directory in generation_dirs for path in sorted(directory.glob("*_results.txt"))]

    def parse_result_file(result_file: Path) -> list[dict[str, object]]:
        raw_text = result_file.read_text()
        generation_match = re.search(r"Model generation:\s+(.+)", raw_text)
        suite_match = re.search(r"^(M8N\d+)\s+Puzzle Results", raw_text, re.MULTILINE)
        if not generation_match or not suite_match:
            return []

        generation = generation_match.group(1).strip()
        suite = suite_match.group(1).lower()
        rows = []
        current_model = None
        for line in raw_text.splitlines():
            model_match = re.match(r"Model:\s+(.+)", line)
            if model_match:
                current_model = model_match.group(1).strip()
                continue

            solved_match = re.match(r"Solved:\s+(\d+)\s+/\s+(\d+)\s+\(([\d.]+)%\)", line)
            if solved_match and current_model:
                solved, total, accuracy = solved_match.groups()
                family = "pre_2000" if current_model.startswith("pre_2000") else "post_2010"
                rows.append(
                    {
                        "generation": generation,
                        "generation_display": generation_display_names.get(generation, generation),
                        "suite": suite,
                        "family": family,
                        "model": current_model,
                        "accuracy": float(accuracy),
                        "solved": int(solved),
                        "total": int(total),
                        "label": f"{accuracy}% ({solved}/{total})",
                    }
                )
        return rows

    if result_files:
        all_rows = []
        for result_file in result_files:
            all_rows.extend(parse_result_file(result_file))

        if all_rows:
            suite_order = ["m8n1", "m8n2", "m8n3", "m8n4"]
            family_titles = {
                "pre_2000": "Pre-2000",
                "post_2010": "Post-2010",
            }
            color_scale = alt.Scale(
                domain=[generation_display_names[name] for name in generation_order],
                range=["#5dade2", "#58d68d", "#f5b041", "#ec7063"],
            )
            tabs = st.tabs(["Overview", "Model Generation Accuracy Comparison"])

            with tabs[0]:
                st.markdown("### 📊 V6 Model Accuracy Overview")
                v6_overview_rows = [
                    row for row in all_rows
                    if row["generation"] == "v6" and row["model"] in {"pre_2000_final_v6", "post_2010_final_v6"}
                ]
                if v6_overview_rows:
                    v6_chart = (
                        alt.Chart(alt.Data(values=v6_overview_rows))
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("suite:N", title="Puzzle suite", sort=suite_order),
                            y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 30])),
                            xOffset=alt.XOffset("model:N", sort=["pre_2000_final_v6", "post_2010_final_v6"]),
                            color=alt.Color(
                                "model:N",
                                scale=alt.Scale(
                                    domain=["pre_2000_final_v6", "post_2010_final_v6"],
                                    range=["#3498db", "#e74c3c"],
                                ),
                                title="Model",
                            ),
                            tooltip=[
                                alt.Tooltip("suite:N", title="Suite"),
                                alt.Tooltip("model:N", title="Model"),
                                alt.Tooltip("solved:Q", title="Solved"),
                                alt.Tooltip("total:Q", title="Total"),
                                alt.Tooltip("accuracy:Q", title="Accuracy", format=".1f"),
                            ],
                        )
                        .properties(height=360)
                    )
                    v6_labels = (
                        alt.Chart(alt.Data(values=v6_overview_rows))
                        .mark_text(dy=-8, color="#e0e0e0")
                        .encode(
                            x=alt.X("suite:N", sort=suite_order),
                            y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 30])),
                            xOffset=alt.XOffset("model:N", sort=["pre_2000_final_v6", "post_2010_final_v6"]),
                            text="label:N",
                        )
                    )
                    st.altair_chart(v6_chart + v6_labels, use_container_width=True)
                else:
                    st.info("No v6 result files found yet.")

                st.markdown("### 📊 Old Model Accuracy Overview")
                overview_rows = [
                    row for row in all_rows
                    if row["generation"] == "old" and row["model"] in {"pre_2000", "post_2020"}
                ]
                if overview_rows:
                    overview_chart = (
                        alt.Chart(alt.Data(values=overview_rows))
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("suite:N", title="Puzzle suite", sort=suite_order),
                            y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 30])),
                            xOffset=alt.XOffset("model:N", sort=["pre_2000", "post_2020"]),
                            color=alt.Color(
                                "model:N",
                                scale=alt.Scale(
                                    domain=["pre_2000", "post_2020"],
                                    range=["#5dade2", "#ec7063"],
                                ),
                                title="Model",
                            ),
                            tooltip=[
                                alt.Tooltip("suite:N", title="Suite"),
                                alt.Tooltip("model:N", title="Model"),
                                alt.Tooltip("solved:Q", title="Solved"),
                                alt.Tooltip("total:Q", title="Total"),
                                alt.Tooltip("accuracy:Q", title="Accuracy", format=".1f"),
                            ],
                        )
                        .properties(height=360)
                    )
                    overview_labels = (
                        alt.Chart(alt.Data(values=overview_rows))
                        .mark_text(dy=-8, color="#e0e0e0")
                        .encode(
                            x=alt.X("suite:N", sort=suite_order),
                            y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 30])),
                            xOffset=alt.XOffset("model:N", sort=["pre_2000", "post_2020"]),
                            text="label:N",
                        )
                    )
                    st.altair_chart(overview_chart + overview_labels, use_container_width=True)
                else:
                    st.info("No old-model result files found yet.")

            with tabs[1]:
                st.markdown("### 📊 Model Generation Accuracy Comparison")
                for family in ["pre_2000", "post_2010"]:
                    st.markdown(f"#### {family_titles[family]} Variants")
                    cols = st.columns(2)
                    for idx, suite in enumerate(suite_order):
                        chart_rows = [
                            row for row in all_rows
                            if row["family"] == family and row["suite"] == suite
                        ]
                        with cols[idx % 2]:
                            st.markdown(f"**{suite}_{family}**")
                            if not chart_rows:
                                st.info("No results found for this suite yet.")
                                continue

                            base = alt.Chart(alt.Data(values=chart_rows))
                            bars = (
                                base
                                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                                .encode(
                                    x=alt.X(
                                        "generation_display:N",
                                        title="Generation",
                                        sort=[generation_display_names[name] for name in generation_order],
                                    ),
                                    y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 30])),
                                    color=alt.Color("generation_display:N", scale=color_scale, legend=None),
                                    tooltip=[
                                        alt.Tooltip("generation_display:N", title="Generation"),
                                        alt.Tooltip("model:N", title="Model"),
                                        alt.Tooltip("solved:Q", title="Solved"),
                                        alt.Tooltip("total:Q", title="Total"),
                                        alt.Tooltip("accuracy:Q", title="Accuracy", format=".1f"),
                                    ],
                                )
                            )
                            labels = (
                                base
                                .mark_text(dy=-8, color="#e0e0e0")
                                .encode(
                                    x=alt.X(
                                        "generation_display:N",
                                        sort=[generation_display_names[name] for name in generation_order],
                                    ),
                                    y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 30])),
                                    text="label:N",
                                )
                            )
                            chart = (bars + labels).properties(height=260)
                            st.altair_chart(chart, use_container_width=True)

                st.markdown("### 📄 Raw Result Files")
                for result_file in result_files:
                    display_dir = generation_display_names.get(result_file.parent.name, result_file.parent.name)
                    with st.expander(f"Show details for {display_dir}/{result_file.name}", expanded=False):
                        st.code(result_file.read_text(), language="text")
    else:
        st.markdown("*No generated results yet. Run `tests/eval_m8n2.py` first.*")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main routing
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.mode == "Human vs Bot":
    page_human_vs_bot()
elif st.session_state.mode == "Bot vs Bot":
    page_bot_vs_bot()
elif st.session_state.mode == "Tournament":
    page_tournament()
elif st.session_state.mode == "Reports":
    render_reports()
elif st.session_state.mode == "Tests":
    page_tests()
