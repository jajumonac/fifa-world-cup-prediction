"""Shared UI helpers — injected on every page."""
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np


LINKEDIN = "https://www.linkedin.com/in/jadiel-montero-230814142/"
GITHUB   = "https://github.com/jajumonac"


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    /* ── Page title ── */
    h1 { color: #0A1628 !important; font-weight: 800 !important; letter-spacing: -0.5px; }
    h2 { color: #1B3A5C !important; font-weight: 700 !important; }
    h3 { color: #1B3A5C !important; font-weight: 600 !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #F4F7FF 0%, #EAF0FF 100%);
        border-radius: 14px;
        padding: 18px 22px !important;
        border: 1px solid #D0DEFF;
        box-shadow: 0 2px 10px rgba(26,84,196,0.07);
    }
    [data-testid="stMetricLabel"] { font-size: 0.73rem !important; color: #667799 !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.04em; }
    [data-testid="stMetricValue"] { color: #0A1628 !important; font-weight: 800 !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0A1628 0%, #0F2040 100%) !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label { color: #C8D8F0 !important; }
    [data-testid="stSidebarNavLink"] {
        border-radius: 8px !important; margin: 2px 4px !important;
        color: #C8D8F0 !important;
    }
    [data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,0.08) !important; }
    [data-testid="stSidebarNavLink"][aria-selected="true"] {
        background: rgba(200,216,240,0.15) !important;
        color: #FFFFFF !important; font-weight: 600 !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #C41E3A, #8B1429) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        padding: 0.45rem 1.4rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(196,30,58,0.35) !important;
    }
    /* ── Regular button ── */
    .stButton > button {
        border-radius: 10px !important; font-weight: 500 !important;
        transition: transform 0.12s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #D8E4FF !important;
        border-radius: 12px !important;
        background: #FAFBFF !important;
    }
    summary { font-weight: 600 !important; color: #1B3A5C !important; }

    /* ── Alert / info boxes ── */
    [data-testid="stAlert"] { border-radius: 12px !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] iframe { border-radius: 10px !important; }

    /* ── Divider ── */
    hr { border-color: #E4EAFF !important; margin: 2rem 0 !important; }

    /* ── Footer ── */
    .wc-footer {
        margin-top: 2.5rem; padding: 1.2rem 0 0.5rem;
        border-top: 1px solid #E4EAFF;
        text-align: center; color: #99A8C0; font-size: 0.78rem; line-height: 1.8;
    }
    .wc-footer a { color: #1A54C8; text-decoration: none; font-weight: 500; }
    .wc-footer a:hover { text-decoration: underline; }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important; font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown(
        f'<div class="wc-footer">'
        f'Built by <strong>Jadiel Montero</strong> &nbsp;·&nbsp; '
        f'<a href="{LINKEDIN}" target="_blank">LinkedIn</a> &nbsp;·&nbsp; '
        f'<a href="{GITHUB}" target="_blank">GitHub</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.warning(
        "**Disclaimer:** This tool is for educational and entertainment purposes only. "
        "It is **not** financial or betting advice. Predictions are probabilistic estimates "
        "based on historical data and do not guarantee any specific outcome.",
        icon="⚠️",
    )


def render_model_explainer():
    with st.expander("🔬 How does the model work?", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**1. Elo Rating System**
Custom Elo engine trained on 150+ years of international results (1872–present).
Weighted by tournament importance, goal margin, and home advantage.

**2. Match Outcome Model**
Logistic regression on Elo difference predicts win probability for any matchup.
ROC-AUC **0.855** on held-out 2018–present matches.
Four model families tested — all within 0.001 AUC of each other.
""")
        with col2:
            st.markdown("""
**3. Feature Enrichment (simulation inputs)**
- Recent form: weighted PPG in last 10 matches (±75 Elo)
- WC psychology: WC win rate vs overall win rate, last 3 tournaments (±40 Elo)
- Squad trajectory: 2-year Elo velocity (±30 Elo)

**4. Monte Carlo Simulation**
Full tournament simulated 10,000 times using the official group draw.
Output: empirical probability distribution over all outcomes.
""")
        st.markdown(
            f"Full methodology: [HANDOUT.md](https://github.com/jajumonac/fifa-world-cup-prediction/blob/master/HANDOUT.md) "
            f"· [GitHub repo]({GITHUB}/fifa-world-cup-prediction)",
            unsafe_allow_html=False,
        )


def draw_predicted_bracket(positions, rounds_reached, N):
    """Predicted tournament bracket: SF → Final → Champion/RU/3rd/4th."""
    all_teams = list(positions.keys())

    champion  = max(all_teams, key=lambda t: positions[t][1])
    runner_up = max(all_teams, key=lambda t: positions[t][2])
    third     = max(all_teams, key=lambda t: positions[t][3])
    fourth    = max(all_teams, key=lambda t: positions[t][4])

    sf4  = sorted(all_teams, key=lambda t: -rounds_reached[t]["SF"])[:4]
    fin2 = sorted(all_teams, key=lambda t: -rounds_reached[t]["Final"])[:2]

    # ── Figure ────────────────────────────────────────────────────
    BG = "#0D1B2A"
    fig, ax = plt.subplots(figsize=(14, 6.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.3, 7.5)
    ax.axis("off")

    GOLD, BLUE, DIM, TEXT = "#FFD700", "#4A9BE8", "#5A6A80", "#FFFFFF"
    BOX_W, BOX_H = 2.45, 0.78

    def box_color(prob):
        t = min(prob / 0.35, 1.0)
        r = int(0x4A + (0xFF - 0x4A) * t)
        g = int(0x9B + (0xD7 - 0x9B) * t)
        b = int(0xE8 + (0x00 - 0xE8) * t)
        return f"#{r:02X}{g:02X}{b:02X}"

    def draw_box(x, y, name, prob, sublabel="", champion=False):
        color = GOLD if champion else box_color(prob)
        fill  = mcolors.to_rgba(color, 0.18 if champion else 0.12)
        rect  = mpatches.FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.1",
            facecolor=fill, edgecolor=color,
            linewidth=2.2 if champion else 1.0, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.12, name[:17], ha="center", va="center",
                fontsize=9 if champion else 8.5, color=TEXT,
                fontweight="bold", zorder=4)
        ax.text(x, y - 0.18, f"{prob:.0%}", ha="center", va="center",
                fontsize=8, color=color, zorder=4)
        if sublabel:
            ax.text(x, y + BOX_H / 2 + 0.17, sublabel,
                    ha="center", va="bottom", fontsize=6.5,
                    color="#7A9ABB", style="italic", zorder=4)

    def lconnect(x1, y1, x2, y2):
        mx = (x1 + x2) / 2
        ax.plot([x1, mx, mx, x2], [y1, y1, y2, y2],
                color="#1E3A5A", lw=1.4, zorder=1,
                solid_capstyle="round", solid_joinstyle="round")

    # ── Coordinates ───────────────────────────────────────────────
    XS, XF, XC, XP = 2.0, 5.8, 9.6, 13.2
    SF_Y  = [5.7, 4.1, 2.5, 0.9]
    FIN_Y = [(SF_Y[0] + SF_Y[1]) / 2, (SF_Y[2] + SF_Y[3]) / 2]  # 4.9, 1.7

    # Determine which finalist side the champion comes from
    champ_side = 0 if fin2[0] == champion else 1
    CHAMP_Y = FIN_Y[champ_side]
    RU_Y    = FIN_Y[1 - champ_side]
    CY      = (CHAMP_Y + RU_Y) / 2  # vertical center

    # ── Draw SF ───────────────────────────────────────────────────
    for i, t in enumerate(sf4):
        lbl = "Semi-Finalist" if i == 0 else ""
        draw_box(XS, SF_Y[i], t, rounds_reached[t]["SF"] / N, lbl)

    # ── Draw Finalists ────────────────────────────────────────────
    draw_box(XF, FIN_Y[0], fin2[0], rounds_reached[fin2[0]]["Final"] / N, "Finalist")
    draw_box(XF, FIN_Y[1], fin2[1], rounds_reached[fin2[1]]["Final"] / N, "Finalist")

    # ── Draw Champion & Runner-up ─────────────────────────────────
    draw_box(XC, CHAMP_Y, champion,  positions[champion][1]  / N, "🏆 Champion",  champion=True)
    draw_box(XC, RU_Y,    runner_up, positions[runner_up][2] / N, "Runner-up")

    # ── Draw 3rd & 4th ────────────────────────────────────────────
    draw_box(XP, CY + 1.0, third,  positions[third][3]  / N, "3rd Place")
    draw_box(XP, CY - 1.0, fourth, positions[fourth][4] / N, "4th Place")

    # ── Connections ───────────────────────────────────────────────
    for i in range(4):
        fin_idx = 0 if i < 2 else 1
        lconnect(XS + BOX_W / 2, SF_Y[i], XF - BOX_W / 2, FIN_Y[fin_idx])
    lconnect(XF + BOX_W / 2, FIN_Y[champ_side],     XC - BOX_W / 2, CHAMP_Y)
    lconnect(XF + BOX_W / 2, FIN_Y[1 - champ_side], XC - BOX_W / 2, RU_Y)

    # ── Stage headers ─────────────────────────────────────────────
    for x, lbl in [(XS, "SEMI-FINALS"), (XF, "FINALS"),
                   (XC, "FINAL RESULT"), (XP, "PODIUM")]:
        ax.text(x, 7.1, lbl, ha="center", va="center",
                fontsize=7, color="#445566", fontweight="bold",
                fontfamily="monospace")

    # Vertical dividers
    for xv in [XS + 1.7, XF + 1.6, XC + 1.8]:
        ax.axvline(xv, ymin=0.04, ymax=0.88, color="#152030", lw=0.8, zorder=0)

    ax.text(0.15, 7.1, "Predicted Tournament Path  ·  10,000 simulations",
            ha="left", va="center", fontsize=8.5, color="#6688AA", style="italic")

    plt.tight_layout(pad=0.4)
    return fig
