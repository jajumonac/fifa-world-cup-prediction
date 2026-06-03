import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model.core import ensure_session_state
from model.data import GROUPS
from model.ui import (
    inject_css, render_footer, render_disclaimer,
    render_github_link, render_interactive_bracket,
)

st.set_page_config(
    page_title="2026 FIFA World Cup Predictions",
    page_icon="",
    layout="wide",
)
inject_css()
ensure_session_state()

metrics        = st.session_state.metrics
team_elos      = st.session_state.team_elos
model          = st.session_state.model
baseline       = st.session_state.baseline
positions      = baseline["positions"]
group_finish   = baseline["group_finish"]
rounds_reached = baseline["rounds_reached"]
matchup_counts = baseline["matchup_counts"]
N              = baseline["n_sims"]
all_teams      = [t for g in GROUPS.values() for t in g]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:4px;">2026 FIFA World Cup Predictions</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#556B8A; font-size:0.92rem; margin-top:0;">'
    'Elo rating system &middot; 150+ years of match data &middot; '
    '10,000 Monte Carlo simulations &middot; v4 enrichment model'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Model metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC",          f"{metrics['roc_auc']:.3f}",  help="Match outcome accuracy (0.5 = random)")
c2.metric("Brier Score",      f"{metrics['brier']:.3f}",    help="Probability calibration (0.25 = random, lower = better)")
c3.metric("Training matches", f"{metrics['train_size']:,}")
c4.metric("Simulations",      f"{N:,}")

st.markdown("---")
render_disclaimer()
st.markdown("---")

# ── Championship table ────────────────────────────────────────────────────────
st.subheader("Championship Probabilities — All 48 Teams")

results_df = pd.DataFrame([{
    "Team":        t,
    "Group":       next(g for g, ts in GROUPS.items() if t in ts),
    "Adj. Elo":    int(team_elos[t]),
    "Champion":    positions[t][1] / N,
    "Runner-up":   positions[t][2] / N,
    "3rd Place":   positions[t][3] / N,
    "4th Place":   positions[t][4] / N,
    "Reach Final": rounds_reached[t]["Final"] / N,
    "Reach SF":    rounds_reached[t]["SF"] / N,
    "Reach QF":    rounds_reached[t]["QF"] / N,
} for t in all_teams]).sort_values("Champion", ascending=False).reset_index(drop=True)

fmt = {c: "{:.1%}" for c in
       ["Champion", "Runner-up", "3rd Place", "4th Place", "Reach Final", "Reach SF", "Reach QF"]}
st.dataframe(
    results_df.style
        .format(fmt)
        .background_gradient(subset=["Champion"], cmap="YlOrRd", vmin=0, vmax=0.35),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ── Interactive bracket ───────────────────────────────────────────────────────
st.subheader("Predicted Tournament Path")
st.markdown(
    '<p style="color:#556B8A; font-size:0.85rem; margin-top:-8px;">'
    'The model\'s predicted bracket based on simulation probabilities. '
    'Select the winner of each match to explore alternative paths.'
    '</p>',
    unsafe_allow_html=True,
)

render_interactive_bracket(matchup_counts, model, team_elos, N)

st.markdown("---")

# ── Group stage charts ────────────────────────────────────────────────────────
with st.expander("Group Stage Finish Probabilities", expanded=False):
    st.caption("Sorted by probability of qualifying. Gold = 1st, Silver = 2nd, Bronze = 3rd, Red = 4th.")

    pos_colors = ["#FFD700", "#9AA5B4", "#8B6914", "#8B2030"]
    pos_labels = ["1st", "2nd", "3rd", "4th"]

    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    fig.patch.set_facecolor("#0A1F3B")
    axes = axes.flatten()

    for i, (gname, teams) in enumerate(GROUPS.items()):
        ax = axes[i]
        ax.set_facecolor("#0A1F3B")
        sorted_t = sorted(teams, key=lambda t: (group_finish[t][1] + group_finish[t][2]) / N)
        y_pos = np.arange(len(sorted_t))
        left  = np.zeros(len(sorted_t))

        for j, (color, label) in enumerate(zip(pos_colors, pos_labels)):
            widths = np.array([group_finish[t][j + 1] / N for t in sorted_t])
            ax.barh(y_pos, widths, left=left, color=color,
                    label=label, edgecolor="#05172E", height=0.62)
            for k, w in enumerate(widths):
                if w > 0.09:
                    ax.text(left[k] + w / 2, y_pos[k], f"{w:.0%}",
                            ha="center", va="center", fontsize=7.5,
                            fontweight="bold", color="#05172E")
            left += widths

        ax.set_yticks(y_pos)
        ax.set_yticklabels([t[:16] for t in sorted_t], fontsize=8.5, color="#C8D8EE")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="rgba(255,255,255,0.15)", linestyle="--",
                   alpha=0.4, linewidth=0.8)
        ax.set_title(f"Group {gname}", fontweight="bold", fontsize=10.5,
                     color="#FFFFFF", pad=5)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.tick_params(axis="x", labelsize=7.5, colors="#556B8A")
        ax.tick_params(left=False)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7, ncol=2,
                      facecolor="#0A1F3B", edgecolor="rgba(255,255,255,0.1)",
                      labelcolor="#C8D8EE")

    plt.suptitle("Group Stage Finish Probabilities — 10,000 simulations",
                 fontsize=12, y=1.01, fontweight="bold", color="#FFFFFF")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ── Road to the Final line chart ──────────────────────────────────────────────
with st.expander("Road to the Final — Top 12 Contenders", expanded=False):
    stages  = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    prog_df = pd.DataFrame({
        "Team":     all_teams,
        "R32":      [rounds_reached[t]["R32"]   / N for t in all_teams],
        "R16":      [rounds_reached[t]["R16"]   / N for t in all_teams],
        "QF":       [rounds_reached[t]["QF"]    / N for t in all_teams],
        "SF":       [rounds_reached[t]["SF"]    / N for t in all_teams],
        "Final":    [rounds_reached[t]["Final"] / N for t in all_teams],
        "Champion": [positions[t][1]            / N for t in all_teams],
    }).sort_values("Champion", ascending=False).reset_index(drop=True)

    top12  = prog_df.head(12)
    x      = np.arange(len(stages))
    colors = plt.cm.tab20(np.linspace(0, 1, 12))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_facecolor("#0A1F3B")
    fig.patch.set_facecolor("#0A1F3B")
    for i, (_, row) in enumerate(top12.iterrows()):
        ax.plot(x, [row[s] for s in stages], marker="o", linewidth=2,
                markersize=5, label=row["Team"], color=colors[i], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10, color="#C8D8EE")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(colors="#556B8A")
    ax.set_ylim(0, 1.05)
    ax.set_title("Probability of Reaching Each Stage",
                 fontsize=11, fontweight="bold", color="#FFFFFF", pad=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5,
              facecolor="#0A1F3B", edgecolor="rgba(255,255,255,0.08)",
              labelcolor="#C8D8EE")
    ax.spines[["top", "right"]].set_color("rgba(255,255,255,0.06)")
    ax.spines[["left", "bottom"]].set_color("rgba(255,255,255,0.06)")
    ax.grid(True, alpha=0.1, color="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ── Model section ─────────────────────────────────────────────────────────────
with st.expander("About the model", expanded=False):
    st.markdown(
        "Full technical documentation, feature engineering details, and historical backtest results:"
    )
    render_github_link()

render_footer()
