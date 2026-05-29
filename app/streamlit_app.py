import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model.core import ensure_session_state
from model.data import GROUPS

st.set_page_config(
    page_title="2026 FIFA World Cup Predictions",
    page_icon="⚽",
    layout="wide",
)

ensure_session_state()

metrics   = st.session_state.metrics
team_elos = st.session_state.team_elos
baseline  = st.session_state.baseline
positions      = baseline["positions"]
group_finish   = baseline["group_finish"]
rounds_reached = baseline["rounds_reached"]
N = baseline["n_sims"]
all_teams = [t for g in GROUPS.values() for t in g]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚽ 2026 FIFA World Cup Prediction Model")
st.markdown(
    "Powered by a custom Elo rating system trained on **150+ years** of international "
    "results and **10,000 Monte Carlo simulations** of the official tournament draw."
)
st.markdown("---")

# ── Model metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="0.5 = random, 1.0 = perfect")
c2.metric("Brier Score", f"{metrics['brier']:.3f}", help="0.25 = random. Lower is better.")
c3.metric("Training matches", f"{metrics['train_size']:,}")
c4.metric("Simulations", f"{N:,}")
st.markdown("---")

# ── Championship probabilities table ──────────────────────────────────────────
st.subheader("Championship Probabilities — All 48 Teams")

results_df = pd.DataFrame([{
    "Team":         t,
    "Group":        next(g for g, ts in GROUPS.items() if t in ts),
    "Adj. Elo":     int(team_elos[t]),
    "Champion":     positions[t][1] / N,
    "Runner-up":    positions[t][2] / N,
    "3rd Place":    positions[t][3] / N,
    "4th Place":    positions[t][4] / N,
    "Reach Final":  rounds_reached[t]["Final"] / N,
    "Reach SF":     rounds_reached[t]["SF"] / N,
    "Reach QF":     rounds_reached[t]["QF"] / N,
} for t in all_teams]).sort_values("Champion", ascending=False).reset_index(drop=True)

fmt = {c: "{:.1%}" for c in
       ["Champion", "Runner-up", "3rd Place", "4th Place", "Reach Final", "Reach SF", "Reach QF"]}
st.dataframe(
    results_df.style.format(fmt),
    use_container_width=True,
    hide_index=True,
)
st.markdown("---")

# ── Group stage finish probabilities ──────────────────────────────────────────
st.subheader("Group Stage Finish Probabilities")
pos_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#d9534f"]
pos_labels = ["1st (Q)", "2nd (Q)", "3rd", "4th"]

fig, axes = plt.subplots(3, 4, figsize=(20, 13))
axes = axes.flatten()
for i, (gname, teams) in enumerate(GROUPS.items()):
    ax = axes[i]
    sorted_t = sorted(teams, key=lambda t: (group_finish[t][1] + group_finish[t][2]) / N)
    y_pos = np.arange(len(sorted_t))
    left = np.zeros(len(sorted_t))
    for j, (color, label) in enumerate(zip(pos_colors, pos_labels)):
        widths = np.array([group_finish[t][j + 1] / N for t in sorted_t])
        ax.barh(y_pos, widths, left=left, color=color, label=label,
                edgecolor="white", height=0.65)
        for k, w in enumerate(widths):
            if w > 0.07:
                ax.text(left[k] + w / 2, y_pos[k], f"{w:.0%}",
                        ha="center", va="center", fontsize=7, fontweight="bold")
        left += widths
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:15] for t in sorted_t], fontsize=8)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="navy", linestyle="--", alpha=0.2, linewidth=1)
    ax.set_title(f"Group {gname}", fontweight="bold", fontsize=10)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", labelsize=7)
    if i == 0:
        ax.legend(loc="lower right", fontsize=7, ncol=2)
plt.suptitle("Group Stage Finish Probabilities (10,000 simulations)", fontsize=13, y=1.01)
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.markdown("---")

# ── Tournament progression heatmap ────────────────────────────────────────────
st.subheader("Full Tournament Progression — All 48 Teams")
stages = ["R32", "R16", "QF", "SF", "Final", "Champion"]
prog_df = pd.DataFrame({
    "Team":     all_teams,
    "Group":    [next(g for g, ts in GROUPS.items() if t in ts) for t in all_teams],
    "R32":      [rounds_reached[t]["R32"]   / N for t in all_teams],
    "R16":      [rounds_reached[t]["R16"]   / N for t in all_teams],
    "QF":       [rounds_reached[t]["QF"]    / N for t in all_teams],
    "SF":       [rounds_reached[t]["SF"]    / N for t in all_teams],
    "Final":    [rounds_reached[t]["Final"] / N for t in all_teams],
    "Champion": [positions[t][1]            / N for t in all_teams],
}).sort_values("Champion", ascending=False).reset_index(drop=True)

heat_data = prog_df[stages].values
fig, ax = plt.subplots(figsize=(14, 18))
im = ax.imshow(heat_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(prog_df)))
ax.set_yticklabels([f"{r.Team} ({r.Group})" for _, r in prog_df.iterrows()], fontsize=8)
for i in range(len(prog_df)):
    for j in range(len(stages)):
        v = heat_data[i, j]
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.6 else "black")
ax.set_title("Tournament Progression Probabilities", fontsize=13, pad=15)
plt.colorbar(im, ax=ax, label="Probability", shrink=0.35)
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.markdown("---")

# ── Road to the Final ─────────────────────────────────────────────────────────
st.subheader("Road to the Final — Top 12 Contenders")
top12 = prog_df.head(12)
x = np.arange(len(stages))
colors_12 = plt.cm.tab20(np.linspace(0, 1, 12))
fig, ax = plt.subplots(figsize=(12, 6))
for i, (_, row) in enumerate(top12.iterrows()):
    ax.plot(x, [row[s] for s in stages], marker="o", linewidth=2,
            markersize=5, label=row["Team"], color=colors_12[i])
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.set_ylim(0, 1.05)
ax.set_title("Probability of reaching each stage", fontsize=12)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()
