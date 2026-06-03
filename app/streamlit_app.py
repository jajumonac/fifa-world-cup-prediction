import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model.core import ensure_session_state
from model.data import GROUPS
from model.ui import inject_css, render_footer, render_disclaimer, render_model_explainer, draw_predicted_bracket

st.set_page_config(
    page_title="2026 FIFA World Cup Predictions",
    page_icon="⚽",
    layout="wide",
)

inject_css()
ensure_session_state()

metrics        = st.session_state.metrics
team_elos      = st.session_state.team_elos
baseline       = st.session_state.baseline
positions      = baseline["positions"]
group_finish   = baseline["group_finish"]
rounds_reached = baseline["rounds_reached"]
N              = baseline["n_sims"]
all_teams      = [t for g in GROUPS.values() for t in g]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚽ 2026 FIFA World Cup Predictions")
st.markdown(
    "Elo rating system trained on **150+ years** of international results · "
    "**10,000 Monte Carlo simulations** of the official tournament draw · "
    "v4 model with recent form, WC psychology & squad trajectory enrichment"
)
st.markdown("---")

# ── Model metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC",          f"{metrics['roc_auc']:.3f}",   help="Match outcome accuracy (0.5 = random, 1.0 = perfect)")
c2.metric("Brier Score",      f"{metrics['brier']:.3f}",     help="Probability calibration (0.25 = random, lower is better)")
c3.metric("Training matches", f"{metrics['train_size']:,}")
c4.metric("Simulations",      f"{N:,}")
st.markdown("---")

render_disclaimer()
st.markdown("---")

# ── Championship probabilities table ──────────────────────────────────────────
st.subheader("🏆 Championship Probabilities — All 48 Teams")

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
       ["Champion", "Runner-up", "3rd Place", "4th Place",
        "Reach Final", "Reach SF", "Reach QF"]}
st.dataframe(
    results_df.style.format(fmt).background_gradient(
        subset=["Champion"], cmap="Blues", vmin=0, vmax=0.35
    ),
    use_container_width=True,
    hide_index=True,
)
st.markdown("---")

# ── Predicted tournament bracket ──────────────────────────────────────────────
st.subheader("🗺️ Predicted Tournament Path")
st.caption("Most likely semi-finalists, finalists, and final standings based on simulation probabilities.")

bracket_fig = draw_predicted_bracket(positions, rounds_reached, N)
st.pyplot(bracket_fig)
plt.close(bracket_fig)
st.markdown("---")

# ── Group stage finish probabilities ──────────────────────────────────────────
st.subheader("📊 Group Stage Finish Probabilities")
st.caption("Gold = 1st · Silver = 2nd · Bronze = 3rd · Red = 4th.  Teams sorted by qualification probability.")

pos_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#e05c6a"]
pos_labels = ["1st (Q)", "2nd (Q)", "3rd", "4th"]

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
fig.patch.set_color("#FAFBFF")
axes = axes.flatten()
for i, (gname, teams) in enumerate(GROUPS.items()):
    ax = axes[i]
    ax.set_facecolor("#FAFBFF")
    sorted_t = sorted(teams, key=lambda t: (group_finish[t][1] + group_finish[t][2]) / N)
    y_pos = np.arange(len(sorted_t))
    left  = np.zeros(len(sorted_t))
    for j, (color, label) in enumerate(zip(pos_colors, pos_labels)):
        widths = np.array([group_finish[t][j + 1] / N for t in sorted_t])
        ax.barh(y_pos, widths, left=left, color=color, label=label,
                edgecolor="white", height=0.62)
        for k, w in enumerate(widths):
            if w > 0.08:
                ax.text(left[k] + w / 2, y_pos[k], f"{w:.0%}",
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color="#1a1a2e")
        left += widths
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:16] for t in sorted_t], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#B0C4DE", linestyle="--", alpha=0.5, linewidth=0.9)
    ax.set_title(f"Group {gname}", fontweight="bold", fontsize=10.5,
                 color="#0A1628", pad=6)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    if i == 0:
        ax.legend(loc="lower right", fontsize=7, ncol=2,
                  framealpha=0.9, edgecolor="#ccc")

plt.suptitle("Group Stage Finish Probabilities — 10,000 simulations",
             fontsize=13, y=1.01, fontweight="bold", color="#0A1628")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.markdown("---")

# ── Road to the Final ─────────────────────────────────────────────────────────
st.subheader("📈 Road to the Final — Top 12 Contenders")
st.caption("Each line shows a team's probability of reaching each knockout stage.")

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
ax.set_facecolor("#FAFBFF")
fig.patch.set_color("#FAFBFF")
for i, (_, row) in enumerate(top12.iterrows()):
    ax.plot(x, [row[s] for s in stages], marker="o", linewidth=2.2,
            markersize=6, label=row["Team"], color=colors[i],
            alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=10.5, fontweight="500")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.set_ylim(0, 1.05)
ax.set_title("Probability of Reaching Each Stage — Top 12 Contenders",
             fontsize=12, fontweight="bold", color="#0A1628", pad=10)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9,
          framealpha=0.95, edgecolor="#D0DEFF")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.markdown("---")

# ── Model explainer ───────────────────────────────────────────────────────────
render_model_explainer()

# ── Footer ────────────────────────────────────────────────────────────────────
render_footer()
