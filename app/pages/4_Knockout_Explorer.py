import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model.core import ensure_session_state
from model.data import get_all_teams, GROUPS
from model.ui import inject_css, render_footer

st.set_page_config(page_title="Knockout Explorer", page_icon="", layout="wide")
inject_css()
ensure_session_state()

baseline  = st.session_state.baseline
N         = baseline["n_sims"]
mc        = baseline["matchup_counts"]
all_teams = sorted(get_all_teams())

ROUNDS = ["R32", "R16", "QF", "SF", "Final"]
ROUND_LABELS = {
    "R32":   "Round of 32",
    "R16":   "Round of 16",
    "QF":    "Quarter-final",
    "SF":    "Semi-final",
    "Final": "Final",
}

st.markdown('<h1>Knockout Path Explorer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#556B8A; font-size:0.92rem; margin-top:0;">'
    'Select any team to see who they are most likely to face at each knockout stage '
    'based on the official 2026 bracket and 10,000 simulated tournaments.'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

team     = st.selectbox("Select a team", all_teams, index=all_teams.index("Spain"))
group_of = next((g for g, ts in GROUPS.items() if team in ts), "—")
st.caption(f"Group {group_of} — probabilities out of {N:,} simulations")
st.markdown("---")


def opponents_for(team, round_name):
    rows = []
    for (a, b), count in mc[round_name].items():
        if a == team or b == team:
            opp = b if a == team else a
            og  = next((g for g, ts in GROUPS.items() if opp in ts), "—")
            rows.append({"Opponent": opp, "Group": og, "Probability": count / N})
    return (pd.DataFrame(rows)
              .sort_values("Probability", ascending=False)
              .reset_index(drop=True))


st.subheader(f"Likely opponents for {team} — round by round")
tabs = st.tabs([ROUND_LABELS[r] for r in ROUNDS])

for tab, round_name in zip(tabs, ROUNDS):
    with tab:
        df = opponents_for(team, round_name)
        if df.empty:
            st.info(f"{team} never reached the {ROUND_LABELS[round_name]} in any simulation.")
            continue

        reach_prob = df["Probability"].sum()
        st.metric(f"Probability of reaching the {ROUND_LABELS[round_name]}", f"{reach_prob:.1%}")

        col_table, col_chart = st.columns([1, 1])
        with col_table:
            st.dataframe(
                df.style
                    .format({"Probability": "{:.1%}"})
                    .background_gradient(subset=["Probability"], cmap="Blues", vmin=0, vmax=0.35),
                use_container_width=True,
                hide_index=True,
            )
        with col_chart:
            top = df.head(10)
            fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.44)))
            ax.set_facecolor("#0A1F3B")
            fig.patch.set_facecolor("#0A1F3B")
            bar_colors = ["#F0B429" if p > 0.12 else "#2E6EA8" if p > 0.05 else "#1E3A5A"
                          for p in top["Probability"]]
            ax.barh(top["Opponent"][::-1], top["Probability"][::-1],
                    color=bar_colors[::-1], edgecolor="#05172E")
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax.tick_params(colors="#556B8A")
            ax.set_yticklabels(top["Opponent"][::-1], fontsize=8.5, color="#C8D8EE")
            ax.set_title(f"Most likely {ROUND_LABELS[round_name]} opponents",
                         fontsize=9.5, fontweight="bold", color="#FFFFFF")
            ax.spines[["top", "right"]].set_color("rgba(255,255,255,0.06)")
            ax.spines[["left", "bottom"]].set_color("rgba(255,255,255,0.06)")
            for i, v in enumerate(top["Probability"].values[::-1]):
                ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=7.5, color="#8BA3C1")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.markdown("---")

# ── Full heatmap ──────────────────────────────────────────────────────────────
st.subheader(f"Full matchup heatmap — {team}")

heat_rows = []
for opp in all_teams:
    if opp == team:
        continue
    row = {"Team": opp, "Group": next((g for g, ts in GROUPS.items() if opp in ts), "—")}
    for r in ROUNDS:
        key    = (team, opp) if team < opp else (opp, team)
        row[r] = mc[r].get(key, 0) / N
    heat_rows.append(row)

heat_df = (
    pd.DataFrame(heat_rows)
    .assign(_max=lambda d: d[ROUNDS].max(axis=1))
    .sort_values("_max", ascending=False)
    .drop(columns="_max")
    .reset_index(drop=True)
)

st.dataframe(
    heat_df.style
        .format({r: "{:.1%}" for r in ROUNDS})
        .background_gradient(subset=ROUNDS, cmap="YlOrRd", vmin=0, vmax=0.20),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ── Head-to-head ──────────────────────────────────────────────────────────────
st.subheader("Head-to-head deep-dive")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Team A:** {team}")
with col2:
    opponent = st.selectbox("Team B", [t for t in all_teams if t != team], index=0)

opp_group = next((g for g, ts in GROUPS.items() if opponent in ts), "—")

rows_h2h = []
for r in ROUNDS:
    key   = (team, opponent) if team < opponent else (opponent, team)
    count = mc[r].get(key, 0)
    rows_h2h.append({"Round": ROUND_LABELS[r], "Probability": count / N})

h2h_df        = pd.DataFrame(rows_h2h)
total_any_rnd = h2h_df["Probability"].sum()

st.markdown(
    f"**{team}** (Group {group_of}) vs **{opponent}** (Group {opp_group})  \n"
    f"Chance they meet in any round: **{total_any_rnd:.1%}**"
)

cols5 = st.columns(5)
for col, (_, row) in zip(cols5, h2h_df.iterrows()):
    col.metric(row["Round"], f"{row['Probability']:.1%}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.set_facecolor("#0A1F3B")
fig.patch.set_facecolor("#0A1F3B")
bar_colors = ["#F0B429" if p > 0.07 else "#2E6EA8" if p > 0.03 else "#1E3A5A"
              for p in h2h_df["Probability"]]
ax.bar(h2h_df["Round"], h2h_df["Probability"], color=bar_colors,
       edgecolor="#05172E", width=0.55)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.tick_params(colors="#556B8A")
ax.set_xticklabels(h2h_df["Round"], color="#C8D8EE")
ax.set_title(f"{team} vs {opponent} — matchup probability by round",
             fontsize=10.5, fontweight="bold", color="#FFFFFF", pad=10)
ax.spines[["top", "right"]].set_color("rgba(255,255,255,0.06)")
ax.spines[["left", "bottom"]].set_color("rgba(255,255,255,0.06)")
for i, v in enumerate(h2h_df["Probability"]):
    if v > 0:
        ax.text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=9, color="#8BA3C1")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

render_footer()
