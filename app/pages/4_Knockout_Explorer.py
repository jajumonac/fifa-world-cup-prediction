import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model.core import ensure_session_state
from model.data import get_all_teams, GROUPS

st.set_page_config(page_title="Knockout Explorer", page_icon="🗺️", layout="wide")

ensure_session_state()

baseline = st.session_state.baseline
N        = baseline["n_sims"]
mc       = baseline["matchup_counts"]   # {round: {(a,b): count}}
all_teams = sorted(get_all_teams())

ROUNDS = ["R32", "R16", "QF", "SF", "Final"]
ROUND_LABELS = {
    "R32":   "Round of 32",
    "R16":   "Round of 16",
    "QF":    "Quarter-final",
    "SF":    "Semi-final",
    "Final": "Final",
}

st.title("🗺️ Knockout Path Explorer")
st.markdown(
    "Based on the **official 2026 FIFA World Cup bracket**, "
    "pick any team to see who they are most likely to face at each knockout stage "
    "and the probability of each matchup occurring across 10,000 simulated tournaments."
)
st.markdown("---")

# ── Team selector ─────────────────────────────────────────────────────────────
team = st.selectbox(
    "Select a team",
    all_teams,
    index=all_teams.index("Spain"),
)

group_of = next((g for g, ts in GROUPS.items() if team in ts), "—")
st.markdown(f"**Group {group_of}** — probabilities below are out of {N:,} simulated tournaments.")
st.markdown("---")

# ── Build matchup table for selected team ────────────────────────────────────
def opponents_for(team: str, round_name: str) -> pd.DataFrame:
    rows = []
    for (a, b), count in mc[round_name].items():
        if a == team or b == team:
            opp = b if a == team else a
            opp_group = next((g for g, ts in GROUPS.items() if opp in ts), "—")
            rows.append({
                "Opponent":       opp,
                "Group":          opp_group,
                "Prob. matchup":  count / N,
            })
    return (
        pd.DataFrame(rows)
        .sort_values("Prob. matchup", ascending=False)
        .reset_index(drop=True)
    )

# ── Round-by-round cards ──────────────────────────────────────────────────────
st.subheader(f"Likely opponents for {team} — round by round")

tabs = st.tabs([ROUND_LABELS[r] for r in ROUNDS])

for tab, round_name in zip(tabs, ROUNDS):
    with tab:
        df = opponents_for(team, round_name)
        if df.empty:
            st.info(f"{team} never reached the {ROUND_LABELS[round_name]} in any simulation.")
            continue

        total_appearance = df["Prob. matchup"].sum()
        st.metric(
            f"{team} reaches the {ROUND_LABELS[round_name]}",
            f"{total_appearance:.1%}",
            help="Sum of all matchup probabilities = probability of reaching this round",
        )

        col_table, col_chart = st.columns([1, 1])

        with col_table:
            st.dataframe(
                df.style.format({"Prob. matchup": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )

        with col_chart:
            top = df.head(10)
            fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.45)))
            colors = ["#2ca02c" if p > 0.10 else "#aec7e8" for p in top["Prob. matchup"]]
            ax.barh(
                top["Opponent"][::-1],
                top["Prob. matchup"][::-1],
                color=colors[::-1],
                edgecolor="white",
            )
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax.set_title(f"Most likely {ROUND_LABELS[round_name]} opponents", fontsize=10)
            for i, v in enumerate(top["Prob. matchup"].values[::-1]):
                ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

st.markdown("---")

# ── Full bracket heat-map: who can this team face, at which stage ─────────────
st.subheader(f"Full matchup heatmap — {team} vs all teams")
st.markdown(
    "Each cell shows the probability that **" + team + "** faces that opponent in that specific round. "
    "Zero means the bracket structure makes a meeting impossible in that round."
)

heat_rows = []
for opp in all_teams:
    if opp == team:
        continue
    row = {"Team": opp, "Group": next((g for g, ts in GROUPS.items() if opp in ts), "—")}
    for r in ROUNDS:
        key = (team, opp) if team < opp else (opp, team)
        row[r] = mc[r].get(key, 0) / N
    heat_rows.append(row)

heat_df = (
    pd.DataFrame(heat_rows)
    .assign(_max=lambda d: d[ROUNDS].max(axis=1))
    .sort_values("_max", ascending=False)
    .drop(columns="_max")
    .reset_index(drop=True)
)

# Display as styled dataframe
fmt = {r: "{:.1%}" for r in ROUNDS}
st.dataframe(
    heat_df.style
        .format(fmt)
        .background_gradient(subset=ROUNDS, cmap="YlOrRd", vmin=0, vmax=0.20),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ── Head-to-head deep-dive ────────────────────────────────────────────────────
st.subheader("Head-to-head matchup deep-dive")
st.markdown("Pick a second team to see the full probability breakdown between the two.")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Team A:** {team}")
with col2:
    opponent = st.selectbox(
        "Team B",
        [t for t in all_teams if t != team],
        index=0,
    )

opp_group = next((g for g, ts in GROUPS.items() if opponent in ts), "—")

rows_h2h = []
for r in ROUNDS:
    key = (team, opponent) if team < opponent else (opponent, team)
    count = mc[r].get(key, 0)
    rows_h2h.append({
        "Round":       ROUND_LABELS[r],
        "Probability": count / N,
        "Count":       count,
    })

h2h_df = pd.DataFrame(rows_h2h)
total_any_round = h2h_df["Probability"].sum()

st.markdown(
    f"**{team}** (Group {group_of}) vs **{opponent}** (Group {opp_group})  \n"
    f"Chance they meet in *any* round: **{total_any_round:.1%}**"
)

m1, m2, m3, m4, m5 = st.columns(5)
for col, (_, row) in zip([m1, m2, m3, m4, m5], h2h_df.iterrows()):
    col.metric(row["Round"], f"{row['Probability']:.1%}")

fig, ax = plt.subplots(figsize=(8, 3))
bar_colors = ["#2ca02c" if p > 0.05 else "#aec7e8" for p in h2h_df["Probability"]]
ax.bar(h2h_df["Round"], h2h_df["Probability"], color=bar_colors, edgecolor="white")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.set_title(f"{team} vs {opponent} — matchup probability by round", fontsize=11)
for i, v in enumerate(h2h_df["Probability"]):
    if v > 0:
        ax.text(i, v + 0.002, f"{v:.1%}", ha="center", fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close()
