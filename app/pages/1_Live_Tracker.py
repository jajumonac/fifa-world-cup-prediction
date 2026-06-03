import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model.core import ensure_session_state
from model.data import get_all_teams, GROUPS
from model.elo import update_elo_for_match
from model.simulate import sim_tournament
from model.ui import inject_css, render_footer

st.set_page_config(page_title="Live Tracker", page_icon="📡", layout="wide")
inject_css()
ensure_session_state()

model     = st.session_state.model
baseline  = st.session_state.baseline
all_teams = get_all_teams()

baseline_positions = baseline["positions"]
baseline_n         = baseline["n_sims"]

st.title("📡 Live Tournament Tracker")
st.markdown(
    "Enter match results as the tournament unfolds. "
    "The model updates Elo ratings after each result and re-simulates 10,000 scenarios — "
    "showing how each match shifts each team's championship probability."
)
st.markdown("---")

# ── Enter match result ─────────────────────────────────────────────────────────
st.subheader("Enter a Match Result")

with st.form("match_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
    with c1:
        home = st.selectbox("Home Team", sorted(all_teams))
    with c2:
        home_score = st.number_input("Score", min_value=0, max_value=20, value=0, key="hs")
    with c3:
        away_score = st.number_input("Score", min_value=0, max_value=20, value=0, key="as")
    with c4:
        away = st.selectbox("Away Team", sorted(all_teams))
    neutral   = st.checkbox("Neutral venue (knockout stage)", value=False)
    submitted = st.form_submit_button("Submit Result ➜", type="primary")

if submitted:
    if home == away:
        st.error("Select two different teams.")
    else:
        st.session_state.live_elos = update_elo_for_match(
            home, away, int(home_score), int(away_score),
            st.session_state.live_elos, neutral=neutral,
        )
        outcome = ("Win" if home_score > away_score
                   else "Draw" if home_score == away_score else "Loss")
        st.session_state.match_log.append(
            f"{home} {int(home_score)}–{int(away_score)} {away} ({outcome})"
        )
        st.session_state.live_results = None
        st.success(f"✅  Recorded: {home} {int(home_score)}–{int(away_score)} {away}")

# ── Match log ─────────────────────────────────────────────────────────────────
if st.session_state.match_log:
    with st.expander(f"📋 Match log — {len(st.session_state.match_log)} result(s) entered"):
        for entry in reversed(st.session_state.match_log):
            st.write(f"• {entry}")
    if st.button("↩ Reset all results"):
        st.session_state.live_elos    = st.session_state.team_elos.copy()
        st.session_state.match_log    = []
        st.session_state.live_results = None
        st.rerun()

st.markdown("---")

# ── Run updated simulation ─────────────────────────────────────────────────────
if not st.session_state.match_log:
    st.info("Enter a match result above to see live probability updates.")
    st.stop()

if st.session_state.live_results is None:
    with st.spinner("Getting your predictions..."):
        st.session_state.live_results = sim_tournament(
            GROUPS, st.session_state.live_elos, model, n_sims=10_000
        )

live = st.session_state.live_results
N    = live["n_sims"]

# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("Updated Championship Probabilities")
st.caption("Change column shows the shift since pre-tournament baseline.")

rows = [{
    "Team":     t,
    "Group":    next(g for g, ts in GROUPS.items() if t in ts),
    "Baseline": baseline_positions[t][1] / baseline_n,
    "Updated":  live["positions"][t][1] / N,
    "Change":   live["positions"][t][1] / N - baseline_positions[t][1] / baseline_n,
} for t in all_teams]

cmp_df = (pd.DataFrame(rows)
           .sort_values("Updated", ascending=False)
           .reset_index(drop=True))

st.dataframe(
    cmp_df.style
        .format({"Baseline": "{:.1%}", "Updated": "{:.1%}", "Change": "{:+.1%}"})
        .background_gradient(subset=["Change"], cmap="RdYlGn", vmin=-0.10, vmax=0.10),
    use_container_width=True,
    hide_index=True,
)

# ── Bar chart: top 12 ──────────────────────────────────────────────────────────
st.subheader("Top 12 — Before vs After")
top12 = cmp_df.head(12)
x, w  = np.arange(len(top12)), 0.35

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_facecolor("#FAFBFF")
fig.patch.set_color("#FAFBFF")
ax.bar(x - w / 2, top12["Baseline"], w, label="Pre-tournament",
       color="#B0C4DE", edgecolor="white")
ax.bar(x + w / 2, top12["Updated"],  w, label="Updated",
       color="#2E86C1", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(top12["Team"], rotation=30, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.set_title("Championship Probability: Pre-tournament vs Updated", fontsize=12,
             fontweight="bold", color="#0A1628", pad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=10)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

render_footer()
