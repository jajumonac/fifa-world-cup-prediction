import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model.core import ensure_session_state
from model.data import get_all_teams, GROUPS
from model.predict import wdl_probs
from model.simulate import sim_from_r32
from model.ui import inject_css, render_footer

st.set_page_config(page_title="Bracket Builder", page_icon="", layout="wide")
inject_css()
ensure_session_state()

team_elos = st.session_state.team_elos
model     = st.session_state.model
all_teams = get_all_teams()

st.markdown('<h1>Bracket Builder</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#556B8A; font-size:0.92rem; margin-top:0;">'
    'Pick which 32 teams you think will qualify, then simulate the knockout '
    'or pick winners yourself round by round.'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Step 1 ─────────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Select your 32 qualifiers")

default_r32 = []
for group in GROUPS.values():
    sorted_g = sorted(group, key=lambda t: -team_elos.get(t, 1500))
    default_r32 += sorted_g[:2]

selected = st.multiselect(
    "Select exactly 32 teams:",
    options=sorted(all_teams),
    default=sorted(default_r32),
    max_selections=32,
)

n_sel = len(selected)
if n_sel < 32:
    st.warning(f"Select {32 - n_sel} more team(s) to continue.")
    st.stop()
elif n_sel > 32:
    st.error("Too many teams selected. Remove some.")
    st.stop()
else:
    st.success("32 teams selected.")

st.markdown("---")

# ── Step 2 ─────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Choose simulation mode")
mode = st.radio(
    "",
    ["Simulate (model picks all winners)",
     "Pick winners yourself, round by round"],
    horizontal=True,
)

# ── Mode A: simulate ────────────────────────────────────────────────────────────
if mode.startswith("Simulate"):
    n_sims = st.slider("Number of simulations", 1_000, 20_000, 10_000, step=1_000)

    if st.button("Run simulation", type="primary"):
        with st.spinner("Getting your predictions..."):
            probs = sim_from_r32(selected, team_elos, model, n_sims=n_sims)

        prob_df = (
            pd.DataFrame({"Team": list(probs.keys()), "Champion": list(probs.values())})
            .assign(Group=lambda d: d["Team"].apply(
                lambda t: next((g for g, ts in GROUPS.items() if t in ts), "—")))
            .sort_values("Champion", ascending=False)
            .reset_index(drop=True)
        )

        st.subheader("Championship Probabilities from Your Bracket")
        st.dataframe(
            prob_df[["Team", "Group", "Champion"]].style
                .format({"Champion": "{:.1%}"})
                .background_gradient(subset=["Champion"], cmap="YlOrRd", vmin=0, vmax=0.5),
            use_container_width=True,
            hide_index=True,
        )

        top = prob_df[prob_df["Champion"] > 0.005].head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#0A1F3B")
        fig.patch.set_facecolor("#0A1F3B")
        bar_colors = ["#F0B429" if i == 0 else "#2E6EA8" for i in range(len(top))]
        ax.barh(top["Team"][::-1], top["Champion"][::-1],
                color=bar_colors[::-1], edgecolor="#05172E")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.tick_params(colors="#556B8A")
        ax.set_xticklabels(ax.get_xticklabels(), color="#556B8A")
        ax.set_yticklabels(top["Team"][::-1], fontsize=8.5, color="#C8D8EE")
        ax.set_title("Champion Probability from Your R32 Bracket",
                     fontsize=11, fontweight="bold", color="#FFFFFF", pad=10)
        ax.spines[["top", "right"]].set_color("rgba(255,255,255,0.06)")
        ax.spines[["left", "bottom"]].set_color("rgba(255,255,255,0.06)")
        for i, v in enumerate(top["Champion"].values[::-1]):
            ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=7.5, color="#8BA3C1")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Mode B: manual ──────────────────────────────────────────────────────────────
else:
    ROUND_NAMES = {32: "Round of 32", 16: "Round of 16", 8: "Quarter-finals",
                   4: "Semi-finals", 2: "Final"}

    if st.button("Start / Reset bracket"):
        r = list(selected)
        np.random.shuffle(r)
        st.session_state.bb_teams   = r
        st.session_state.bb_history = []

    if "bb_teams" not in st.session_state:
        st.info("Click Start / Reset bracket to begin.")
        st.stop()

    state     = st.session_state.bb_teams
    n_teams   = len(state)
    round_lbl = ROUND_NAMES.get(n_teams, "Next round")

    if n_teams == 1:
        st.balloons()
        st.success(f"Champion: {state[0]}")
        if st.button("Play again"):
            del st.session_state.bb_teams
            del st.session_state.bb_history
            st.rerun()
        st.stop()

    st.markdown(f"### {round_lbl}")
    st.caption("Pick one winner per match. Win probability shown in parentheses.")

    winners = []
    cols = st.columns(2)
    for idx in range(0, n_teams, 2):
        a, b     = state[idx], state[idx + 1]
        elo_a, elo_b = team_elos.get(a, 1500), team_elos.get(b, 1500)
        pw, _, pl = wdl_probs(elo_a, elo_b)
        col = cols[idx % 2]
        with col:
            pick = st.radio(
                f"{a} ({pw:.0%}) vs {b} ({pl:.0%})",
                [a, b], horizontal=True,
                key=f"bb_{idx}_{n_teams}",
            )
            winners.append(pick)

    if st.button("Confirm picks", type="primary"):
        st.session_state.bb_history.append(state)
        st.session_state.bb_teams = winners
        st.rerun()

    if st.session_state.get("bb_history"):
        if st.button("Undo last round"):
            st.session_state.bb_teams = st.session_state.bb_history.pop()
            st.rerun()

render_footer()
