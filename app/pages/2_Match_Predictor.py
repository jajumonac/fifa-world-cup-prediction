import streamlit as st
import matplotlib.pyplot as plt

from model.core import ensure_session_state
from model.data import get_all_teams, GROUPS
from model.predict import wdl_probs
from model.ui import inject_css, render_footer

st.set_page_config(page_title="Match Predictor", page_icon="", layout="wide")
inject_css()
ensure_session_state()

team_elos = st.session_state.team_elos
all_teams = sorted(get_all_teams())

st.markdown('<h1>Match Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#556B8A; font-size:0.92rem; margin-top:0;">'
    'Win / draw / loss probabilities for any head-to-head matchup based on current Elo ratings.'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

c1, c2 = st.columns(2)
with c1:
    team_a = st.selectbox("Team A", all_teams, index=all_teams.index("Argentina"))
with c2:
    team_b = st.selectbox("Team B", all_teams, index=all_teams.index("France"))

neutral = st.checkbox("Neutral venue", value=True)

if team_a == team_b:
    st.warning("Select two different teams.")
    st.stop()

elo_a     = team_elos.get(team_a, 1500.0)
elo_b     = team_elos.get(team_b, 1500.0)
elo_a_eff = elo_a + (0 if neutral else 100)

pw, pd_, pl = wdl_probs(elo_a_eff, elo_b)

st.markdown("---")
st.subheader(f"{team_a}  vs  {team_b}")

m1, m2, m3 = st.columns(3)
m1.metric(f"{team_a} Win", f"{pw:.1%}")
m2.metric("Draw",          f"{pd_:.1%}")
m3.metric(f"{team_b} Win", f"{pl:.1%}")

# ── Probability bar ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 1.3))
fig.patch.set_facecolor("#05172E")
ax.set_facecolor("#05172E")
left = 0
for prob, color, label in [
    (pw,  "#1A6B3A", f"{team_a}  {pw:.1%}"),
    (pd_, "#6B5A10", f"Draw  {pd_:.1%}"),
    (pl,  "#7B0020", f"{team_b}  {pl:.1%}"),
]:
    ax.barh(0, prob, left=left, color=color, edgecolor="#05172E", height=0.55)
    if prob > 0.07:
        ax.text(left + prob / 2, 0, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#FFFFFF")
    left += prob
ax.set_xlim(0, 1)
ax.axis("off")
plt.tight_layout(pad=0.3)
st.pyplot(fig)
plt.close(fig)

# ── Elo breakdown ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Elo Breakdown")
c1, c2, c3 = st.columns(3)
c1.metric(f"{team_a}", int(elo_a_eff),
          delta="+100 home advantage" if not neutral else None)
c2.metric("Elo gap", f"{int(elo_a_eff - elo_b):+d}",
          help="Positive = Team A has the Elo advantage")
c3.metric(f"{team_b}", int(elo_b))

# ── Group context ──────────────────────────────────────────────────────────────
st.markdown("---")
group_a = next((g for g, ts in GROUPS.items() if team_a in ts), None)
group_b = next((g for g, ts in GROUPS.items() if team_b in ts), None)
if group_a and group_b:
    if group_a == group_b:
        st.info(f"Both teams are in Group {group_a} — they will meet in the group stage.")
    else:
        st.info(
            f"{team_a} is in Group {group_a} · "
            f"{team_b} is in Group {group_b} — "
            "they can only meet from the Round of 16 onward."
        )

render_footer()
