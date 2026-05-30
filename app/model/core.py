"""Shared initialization — called by every page to ensure model is in session state."""
import streamlit as st
from .data import load_data, get_all_teams, GROUPS
from .elo import compute_elo_ratings, build_team_elos
from .predict import train_model
from .simulate import sim_tournament


@st.cache_resource(show_spinner="Loading 150 years of match data and training model…")
def _load_model_and_data():
    df = load_data()
    elo_ratings, match_df = compute_elo_ratings(df)
    model, metrics = train_model(match_df)
    all_teams = get_all_teams()
    team_elos = build_team_elos(all_teams, elo_ratings, df)
    return df, elo_ratings, model, metrics, team_elos


@st.cache_data(show_spinner="Running 10,000 baseline simulations…")
def _baseline_simulation(_model, _team_elos_frozen, _version="v2"):
    # _version bump forces cache invalidation when sim_tournament changes
    return sim_tournament(GROUPS, dict(_team_elos_frozen), _model, n_sims=10_000)


def ensure_session_state():
    """Idempotent — safe to call at the top of every page."""
    if st.session_state.get("_initialized"):
        return

    df, elo_ratings, model, metrics, team_elos = _load_model_and_data()
    baseline = _baseline_simulation(model, tuple(sorted(team_elos.items())))

    st.session_state._initialized = True
    st.session_state.df = df
    st.session_state.elo_ratings = elo_ratings
    st.session_state.model = model
    st.session_state.metrics = metrics
    st.session_state.team_elos = team_elos
    st.session_state.baseline = baseline

    # Live Tracker state (initialise once, preserved across navigations)
    st.session_state.setdefault("live_elos", team_elos.copy())
    st.session_state.setdefault("match_log", [])
    st.session_state.setdefault("live_results", None)
