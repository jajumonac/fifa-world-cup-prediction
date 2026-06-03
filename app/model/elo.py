import numpy as np
import pandas as pd
from .data import DATASET_NAME, WC_START_DATES, get_dataset_name


def get_k_factor(tournament: str) -> int:
    t = tournament.lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 60
    elif any(x in t for x in ["confederation", "continental", "copa america", "euro",
                               "africa cup", "gold cup", "asian cup"]):
        return 50
    elif any(x in t for x in ["qualification", "qualifier"]):
        return 40
    elif "friendly" in t:
        return 20
    return 35


def goal_diff_multiplier(gd: int) -> float:
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    elif gd == 3:
        return 1.75
    return 1.75 + (gd - 3) * 0.05


def compute_elo_ratings(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compute Elo ratings for every team from historical match data.

    Returns (elo_ratings dict, match_df with elo_diff and result columns).
    """
    elo_ratings: dict[str, float] = {}
    match_rows = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in elo_ratings:
            elo_ratings[h] = 1500.0
        if a not in elo_ratings:
            elo_ratings[a] = 1500.0

        he, ae = elo_ratings[h], elo_ratings[a]
        ha = 0 if row["neutral"] else 100
        exp_h = 1 / (1 + 10 ** (-(he + ha - ae) / 400))
        actual = (1.0 if row["home_score"] > row["away_score"]
                  else 0.5 if row["home_score"] == row["away_score"]
                  else 0.0)
        gd = abs(row["home_score"] - row["away_score"])
        delta = get_k_factor(row["tournament"]) * goal_diff_multiplier(gd) * (actual - exp_h)

        elo_ratings[h] += delta
        elo_ratings[a] -= delta

        match_rows.append({
            "date": row["date"],
            "neutral": row["neutral"],
            "elo_diff": he - ae,
            "result": actual,
        })

    return elo_ratings, pd.DataFrame(match_rows)


def get_elo(team: str, elo_ratings: dict) -> float:
    """Look up Elo for a WC team, handling name variants."""
    dataset_name = get_dataset_name(team)
    return elo_ratings.get(dataset_name, elo_ratings.get(team, 1500.0))


def compute_form_adj(wc_name: str, df: pd.DataFrame, n: int = 10) -> float:
    """Weighted points-per-game in last n matches (2025+) → Elo adjustment."""
    team = get_dataset_name(wc_name)
    recent = df[df["date"] >= "2025-01-01"]
    matches = recent[
        (recent["home_team"] == team) | (recent["away_team"] == team)
    ].sort_values("date").tail(n)

    if len(matches) == 0:
        return 0.0

    pts, weights = [], []
    for i, (_, row) in enumerate(matches.iterrows()):
        w = (i + 1) / len(matches)
        if row["home_team"] == team:
            p = (3 if row["home_score"] > row["away_score"]
                 else 1 if row["home_score"] == row["away_score"] else 0)
        else:
            p = (3 if row["away_score"] > row["home_score"]
                 else 1 if row["home_score"] == row["away_score"] else 0)
        pts.append(p)
        weights.append(w)

    return (np.average(pts, weights=weights) - 1.5) * 50  # ±75 Elo max


def compute_elo_with_snapshot(
    df: pd.DataFrame, snapshot_lookback_years: int = 2
) -> tuple[dict, dict, pd.DataFrame]:
    """Compute Elo ratings with a mid-point snapshot for velocity calculation.

    The snapshot is taken at (latest match date − snapshot_lookback_years),
    enabling squad trajectory computation without a second full pass.

    Returns (elo_current, elo_snapshot, match_df).
    """
    snapshot_date = df["date"].max() - pd.DateOffset(years=snapshot_lookback_years)
    elo_ratings: dict[str, float] = {}
    elo_snapshot: dict[str, float] = {}
    snapshot_taken = False
    match_rows = []

    for _, row in df.sort_values("date").iterrows():
        if not snapshot_taken and row["date"] >= snapshot_date:
            elo_snapshot = elo_ratings.copy()
            snapshot_taken = True

        h, a = row["home_team"], row["away_team"]
        if h not in elo_ratings:
            elo_ratings[h] = 1500.0
        if a not in elo_ratings:
            elo_ratings[a] = 1500.0

        he, ae = elo_ratings[h], elo_ratings[a]
        ha = 0 if row["neutral"] else 100
        exp_h = 1 / (1 + 10 ** (-(he + ha - ae) / 400))
        actual = (1.0 if row["home_score"] > row["away_score"]
                  else 0.5 if row["home_score"] == row["away_score"]
                  else 0.0)
        gd = abs(row["home_score"] - row["away_score"])
        delta = get_k_factor(row["tournament"]) * goal_diff_multiplier(gd) * (actual - exp_h)

        elo_ratings[h] += delta
        elo_ratings[a] -= delta
        match_rows.append({
            "date": row["date"],
            "neutral": row["neutral"],
            "elo_diff": he - ae,
            "result": actual,
        })

    if not snapshot_taken:
        elo_snapshot = elo_ratings.copy()

    return elo_ratings, elo_snapshot, pd.DataFrame(match_rows)


def compute_elo_velocity(
    wc_name: str,
    elo_current: dict,
    elo_snapshot: dict,
    lookback_years: int = 2,
) -> float:
    """Annual Elo improvement rate over the past lookback_years → trajectory adjustment.

    Positive = improving team (often young squad coming into form).
    Negative = declining team (often aging squad past peak).
    Scaled to ±30 Elo: a ±200 Elo/year annual change hits the cap.
    """
    team     = get_dataset_name(wc_name)
    elo_now  = elo_current.get(team, 1500.0)
    elo_then = elo_snapshot.get(team, elo_now)
    annual_rate = (elo_now - elo_then) / lookback_years
    return float(np.clip(annual_rate * 0.15, -30, 30))


def compute_wc_uplift(wc_name: str, df: pd.DataFrame, n_wcs: int = 3) -> float:
    """WC win rate vs overall win rate for the last n_wcs tournaments → Elo adjustment.

    Uses a sliding window of the most recent n_wcs World Cups rather than all
    history since 1990. This prevents teams with old titles (e.g. Brazil 2002)
    from dominating predictions in later tournaments where their recent WC form
    is poor.
    """
    team = get_dataset_name(wc_name)

    past_wcs = sorted([d for d in WC_START_DATES if d < df["date"].max()])
    if not past_wcs:
        return 0.0
    window_start = past_wcs[-min(n_wcs, len(past_wcs))]

    wc_hist  = df[
        df["tournament"].str.contains("FIFA World Cup", na=False) &
        ~df["tournament"].str.contains("ualif", case=False, na=False) &
        (df["date"] >= window_start)
    ]
    all_hist = df[df["date"] >= window_start]

    def win_rate(subset: pd.DataFrame):
        sub = subset[(subset["home_team"] == team) | (subset["away_team"] == team)]
        if len(sub) < 5:
            return None
        home_wins = ((sub["home_team"] == team) & (sub["home_score"] > sub["away_score"])).sum()
        away_wins = ((sub["away_team"] == team) & (sub["away_score"] > sub["home_score"])).sum()
        return (home_wins + away_wins) / len(sub)

    wc_r  = win_rate(wc_hist)
    all_r = win_rate(all_hist)
    if wc_r is None or all_r is None:
        return 0.0
    return float(np.clip((wc_r - all_r) * 300, -40, 40))


def build_team_elos(
    all_teams: list[str],
    elo_ratings: dict,
    df: pd.DataFrame,
    elo_snapshot: dict | None = None,
) -> dict[str, float]:
    """Return enriched Elo dict for all 48 WC teams.

    When elo_snapshot is provided the squad trajectory (Elo velocity) feature
    is added as a fourth enrichment signal.
    """
    result = {}
    for t in all_teams:
        base = get_elo(t, elo_ratings)
        form = compute_form_adj(t, df)
        wc   = compute_wc_uplift(t, df)
        vel  = compute_elo_velocity(t, elo_ratings, elo_snapshot) if elo_snapshot is not None else 0.0
        result[t] = base + form + wc + vel
    return result


def update_elo_for_match(
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    team_elos: dict,
    neutral: bool = False,
    tournament: str = "FIFA World Cup",
) -> dict:
    """Apply a single match result to team_elos and return the updated dict."""
    updated = team_elos.copy()
    he, ae = updated.get(home, 1500.0), updated.get(away, 1500.0)
    ha = 0 if neutral else 100
    exp_h = 1 / (1 + 10 ** (-(he + ha - ae) / 400))
    actual = (1.0 if home_score > away_score
              else 0.5 if home_score == away_score
              else 0.0)
    gd = abs(home_score - away_score)
    delta = get_k_factor(tournament) * goal_diff_multiplier(gd) * (actual - exp_h)
    updated[home] = he + delta
    updated[away] = ae - delta
    return updated
