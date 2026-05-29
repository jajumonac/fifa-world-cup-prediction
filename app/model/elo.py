import numpy as np
import pandas as pd
from .data import DATASET_NAME, get_dataset_name


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


def compute_wc_uplift(wc_name: str, df: pd.DataFrame) -> float:
    """WC win rate vs overall win rate (1990+) → Elo adjustment."""
    team = get_dataset_name(wc_name)

    wc_hist = df[
        df["tournament"].str.contains("FIFA World Cup", na=False) &
        ~df["tournament"].str.contains("ualif", case=False, na=False) &
        (df["date"] >= "1990-01-01")
    ]
    all_hist = df[df["date"] >= "1990-01-01"]

    def win_rate(subset: pd.DataFrame):
        sub = subset[(subset["home_team"] == team) | (subset["away_team"] == team)]
        if len(sub) < 5:
            return None
        wins = sum(
            (r.home_team == team and r.home_score > r.away_score) or
            (r.away_team == team and r.away_score > r.home_score)
            for _, r in sub.iterrows()
        )
        return wins / len(sub)

    wc_r = win_rate(wc_hist)
    all_r = win_rate(all_hist)
    if wc_r is None or all_r is None:
        return 0.0
    return float(np.clip((wc_r - all_r) * 300, -40, 40))


def build_team_elos(
    all_teams: list[str],
    elo_ratings: dict,
    df: pd.DataFrame,
) -> dict[str, float]:
    """Return enriched Elo dict for all 48 WC teams."""
    return {
        t: get_elo(t, elo_ratings) + compute_form_adj(t, df) + compute_wc_uplift(t, df)
        for t in all_teams
    }


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
