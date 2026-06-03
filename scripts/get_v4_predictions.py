"""Print v4 2026 championship probabilities."""
import sys
sys.path.insert(0, ".")

from model.data import load_data, get_all_teams, GROUPS
from model.elo import compute_elo_with_snapshot, build_team_elos
from model.predict import train_model
from model.simulate import sim_tournament

import numpy as np
np.random.seed(42)

print("Loading data...")
df = load_data()
print(f"Matches: {len(df):,}  |  Latest: {df['date'].max().date()}")

print("Computing Elo with snapshot...")
elo_ratings, elo_snapshot, match_df = compute_elo_with_snapshot(df)

print("Training model...")
model, metrics = train_model(match_df)
print(f"ROC-AUC: {metrics['roc_auc']}  Brier: {metrics['brier']}")

all_teams = get_all_teams()
team_elos = build_team_elos(all_teams, elo_ratings, df, elo_snapshot=elo_snapshot)

print("\nRunning 10,000 simulations...")
results = sim_tournament(GROUPS, team_elos, model, n_sims=10_000)
positions      = results["positions"]
rounds_reached = results["rounds_reached"]
N = results["n_sims"]

# Print top-15 sorted by champion probability
ranked = sorted(all_teams, key=lambda t: -positions[t][1])
print(f"\n2026 WC Championship Probabilities (v4, {N:,} simulations)\n")
print(f"{'Team':<22} {'Group':<6} {'Adj.Elo':>8} {'Champion':>10} {'Runner-up':>10} {'Reach Final':>12} {'Reach SF':>10}")
print("-" * 82)
for t in ranked[:15]:
    grp = next(g for g, ts in GROUPS.items() if t in ts)
    print(
        f"{t:<22} {grp:<6} {int(team_elos[t]):>8} "
        f"{positions[t][1]/N:>10.1%} "
        f"{positions[t][2]/N:>10.1%} "
        f"{rounds_reached[t]['Final']/N:>12.1%} "
        f"{rounds_reached[t]['SF']/N:>10.1%}"
    )

print("\n--- Velocity adjustments (top movers) ---")
from model.elo import compute_elo_velocity, get_elo, get_dataset_name
vel_rows = []
for t in all_teams:
    vel = compute_elo_velocity(t, elo_ratings, elo_snapshot)
    vel_rows.append((t, vel))
vel_rows.sort(key=lambda x: -x[1])
print("Biggest positive trajectory (improving teams):")
for t, v in vel_rows[:8]:
    print(f"  {t:<22} {v:+.1f}")
print("Biggest negative trajectory (declining teams):")
for t, v in vel_rows[-8:]:
    print(f"  {t:<22} {v:+.1f}")
