"""Print v4 2026 championship probabilities."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from model.data import load_data, get_all_teams, GROUPS
from model.elo import compute_elo_with_snapshot, build_team_elos, compute_elo_velocity, get_dataset_name
from model.predict import train_model
from model.simulate import sim_tournament
import numpy as np
np.random.seed(42)

df = load_data()
elo_ratings, elo_snapshot, match_df = compute_elo_with_snapshot(df)
model, metrics = train_model(match_df)
all_teams = get_all_teams()
team_elos = build_team_elos(all_teams, elo_ratings, df, elo_snapshot=elo_snapshot)

results = sim_tournament(GROUPS, team_elos, model, n_sims=10_000)
positions      = results["positions"]
rounds_reached = results["rounds_reached"]
N = results["n_sims"]

print(f"ROC-AUC: {metrics['roc_auc']}  Brier: {metrics['brier']}")
print(f"\n2026 WC v4 Predictions ({N:,} simulations)\n")
print(f"{'Team':<22} {'Grp':<5} {'AdjElo':>7} {'Champ':>8} {'R-up':>7} {'Final':>8} {'SF':>7}")
print("-" * 70)
for t in sorted(all_teams, key=lambda t: -positions[t][1])[:15]:
    grp = next(g for g, ts in GROUPS.items() if t in ts)
    print(f"{t:<22} {grp:<5} {int(team_elos[t]):>7} {positions[t][1]/N:>8.1%} "
          f"{positions[t][2]/N:>7.1%} {rounds_reached[t]['Final']/N:>8.1%} "
          f"{rounds_reached[t]['SF']/N:>7.1%}")

print("\nVelocity top movers (2024→2026):")
vels = sorted([(t, compute_elo_velocity(t, elo_ratings, elo_snapshot)) for t in all_teams], key=lambda x: -x[1])
for t, v in vels[:6]:
    print(f"  {t:<22} {v:+.1f}")
print("  ...")
for t, v in vels[-6:]:
    print(f"  {t:<22} {v:+.1f}")
