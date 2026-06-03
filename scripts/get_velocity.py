import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from model.data import load_data, get_all_teams, GROUPS
from model.elo import compute_elo_with_snapshot, compute_elo_velocity
import numpy as np
df = load_data()
elo_ratings, elo_snapshot, _ = compute_elo_with_snapshot(df)
all_teams = get_all_teams()
vels = sorted([(t, compute_elo_velocity(t, elo_ratings, elo_snapshot)) for t in all_teams], key=lambda x: -x[1])
print("Top positive trajectory (2024-2026):")
for t, v in vels[:8]:
    print(f"  {t:<22} {v:+.1f}")
print("Top negative trajectory:")
for t, v in vels[-8:]:
    print(f"  {t:<22} {v:+.1f}")
