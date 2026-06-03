"""Patch world_cup_prediction.ipynb: replace compute_wc_uplift with v2 (recent WC window)."""
import json

NOTEBOOK = r"C:\Users\jadi_\Documents\github-repos\fifa-world-cup-prediction\notebooks\world_cup_prediction.ipynb"

NEW_SOURCE = """\
DATASET_NAME = {
    'Czechia': 'Czech Republic', 'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Ivory Coast': "Côte d'Ivoire", 'Curacao': 'Curaçao',
}
def get_dataset_name(team):
    return DATASET_NAME.get(team, team)

# ── Recent WC window (v2 feature) ─────────────────────────────────────────────
# Use only the last 3 World Cups instead of all history since 1990.
# Prevents old titles (e.g. Brazil 2002) from dominating predictions in later
# tournaments where recent WC form is poor.
WC_START_DATES = pd.to_datetime([
    '1990-06-08', '1994-06-17', '1998-06-10', '2002-05-31',
    '2006-06-09', '2010-06-11', '2014-06-12', '2018-06-14', '2022-11-20',
])
N_WCS        = 3
past_wcs     = sorted([d for d in WC_START_DATES if d < df['date'].max()])
window_start = past_wcs[-N_WCS]   # 2014-06-12 for 2026 prediction

# Feature 1: Recent Form — weighted points-per-game, last 10 matches (2025+)
recent_df = df[df['date'] >= '2025-01-01']

def compute_form_adj(wc_name, n=10):
    team    = get_dataset_name(wc_name)
    matches = recent_df[
        (recent_df['home_team'] == team) | (recent_df['away_team'] == team)
    ].sort_values('date').tail(n)
    if len(matches) == 0:
        return 0.0
    pts, weights = [], []
    for i, (_, row) in enumerate(matches.iterrows()):
        w = (i + 1) / len(matches)
        if row['home_team'] == team:
            p = 3 if row['home_score'] > row['away_score'] else (1 if row['home_score'] == row['away_score'] else 0)
        else:
            p = 3 if row['away_score'] > row['home_score'] else (1 if row['home_score'] == row['away_score'] else 0)
        pts.append(p); weights.append(w)
    return (np.average(pts, weights=weights) - 1.5) * 50  # ±75 Elo max

# Feature 2: WC Psychology Proxy — WC win rate vs overall win rate (last 3 WC window)
wc_hist  = df[df['tournament'].str.contains('FIFA World Cup', na=False) &
              ~df['tournament'].str.contains('ualif', case=False, na=False) &
              (df['date'] >= window_start)]
all_hist = df[df['date'] >= window_start]

def compute_wc_uplift(wc_name):
    team = get_dataset_name(wc_name)
    def wr(subset):
        sub = subset[(subset['home_team'] == team) | (subset['away_team'] == team)]
        if len(sub) < 5: return None
        home_wins = ((sub['home_team'] == team) & (sub['home_score'] > sub['away_score'])).sum()
        away_wins = ((sub['away_team'] == team) & (sub['away_score'] > sub['home_score'])).sum()
        return (home_wins + away_wins) / len(sub)
    wc_r  = wr(wc_hist[(wc_hist['home_team'] == team) | (wc_hist['away_team'] == team)])
    all_r = wr(all_hist[(all_hist['home_team'] == team) | (all_hist['away_team'] == team)])
    if wc_r is None or all_r is None: return 0.0
    return float(np.clip((wc_r - all_r) * 300, -40, 40))

form_adj = {t: compute_form_adj(t) for t in all_teams}
wc_adj   = {t: compute_wc_uplift(t) for t in all_teams}
print(f"WC window start:        {window_start.date()} (last {N_WCS} tournaments: 2014, 2018, 2022)")
print(f"Form adj range:     {min(form_adj.values()):+.1f} to {max(form_adj.values()):+.1f}")
print(f"WC uplift range:    {min(wc_adj.values()):+.1f} to {max(wc_adj.values()):+.1f}")
"""

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = 0
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    # Target the cell that defines compute_wc_uplift and wc_adj
    if "compute_wc_uplift" in src and "wc_adj" in src and "compute_form_adj" in src:
        cell["source"] = NEW_SOURCE.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        patched += 1

print(f"Patched {patched} cell(s).")

# Also clear outputs of all downstream cells so notebook is clean before re-execution
clearing = False
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "compute_wc_uplift" in src and "wc_adj" in src:
        clearing = True
    if clearing and cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done. Downstream cells cleared.")
