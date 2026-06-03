"""Add squad trajectory (age proxy) feature — v4 — to the backtest notebook."""
import json

NOTEBOOK = r"C:\Users\jadi_\Documents\github-repos\fifa-world-cup-prediction\notebooks\backtest_historical_wc.ipynb"

def code_cell(cell_id, source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }

def md_cell(cell_id, source):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


# ─────────────────────────────────────────────────────────────────────────────

cell_header = md_cell("v4-header", """\
## 9. Feature: Squad Trajectory (Age Proxy)

**Motivation:** The 2018 France miss (rank #8) and the structural gap we identified: \
Elo captures *accumulated* performance, not whether a team is on an upward or downward \
trajectory. A young squad peaking — like France 2018 (Mbappé 19, Kanté, Pogba in form) \
— will be undervalued because their recent *improvement* isn't reflected in the Elo \
level alone.

**Feature: Elo Velocity** — the annual rate of Elo change over the 2 years before the \
tournament. Computed entirely from match history; no external squad data needed.

- **Positive velocity** → team is improving → often a young squad coming into form
- **Negative velocity** → team is declining → often an aging squad past peak
- Scaled to ±30 Elo (smaller than form ±75 or WC uplift ±40 — trajectory is a softer signal)

This is a **proxy** for squad age dynamics. True squad age would require a player-level \
dataset (e.g., tournament squad databases), but Elo trajectory captures the same \
underlying effect from match results alone.

**v4 = v2 (recent WC window) + Elo velocity**
""")

cell_v4_funcs = code_cell("v4-funcs", """\
def compute_elo_with_snapshot(df, cutoff, snapshot_lookback_years=2):
    \"\"\"Single-pass Elo computation that also captures a mid-point snapshot.

    Returns (elo_current, elo_snapshot, match_df) where elo_snapshot is the
    Elo dict as of (cutoff - snapshot_lookback_years). Used to compute velocity
    without a second full pass through the dataset.
    \"\"\"
    snapshot_date = cutoff - pd.DateOffset(years=snapshot_lookback_years)
    hist = df[df['date'] < cutoff].sort_values('date')
    elo = {}
    elo_snapshot = {}
    snapshot_taken = False
    match_rows = []

    for _, row in hist.iterrows():
        if not snapshot_taken and row['date'] >= snapshot_date:
            elo_snapshot = elo.copy()
            snapshot_taken = True

        h, a = row['home_team'], row['away_team']
        if h not in elo: elo[h] = 1500.0
        if a not in elo: elo[a] = 1500.0

        he, ae = elo[h], elo[a]
        ha = 0 if row['neutral'] else 100
        exp_h = 1 / (1 + 10 ** (-(he + ha - ae) / 400))
        actual = (1.0 if row['home_score'] > row['away_score']
                  else 0.5 if row['home_score'] == row['away_score'] else 0.0)
        gd = abs(row['home_score'] - row['away_score'])
        delta = get_k_factor(row['tournament']) * goal_diff_multiplier(gd) * (actual - exp_h)
        elo[h] += delta
        elo[a] -= delta
        match_rows.append({'date': row['date'], 'neutral': row['neutral'],
                           'elo_diff': he - ae, 'result': actual})

    if not snapshot_taken:
        elo_snapshot = elo.copy()   # no data before snapshot_date

    return elo, elo_snapshot, pd.DataFrame(match_rows)


def compute_elo_velocity(wc_name, elo_current, elo_snapshot, lookback_years=2):
    \"\"\"Annual Elo improvement rate over the past lookback_years → momentum adjustment.

    Scale: ±200 Elo/year change maps to the ±30 max adjustment.
    A typical strong improvement of +75/yr → +11 Elo boost.
    \"\"\"
    team     = get_dataset_name(wc_name)
    elo_now  = elo_current.get(team, 1500.0)
    elo_then = elo_snapshot.get(team, elo_now)   # default: no change
    annual_rate = (elo_now - elo_then) / lookback_years
    return float(np.clip(annual_rate * 0.15, -30, 30))


def build_team_elos_v4(teams, elo_current, elo_snapshot, df, cutoff, n_wcs=3):
    \"\"\"v2 enrichment + Elo velocity (squad trajectory / age proxy).\"\"\"
    result = {}
    for t in teams:
        ds   = get_dataset_name(t)
        base = elo_current.get(ds, elo_current.get(t, 1500.0))
        form = compute_form_adj(t, df, cutoff)
        wc   = compute_wc_uplift_v2(t, df, cutoff, n_wcs=n_wcs)
        vel  = compute_elo_velocity(t, elo_current, elo_snapshot)
        result[t] = base + form + wc + vel
    return result
""")

cell_v4_diagnostic = code_cell("v4-diagnostic", """\
# Show velocity for all 2018 WC teams — the key question is: does France get a boost?
year_diag = 2018
meta_d    = TOURNAMENTS[year_diag]
cutoff_d  = meta_d['start']
teams_d   = [t for g in meta_d['groups'].values() for t in g]

elo_cur, elo_snap, _ = compute_elo_with_snapshot(df, cutoff_d, snapshot_lookback_years=2)

diag_rows = []
for t in teams_d:
    ds   = get_dataset_name(t)
    now  = elo_cur.get(ds,  elo_cur.get(t,  1500.0))
    then = elo_snap.get(ds, elo_snap.get(t, 1500.0))
    vel  = compute_elo_velocity(t, elo_cur, elo_snap)
    diag_rows.append({
        'Team':         t,
        'Elo (2016)':   int(then),
        'Elo (2018)':   int(now),
        '2yr change':   int(now - then),
        'Velocity adj': round(vel, 1),
    })

diag_df = pd.DataFrame(diag_rows).sort_values('Velocity adj', ascending=False)
print('Squad Trajectory — 2018 World Cup  (2-year Elo velocity, 2016→2018)\\n')
print(diag_df.to_string(index=False))
""")

cell_v4_backtest = code_cell("v4-backtest", """\
backtest_results_v4 = {}

for year, meta in sorted(TOURNAMENTS.items()):
    cutoff    = meta['start']
    groups    = meta['groups']
    all_teams = [t for g in groups.values() for t in g]

    elo_cur, elo_snap, match_df = compute_elo_with_snapshot(df, cutoff, snapshot_lookback_years=2)
    model     = train_model_at_cutoff(match_df)
    team_elos = build_team_elos_v4(all_teams, elo_cur, elo_snap, df, cutoff, n_wcs=3)
    probs     = sim_historical_wc(groups, team_elos, model, n_sims=N_SIMS)
    backtest_results_v4[year] = probs

    ranked = sorted(probs.items(), key=lambda x: -x[1])
    actual = meta['champion']
    rank   = next(i + 1 for i, (t, _) in enumerate(ranked) if t == actual)
    top5   = [(t, f'{p:.1%}') for t, p in ranked[:5]]
    print(f'{year}  Champion: {actual} (rank #{rank}, {probs[actual]:.1%}) | '
          f'Top pick: {ranked[0][0]} ({ranked[0][1]:.1%})')
    print(f'  Top 5: {top5}')
    print()

print('Done.')
""")

cell_v4_compare = code_cell("v4-compare", """\
all_versions_full = [
    (backtest_results,    'v1: All WC history'),
    (backtest_results_v2, 'v2: Recent WC window'),
    (backtest_results_v3, 'v3: Decay + Recent'),
    (backtest_results_v4, 'v4: v2 + Trajectory'),
]

print('Full Comparison: v1 / v2 / v3 / v4\\n')
for year in sorted(TOURNAMENTS):
    actual = TOURNAMENTS[year]['champion']
    print(f'  {year} — champion: {actual}')
    for results, lbl in all_versions_full:
        ranked   = sorted(results[year].items(), key=lambda x: -x[1])
        rank     = next(i + 1 for i, (t, _) in enumerate(ranked) if t == actual)
        top_pick = ranked[0][0]
        prob     = results[year][actual]
        correct  = 'CORRECT' if top_pick == actual else f'picked {top_pick}'
        print(f'    {lbl:<24} rank #{rank} ({prob:.1%}) — {correct}')
    print()

print('Aggregate Metrics')
print(f'  {"Version":<24}  Top-1  Top-3  Avg rank  Avg champion prob')
print('  ' + '-' * 60)
for results, lbl in all_versions_full:
    ranks_l, probs_l, top1, top3 = [], [], 0, 0
    for year in sorted(TOURNAMENTS):
        actual  = TOURNAMENTS[year]['champion']
        ranked  = sorted(results[year].items(), key=lambda x: -x[1])
        rank    = next(i + 1 for i, (t, _) in enumerate(ranked) if t == actual)
        ranks_l.append(rank)
        probs_l.append(results[year][actual])
        if rank == 1: top1 += 1
        if rank <= 3: top3 += 1
    print(f'  {lbl:<24}  {top1}/4    {top3}/4    {np.mean(ranks_l):.1f}       {np.mean(probs_l):.1%}')
""")

cell_v4_viz = code_cell("v4-viz", """\
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()

for ax, (results, title) in zip(axes, all_versions_full):
    years_s = sorted(TOURNAMENTS)
    champs  = [TOURNAMENTS[y]['champion'] for y in years_s]
    champ_p = [results[y][TOURNAMENTS[y]['champion']] for y in years_s]
    top_p   = [sorted(results[y].items(), key=lambda x: -x[1])[0][1] for y in years_s]
    ranks_l = []
    for y in years_s:
        ranked = sorted(results[y].items(), key=lambda x: -x[1])
        rank   = next(i + 1 for i, (t, _) in enumerate(ranked)
                      if t == TOURNAMENTS[y]['champion'])
        ranks_l.append(rank)

    x, w = np.arange(len(years_s)), 0.35
    ax.bar(x - w/2, top_p,   w, label='Top pick', color='#4c72b0', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, champ_p, w, label='Champion',  color='#e07b39', alpha=0.85, edgecolor='white')

    for xi, (tp, cp) in enumerate(zip(top_p, champ_p)):
        ax.text(xi - w/2, tp + 0.009, f'{tp:.0%}', ha='center', fontsize=7.5)
        ax.text(xi + w/2, cp + 0.009, f'{cp:.0%}', ha='center', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{y}\\n({c[:3]}, rank #{r})' for y, c, r in zip(years_s, champs, ranks_l)],
        fontsize=9,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.65)

plt.suptitle(
    'Model Evolution: v1 → v2 → v3 → v4 (Squad Trajectory)\\n'
    'Blue = top-pick prob | Orange = actual champion prob | (rank) = champion rank',
    fontsize=12,
)
plt.tight_layout()
plt.show()
""")


# ── Load notebook, strip old v4 cells (idempotent), insert new ones ───────────
with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

v4_ids = {"v4-header", "v4-funcs", "v4-diagnostic", "v4-backtest", "v4-compare", "v4-viz"}
nb["cells"] = [c for c in nb["cells"] if c.get("id") not in v4_ids]

new_cells = [cell_header, cell_v4_funcs, cell_v4_diagnostic,
             cell_v4_backtest, cell_v4_compare, cell_v4_viz]

last_idx = next(
    (i for i, c in enumerate(nb["cells"]) if c.get("id") == "354faa98"),
    len(nb["cells"]),
)
nb["cells"] = nb["cells"][:last_idx] + new_cells + nb["cells"][last_idx:]

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added {len(new_cells)} cells. Total cells: {len(nb['cells'])}")
