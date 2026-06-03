"""Add time-decayed Elo (v3) exploration cells to the backtest notebook."""
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


# ── New cells ─────────────────────────────────────────────────────────────────

cell_header = md_cell("v3-header", """\
## 8. Feature Improvement: Time-Decayed Elo

**Root cause:** Brazil's base Elo of ~2190 dominates every prediction because the standard Elo \
system treats a match from 1958 the same as a match from 2017. 150 years of accumulated wins \
are permanently baked into the rating.

**The fix:** Apply **annual mean reversion** — each year, pull every team's Elo 10% toward \
1500. This is the same approach used by FiveThirtyEight's club soccer ratings.

The effect:
- Teams that were dominant decades ago (Brazil 1958–1970) lose that historical credit gradually
- Teams in strong current form maintain high ratings (they keep winning, so reversion is offset)
- Elo differences between top teams compress, making form adjustments more influential
- Combined with v2's recent WC window, this attacks both layers of the Brazil bias
""")

cell_v3_funcs = code_cell("v3-funcs", """\
def compute_elo_at_cutoff_v3(df, cutoff, decay_per_year=0.1):
    \"\"\"Elo with annual mean reversion toward 1500.

    For each team, when they appear in a match, any years elapsed since their
    last match trigger compound reversion: Elo = Elo * (1-d)^n + 1500 * (1-(1-d)^n).
    \"\"\"
    hist = df[df['date'] < cutoff].sort_values('date')
    elo, last_year, match_rows = {}, {}, []

    for _, row in hist.iterrows():
        h, a     = row['home_team'], row['away_team']
        row_year = row['date'].year

        for team in [h, a]:
            if team not in elo:
                elo[team]      = 1500.0
                last_year[team] = row_year
            years_elapsed = row_year - last_year[team]
            if years_elapsed > 0:
                factor     = (1 - decay_per_year) ** years_elapsed
                elo[team]  = elo[team] * factor + 1500.0 * (1 - factor)
            last_year[team] = row_year

        he, ae = elo[h], elo[a]
        ha     = 0 if row['neutral'] else 100
        exp_h  = 1 / (1 + 10 ** (-(he + ha - ae) / 400))
        actual = (1.0 if row['home_score'] > row['away_score']
                  else 0.5 if row['home_score'] == row['away_score'] else 0.0)
        gd     = abs(row['home_score'] - row['away_score'])
        delta  = get_k_factor(row['tournament']) * goal_diff_multiplier(gd) * (actual - exp_h)
        elo[h] += delta
        elo[a] -= delta
        match_rows.append({'date': row['date'], 'neutral': row['neutral'],
                           'elo_diff': he - ae, 'result': actual})

    return elo, pd.DataFrame(match_rows)


def build_team_elos_v3(teams, elo_ratings, df, cutoff, n_wcs=3):
    \"\"\"Decayed base Elo + form adj + recent WC uplift (v2).\"\"\"
    result = {}
    for t in teams:
        ds_name = get_dataset_name(t)
        base    = elo_ratings.get(ds_name, elo_ratings.get(t, 1500.0))
        form    = compute_form_adj(t, df, cutoff)
        wc      = compute_wc_uplift_v2(t, df, cutoff, n_wcs=n_wcs)
        result[t] = base + form + wc
    return result
""")

cell_v3_diagnostic = code_cell("v3-diagnostic", """\
# Compare base Elo: no decay (v1) vs 10% annual decay (v3) for 2018 WC teams
year_diag = 2018
meta_d    = TOURNAMENTS[year_diag]
cutoff_d  = meta_d['start']
teams_d   = [t for g in meta_d['groups'].values() for t in g]

elo_v1_d, _ = compute_elo_at_cutoff(df, cutoff_d)
elo_v3_d, _ = compute_elo_at_cutoff_v3(df, cutoff_d, decay_per_year=0.1)

diag_rows = []
for t in teams_d:
    ds = get_dataset_name(t)
    e1 = elo_v1_d.get(ds, elo_v1_d.get(t, 1500.0))
    e3 = elo_v3_d.get(ds, elo_v3_d.get(t, 1500.0))
    diag_rows.append({
        'Team': t,
        'Base Elo (no decay)': int(e1),
        'Base Elo (10% decay)': int(e3),
        'Change': int(e3 - e1),
    })

diag_df = pd.DataFrame(diag_rows).sort_values('Change')
print('Base Elo Shift — 2018 WC (10% annual decay vs no decay)\\n')
print(diag_df.to_string(index=False))
""")

cell_v3_backtest = code_cell("v3-backtest", """\
DECAY_RATE = 0.1
backtest_results_v3 = {}

for year, meta in sorted(TOURNAMENTS.items()):
    cutoff    = meta['start']
    groups    = meta['groups']
    all_teams = [t for g in groups.values() for t in g]

    elo_ratings, match_df = compute_elo_at_cutoff_v3(df, cutoff, decay_per_year=DECAY_RATE)
    model     = train_model_at_cutoff(match_df)
    team_elos = build_team_elos_v3(all_teams, elo_ratings, df, cutoff, n_wcs=3)
    probs     = sim_historical_wc(groups, team_elos, model, n_sims=N_SIMS)
    backtest_results_v3[year] = probs

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

cell_v3_compare = code_cell("v3-compare", """\
all_versions = [
    (backtest_results,    'v1: All history'),
    (backtest_results_v2, 'v2: Recent WC window'),
    (backtest_results_v3, 'v3: Decay + Recent'),
]

print('Full Comparison: v1 vs v2 vs v3\\n')

for year in sorted(TOURNAMENTS):
    actual = TOURNAMENTS[year]['champion']
    print(f'  {year} — actual champion: {actual}')
    for results, lbl in all_versions:
        ranked   = sorted(results[year].items(), key=lambda x: -x[1])
        rank     = next(i + 1 for i, (t, _) in enumerate(ranked) if t == actual)
        top_pick = ranked[0][0]
        prob     = results[year][actual]
        correct  = 'CORRECT' if top_pick == actual else f'picked {top_pick}'
        print(f'    {lbl:<24} rank #{rank} ({prob:.1%}) — {correct}')
    print()

print('Aggregate Metrics')
print(f'  {"Version":<24}  Top-1  Top-3  Avg rank  Avg prob')
print('  ' + '-' * 55)
for results, lbl in all_versions:
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

cell_v3_viz = code_cell("v3-viz", """\
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

for ax, (result_dict, title) in zip(axes, all_versions):
    years_s = sorted(TOURNAMENTS)
    champs  = [TOURNAMENTS[y]['champion'] for y in years_s]
    champ_p = [result_dict[y][TOURNAMENTS[y]['champion']] for y in years_s]
    top_p   = [sorted(result_dict[y].items(), key=lambda x: -x[1])[0][1] for y in years_s]
    ranks_l = []
    for y in years_s:
        ranked = sorted(result_dict[y].items(), key=lambda x: -x[1])
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
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.65)

plt.suptitle(
    'Model Evolution: v1 (baseline) → v2 (recent WC window) → v3 (decay + recent WC)\\n'
    'Blue = top-pick probability | Orange = actual champion probability | rank = champion rank',
    fontsize=12,
)
plt.tight_layout()
plt.show()
""")


# ── Load notebook and splice in new cells ─────────────────────────────────────
with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove previously added v3 cells (idempotent re-run)
v3_ids = {"v3-header", "v3-funcs", "v3-diagnostic", "v3-backtest", "v3-compare", "v3-viz"}
nb["cells"] = [c for c in nb["cells"] if c.get("id") not in v3_ids]

new_cells = [cell_header, cell_v3_funcs, cell_v3_diagnostic,
             cell_v3_backtest, cell_v3_compare, cell_v3_viz]

# Insert just before the trailing empty cell
last_empty_idx = next(
    (i for i, c in enumerate(nb["cells"]) if c.get("id") == "354faa98"),
    len(nb["cells"]),
)
nb["cells"] = nb["cells"][:last_empty_idx] + new_cells + nb["cells"][last_empty_idx:]

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added {len(new_cells)} cells. Total cells: {len(nb['cells'])}")
