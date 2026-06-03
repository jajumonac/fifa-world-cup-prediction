"""Add the Recent WC Window feature exploration cells to the backtest notebook."""
import json

NOTEBOOK = r"C:\Users\jadi_\Documents\github-repos\fifa-world-cup-prediction\notebooks\backtest_historical_wc.ipynb"

def code_cell(cell_id, source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def md_cell(cell_id, source_lines):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_lines,
    }


# ── New cells ─────────────────────────────────────────────────────────────────

cell_header = md_cell("v2-header", [
    "## 7. Feature Improvement: Recent WC Window\n",
    "\n",
    "**The problem:** `compute_wc_uplift` uses ALL WC data since 1990. "
    "Brazil won in 1994 and 2002, so their historical WC win rate is inflated — "
    "causing the model to pick Brazil in every backtest despite not winning since 2002.\n",
    "\n",
    "**The fix:** Use only the **last 3 WC tournaments** before the cutoff. "
    "This makes the feature sensitive to recent WC form:\n",
    "\n",
    "| Backtest | Window | Key impact |\n",
    "|----------|--------|------------|\n",
    "| 2010 | 1998, 2002, 2006 | Still includes 2002 win — modest change |\n",
    "| 2014 | 2002, 2006, 2010 | Still includes 2002 win — moderate change |\n",
    "| 2018 | 2006, 2010, 2014 | No WC win + 7-1 semi-final exit — large reduction |\n",
    "| 2022 | 2010, 2014, 2018 | No WC win, 7-1 in 2014 — large reduction |\n",
])

cell_v2_funcs = code_cell("v2-funcs", """\
WC_START_DATES = pd.to_datetime([
    '1990-06-08', '1994-06-17', '1998-06-10', '2002-05-31',
    '2006-06-09', '2010-06-11', '2014-06-12', '2018-06-14', '2022-11-20',
])


def compute_wc_uplift_v2(wc_name, df, cutoff, n_wcs=3):
    \"\"\"WC win rate vs overall win rate using only the last n_wcs tournaments.

    Addresses the Brazil dominance bias: old titles (1994, 2002) are excluded
    from the 2018/2022 windows, so recent poor WC form is reflected.
    \"\"\"
    team = get_dataset_name(wc_name)
    past_wcs = sorted([d for d in WC_START_DATES if d < cutoff])
    if not past_wcs:
        return 0.0
    window_start = past_wcs[-min(n_wcs, len(past_wcs))]

    hist     = df[df['date'] < cutoff]
    wc_hist  = hist[
        hist['tournament'].str.contains('FIFA World Cup', na=False) &
        ~hist['tournament'].str.contains('ualif', case=False, na=False) &
        (hist['date'] >= window_start)
    ]
    all_hist = hist[hist['date'] >= window_start]

    def wr(subset):
        sub = subset[(subset['home_team'] == team) | (subset['away_team'] == team)]
        if len(sub) < 5:
            return None
        home_wins = ((sub['home_team'] == team) & (sub['home_score'] > sub['away_score'])).sum()
        away_wins = ((sub['away_team'] == team) & (sub['away_score'] > sub['home_score'])).sum()
        return (home_wins + away_wins) / len(sub)

    wc_r  = wr(wc_hist)
    all_r = wr(all_hist)
    if wc_r is None or all_r is None:
        return 0.0
    return float(np.clip((wc_r - all_r) * 300, -40, 40))


def build_team_elos_v2(teams, elo_ratings, df, cutoff, n_wcs=3):
    result = {}
    for t in teams:
        ds_name = get_dataset_name(t)
        base = elo_ratings.get(ds_name, elo_ratings.get(t, 1500.0))
        form = compute_form_adj(t, df, cutoff)
        wc   = compute_wc_uplift_v2(t, df, cutoff, n_wcs=n_wcs)
        result[t] = base + form + wc
    return result
""".splitlines(keepends=True))

cell_diagnostic = code_cell("v2-diagnostic", """\
# Show how the uplift shifts for 2018 — the most affected tournament
year_diag = 2018
meta_d    = TOURNAMENTS[year_diag]
cutoff_d  = meta_d['start']
teams_d   = [t for g in meta_d['groups'].values() for t in g]
elo_d, _  = compute_elo_at_cutoff(df, cutoff_d)

rows_diag = []
for t in teams_d:
    ds   = get_dataset_name(t)
    base = elo_d.get(ds, elo_d.get(t, 1500.0))
    v1   = compute_wc_uplift(t, df, cutoff_d)
    v2   = compute_wc_uplift_v2(t, df, cutoff_d, n_wcs=3)
    rows_diag.append({
        'Team': t, 'Base Elo': int(base),
        'WC Uplift v1': round(v1, 1),
        'WC Uplift v2': round(v2, 1),
        'Change': round(v2 - v1, 1),
    })

diag_df = pd.DataFrame(rows_diag).sort_values('Change')
print('WC Uplift Shift — 2018 World Cup  (v2 window: 2006, 2010, 2014)\\n')
print(diag_df.to_string(index=False))
""".splitlines(keepends=True))

cell_v2_run = code_cell("v2-backtest", """\
backtest_results_v2 = {}

for year, meta in sorted(TOURNAMENTS.items()):
    cutoff    = meta['start']
    groups    = meta['groups']
    all_teams = [t for g in groups.values() for t in g]

    elo_ratings, match_df = compute_elo_at_cutoff(df, cutoff)
    model     = train_model_at_cutoff(match_df)
    team_elos = build_team_elos_v2(all_teams, elo_ratings, df, cutoff, n_wcs=3)
    probs     = sim_historical_wc(groups, team_elos, model, n_sims=N_SIMS)
    backtest_results_v2[year] = probs

    ranked = sorted(probs.items(), key=lambda x: -x[1])
    actual = meta['champion']
    rank   = next(i + 1 for i, (t, _) in enumerate(ranked) if t == actual)
    print(f'{year}  Champion: {actual} (rank #{rank}, {probs[actual]:.1%}) | '
          f'Top pick: {ranked[0][0]} ({ranked[0][1]:.1%})')

print('\\nDone.')
""".splitlines(keepends=True))

cell_compare_table = code_cell("v2-compare-table", """\
comp_rows = []
for year in sorted(TOURNAMENTS):
    actual = TOURNAMENTS[year]['champion']
    r1 = sorted(backtest_results[year].items(),    key=lambda x: -x[1])
    r2 = sorted(backtest_results_v2[year].items(), key=lambda x: -x[1])
    rk1 = next(i + 1 for i, (t, _) in enumerate(r1) if t == actual)
    rk2 = next(i + 1 for i, (t, _) in enumerate(r2) if t == actual)
    comp_rows.append({
        'Year': year, 'Champion': actual,
        'v1 Rank': rk1, 'v1 Prob': f'{backtest_results[year][actual]:.1%}',
        'v1 Top Pick': r1[0][0],
        'v2 Rank': rk2, 'v2 Prob': f'{backtest_results_v2[year][actual]:.1%}',
        'v2 Top Pick': r2[0][0],
        'Rank Δ': rk1 - rk2,
    })

comp_df = pd.DataFrame(comp_rows)
print('Feature Comparison: v1 (all WC history) vs v2 (last 3 WC tournaments)\\n')
print(comp_df.to_string(index=False))

print('\\n─── Aggregate Metrics ───')
for label, results in [('v1 (all WC history)', backtest_results),
                        ('v2 (last 3 WCs)',     backtest_results_v2)]:
    ranks_l, probs_l, top3 = [], [], 0
    for year in sorted(TOURNAMENTS):
        actual  = TOURNAMENTS[year]['champion']
        ranked  = sorted(results[year].items(), key=lambda x: -x[1])
        rank    = next(i + 1 for i, (t, _) in enumerate(ranked) if t == actual)
        ranks_l.append(rank)
        probs_l.append(results[year][actual])
        if rank <= 3:
            top3 += 1
    print(f'  {label}: Top-3={top3}/4 | Avg rank={np.mean(ranks_l):.1f} | '
          f'Avg champion prob={np.mean(probs_l):.1%}')
""".splitlines(keepends=True))

cell_compare_viz = code_cell("v2-compare-viz", """\
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, (result_dict, title) in zip(axes, [
    (backtest_results,    'v1: All WC History (since 1990)'),
    (backtest_results_v2, 'v2: Last 3 WC Tournaments'),
]):
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
    ax.bar(x - w/2, top_p,   w, label="Model's top pick", color='#4c72b0', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, champ_p, w, label='Actual champion',  color='#e07b39', alpha=0.85, edgecolor='white')

    for xi, (tp, cp) in enumerate(zip(top_p, champ_p)):
        ax.text(xi - w/2, tp + 0.008, f'{tp:.0%}', ha='center', fontsize=8)
        ax.text(xi + w/2, cp + 0.008, f'{cp:.0%}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{y}\\n({c}, rank #{r})' for y, c, r in zip(years_s, champs, ranks_l)],
        fontsize=9,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.65)

plt.suptitle(
    'Model Comparison: Original vs Recent WC Window\\n'
    'Blue = model top-pick prob | Orange = actual champion prob | (rank) = champion rank',
    fontsize=12,
)
plt.tight_layout()
plt.show()
""".splitlines(keepends=True))


# ── Load notebook and splice in new cells ─────────────────────────────────────
with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove previously added v2 cells (if re-running this script)
v2_ids = {"v2-header", "v2-funcs", "v2-diagnostic", "v2-backtest",
          "v2-compare-table", "v2-compare-viz"}
nb["cells"] = [c for c in nb["cells"] if c.get("id") not in v2_ids]

# Find insertion point: just before the empty trailing cell
new_cells = [cell_header, cell_v2_funcs, cell_diagnostic,
             cell_v2_run, cell_compare_table, cell_compare_viz]

last_empty_idx = next(
    (i for i, c in enumerate(nb["cells"])
     if c.get("id") == "354faa98"),
    len(nb["cells"]),
)
nb["cells"] = nb["cells"][:last_empty_idx] + new_cells + nb["cells"][last_empty_idx:]

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added {len(new_cells)} cells. Total cells: {len(nb['cells'])}")
