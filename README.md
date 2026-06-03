# FIFA World Cup 2026 Prediction Model

A data-driven prediction system for international football match outcomes, built on a custom Elo rating engine and trained on 150+ years of international results. Applied to simulate the 2026 FIFA World Cup and estimate each team's probability of winning.

## Live App

```bash
cd app
streamlit run streamlit_app.py
```

Five pages: Championship overview · Live Tracker · Match Predictor · Bracket Builder · Knockout Explorer

---

## Business Question

Which team is most likely to win the 2026 FIFA World Cup, and what probability does each contender have of lifting the trophy?

---

## Approach

### 1. Elo Rating System (built from scratch)
Elo ratings are updated after every international match since 1872. Weights account for tournament importance, goal difference, and home advantage.

### 2. Match Outcome Model
Logistic regression trained on Elo rating differences predicts win probability for any matchup. Four model families tested (LR, XGBoost, RF, GBM) — all scored within 0.0007 AUC. A linear model is correct here: the relationship between Elo difference and win probability is smooth and monotonic.

Validated on matches from 2018 onward (time-based split, no look-ahead bias).

### 3. Feature Enrichment (Simulation Inputs)
Three signals enrich the simulation inputs — all derived from existing match history, no external data required:

| Feature | What it measures | Range |
|---------|-----------------|-------|
| **Recent form** | Weighted PPG in last 10 matches (2025+) | ±75 Elo |
| **WC psychology proxy** | WC win rate vs overall win rate, last 3 tournaments (2014–2022) | ±40 Elo |
| **Squad trajectory** | Annual Elo improvement rate over the past 2 years | ±30 Elo |

### 4. Full Tournament Simulation (10,000 runs)
- Official FIFA draw (December 5, 2025): 12 groups of 4
- Top 2 per group + 8 best third-place teams advance to Round of 32
- Knockout: R32 → R16 → QF → SF → 3rd Place + Final

---

## Key Results

**Model performance** (tested on 2018–present matches):

| Metric | Score | Benchmark |
|--------|-------|-----------|
| ROC-AUC | **0.855** | 0.5 = random |
| Brier Score | **0.149** | 0.25 = random, lower is better |

**2026 World Cup Championship Probabilities (10,000 simulations, v4 enrichment):**

| Team | Group | Adj. Elo | Champion | Runner-up | Reach Final | Reach SF |
|------|-------|---------|---------|---------|---------|---------|
| France | I | 2222 | **25.7%** | 10.4% | 36.1% | 53.7% |
| Spain | H | 2227 | **23.8%** | 10.6% | 34.4% | 53.7% |
| Argentina | J | 2196 | **17.6%** | 15.3% | 32.9% | 48.0% |
| Colombia | K | 2103 | 5.0% | 7.6% | 12.6% | 24.1% |
| Germany | E | 2075 | 4.4% | 5.8% | 10.2% | 21.5% |
| Netherlands | F | 2085 | 4.3% | 5.4% | 9.8% | 20.4% |
| England | L | 2065 | 3.5% | 6.8% | 10.3% | 22.2% |
| Brazil | C | 2039 | 2.2% | 4.8% | 7.0% | 15.9% |

> France and Spain are the clear favorites. Spain's Euro 2024 win gives them the strongest 2-year trajectory (+11 Elo), pushing them neck-and-neck with France. Argentina enters as champion with a strong WC psychology boost. Brazil ranks 8th — three consecutive QF/SF exits since 2014 produce a significant WC psychology penalty that offsets their high base Elo.

---

## Feature Engineering Iterations

The enrichment layer was developed through four iterations, each tested against a historical backtest across the 2010, 2014, 2018, and 2022 World Cups. All Elo computations, model training, and enrichment used only data available *before* each tournament (zero look-ahead bias).

| Version | What changed | Top-3 accuracy | Avg champion prob |
|---------|-------------|---------------|------------------|
| **v1** | Baseline: WC psychology uses all history since 1990 | 3/4 (75%) | 16.8% |
| **v2** | WC psychology window → last 3 tournaments only | 3/4 (75%) | 17.5% |
| **v3** | Time-decayed Elo (10% annual reversion) + v2 | 3/4 (75%) | 16.9% |
| **v4** ✓ | v2 + squad trajectory (2-year Elo velocity) | 3/4 (75%) | **18.4%** |

**What each version revealed:**

- **v1 → v2:** Brazil was ranked #1 in every historical tournament despite not winning since 2002. Root cause: their 1994 and 2002 titles were permanently baked into the WC psychology feature. Fixing the window to the last 3 tournaments removed this bias and correctly predicted Argentina in 2022.

- **v2 → v3:** Time-decayed Elo compresses all ratings proportionally — the relative rankings barely change. The Brazil bias lives in the base Elo, not just the WC uplift. Decay is not the right tool here.

- **v3 → v4:** The squad trajectory feature (annual Elo velocity) adds a meaningful signal. For 2026, Spain's Euro 2024 win produces a strong +11 Elo trajectory boost, moving them into contention with France. For the backtest, Argentina 2022's unbeaten-run trajectory pushes their probability from 29.9% → 32.1%.

**The persistent miss — 2018 France (rank #8, 2.9%):** France had near-zero Elo change from 2016 to 2018 (+3 points) despite transforming their squad. The trajectory feature gave them almost nothing because Mbappé's emergence and Kanté's role are invisible to any metric derived from match results. Fixing this would require player-level ratings — a different data source entirely.

---

## Historical Backtesting

| Year | Actual Champion | Model Rank (v4) | Prob. Assigned | Model's Top Pick |
|------|----------------|----------------|---------------|-----------------|
| 2010 (South Africa) | Spain | #2 | 24.9% | Brazil |
| 2014 (Brazil) | Germany | #3 | 13.6% | Brazil |
| 2018 (Russia) | France | #8 | 2.9% | Brazil |
| 2022 (Qatar) | Argentina | **#1 ✓** | 32.1% | **Argentina** |

Full methodology and per-version comparison: [`notebooks/backtest_historical_wc.ipynb`](notebooks/backtest_historical_wc.ipynb)

---

## Deployment Analysis

**Verdict: Production-ready with defined limitations.**

**Strengths:** ROC-AUC 0.855 on a noisy sport; calibrated probabilities; millisecond inference; no proprietary data; fully interpretable.

**Deployment recommendations:**
1. **API endpoint** — FastAPI wrapper returning win/draw/loss probabilities with Elo timestamp
2. **Automated Elo refresh** — post-match job after each international window
3. **Prediction monitoring** — alert if rolling Brier Score exceeds 0.20
4. **Annual retrain** — refit on full updated dataset each year
5. **Player-level module (v5)** — the remaining signal; squad age / key-player ratings would fix the 2018-type miss

**Known limitations:**
- Injuries and suspensions not captured
- Squad transformation effects (young talent emerging) visible only after results accumulate
- WC psychology feature has limited data for nations with few WC appearances

---

## Potential Applications

| Domain | Application |
|--------|-------------|
| **Sports media** | Automated probability updates after every match ("Spain's title odds shifted from 22% to 24% after their Group H opener") |
| **Sports betting** | Identify mispriced lines where model probability diverges from bookmaker implied odds by >5% |
| **Fantasy sports** | Weight picks by team's probability of advancing to score-heavy later rounds |
| **National federations** | Benchmark competitive position; quantify Elo gap to the next tier |
| **Brand & sponsorship** | Model expected tournament reach of a sponsor team before signing |
| **Research baseline** | Reproducible Elo + LR benchmark for testing transformer or RL-based football models |

---

## Repository Structure

```
├── notebooks/
│   ├── world_cup_prediction.ipynb      # Full pipeline: Elo, model, simulation, visuals
│   └── backtest_historical_wc.ipynb    # Historical validation: 2010–2022, v1→v4
├── app/
│   ├── streamlit_app.py                # Overview page
│   ├── pages/
│   │   ├── 1_Live_Tracker.py
│   │   ├── 2_Match_Predictor.py
│   │   ├── 3_Bracket_Builder.py
│   │   └── 4_Knockout_Explorer.py
│   └── model/
│       ├── core.py                     # Session state initialisation
│       ├── data.py                     # Data loading, groups, WC dates
│       ├── elo.py                      # Elo engine + all enrichment features
│       ├── predict.py                  # Model training + inference
│       └── simulate.py                 # Monte Carlo tournament simulation
└── HANDOUT.md                          # Full technical project documentation
```

---

## Data Source

[International football results 1872–present](https://github.com/martj42/international_results) — Mart Jürisoo. Updated after every international match.

## Stack

Python 3 · pandas · numpy · scikit-learn · matplotlib · Streamlit

---

## Author

Jadiel Montero — Data Scientist specializing in predictive models for sports and financial analytics.  
[LinkedIn](https://www.linkedin.com/in/jadiel-montero-230814142/) | [Upwork](https://www.upwork.com/freelancers/~01d2e510be92ca3e00?mp_source=share)
