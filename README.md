# FIFA World Cup 2026 Prediction Model

A data-driven prediction system for international football match outcomes, built on a custom Elo rating engine and trained on 150+ years of international results. Applied to simulate the 2026 FIFA World Cup and estimate each team's probability of winning.

## Live App

Run locally:
```bash
cd app
streamlit run streamlit_app.py
```

The app includes five pages:
- **Overview** — Championship probabilities, group stage heatmaps, tournament progression for all 48 teams
- **Live Tracker** — Enter real match results to update Elo ratings and re-run 10,000 simulations live
- **Match Predictor** — Head-to-head win/draw/loss probabilities for any two teams
- **Bracket Builder** — Build a custom Round of 32 and simulate or pick round by round
- **Knockout Explorer** — Likely opponents and matchup probabilities for any team at each stage

---

## Business Question

Which team is most likely to win the 2026 FIFA World Cup, and what probability does each contender have of lifting the trophy?

---

## Approach

### 1. Elo Rating System (built from scratch)
Elo ratings are updated after every international match since 1872. Weights account for:
- Tournament importance (World Cup > confederation > qualifier > friendly)
- Goal difference (margin of victory multiplier)
- Home advantage (+100 Elo for non-neutral venues)

### 2. Match Outcome Model
Logistic regression trained on Elo rating differences to predict win probability for any matchup. Validated on matches from 2018 onward (time-based split, no look-ahead bias).

Four model families tested — Logistic Regression, XGBoost, Random Forest, Gradient Boosting — all scored within 0.0007 AUC of each other. The relationship between Elo difference and win probability is smooth and monotonic; a linear model captures it fully.

### 3. Feature Enrichment (Simulation Inputs)
Two features enrich the simulation inputs — no external data required, both derived from the existing match dataset:

- **Recent form** — weighted points-per-game in each team's last 10 matches (2025+), mapped to a ±75 Elo adjustment
- **World Cup psychology proxy** — each team's WC win rate vs overall international win rate across the **last 3 World Cups (2014, 2018, 2022)**. Using a recent sliding window instead of all-time history prevents teams with old titles (e.g., Brazil's 2002 win) from dominating modern predictions. Teams with strong recent WC form receive a boost: France (winner 2018, finalist 2022) and Argentina (winner 2022, finalist 2014) benefit most.

### 4. Full Tournament Simulation (10,000 runs)
- **Official draw:** The actual FIFA group assignments from December 5, 2025
- **Group stage:** 12 groups of 4. Win/draw/loss probabilities from enriched Elo differences. Top 2 per group + 8 best 3rd-place teams advance to the Round of 32
- **Knockout stage:** R32 → R16 → QF → SF → 3rd Place + Final
- **Output:** Champion/runner-up/3rd/4th probabilities for all 48 teams; full stage-by-stage progression

---

## Key Results

**Model performance** (tested on 2018–present matches):

| Metric | Score | Benchmark |
|--------|-------|-----------|
| ROC-AUC | **0.855** | 0.5 = random |
| Brier Score | **0.149** | 0.25 = random, lower is better |

**2026 World Cup Championship Probabilities (10,000 simulations):**

| Team | Group | Adj. Elo | Champion | Runner-up | Reach Final | Reach SF |
|------|-------|---------|---------|---------|---------|---------|
| France | I | 2223 | **25.9%** | 11.2% | 37.1% | 53.3% |
| Spain | H | 2216 | **21.6%** | 9.9% | 31.5% | 51.3% |
| Argentina | J | 2197 | **17.9%** | 14.9% | 32.8% | 48.7% |
| Colombia | K | 2104 | 5.3% | 7.4% | 12.7% | 24.9% |
| Germany | E | 2075 | 4.9% | 6.5% | 11.4% | 22.5% |
| Netherlands | F | 2087 | 4.9% | 5.5% | 10.3% | 20.8% |
| England | L | 2064 | 3.5% | 6.5% | 10.0% | 22.4% |
| Brazil | C | 2041 | 2.3% | 4.9% | 7.2% | 16.7% |

> France leads on the strength of recent WC performance (winner 2018, finalist 2022) and strong current form. Argentina and Spain are close competitors. Brazil ranks 8th despite a high base Elo — three QF/SF exits since 2014 with no title produce a negative WC psychology adjustment that offsets their historical Elo advantage. Full 48-team projections and head-to-head probabilities are available in the app.

---

## Historical Backtesting

To validate the model against ground truth, I backtested against the four most recent World Cups — computing Elo ratings, model training, and feature enrichment using only data available *before* each tournament start date (no look-ahead bias, 10,000 simulations per tournament).

| Year | Actual Champion | Model Rank | Prob. Assigned | Model's Top Pick |
|------|----------------|-----------|---------------|-----------------|
| 2010 (South Africa) | Spain | #2 | 23.6% | Brazil |
| 2014 (Brazil) | Germany | #3 | 13.7% | Brazil |
| 2018 (Russia) | France | #8 | 2.9% | Brazil |
| 2022 (Qatar) | Argentina | **#1 ✓** | 29.9% | **Argentina** |

**Top-3 accuracy: 3/4 (75%)** — the eventual champion appeared in the model's top 3 in three of four tournaments. The 2022 Argentina prediction was correct. The 2018 France miss (rank #8, 2.9%) reflects a structural gap: France's squad transformation with Mbappé and Kanté peaking wasn't captured by historical Elo. Their 2010 and 2014 group-stage exits dragged the rating down at the time.

Full methodology, feature ablation (v1/v2/v3), and per-tournament analysis: [`notebooks/backtest_historical_wc.ipynb`](notebooks/backtest_historical_wc.ipynb)

---

## Deployment Analysis

**Verdict: Production-ready with defined limitations.**

**Strengths:**
- ROC-AUC of 0.855 is strong for binary match outcome prediction in a low-scoring, high-variance sport
- Logistic regression outputs well-calibrated probabilities — essential for any downstream application that prices outcomes
- Time-based validation mirrors real production conditions; no look-ahead leakage
- Fast and interpretable — inference is milliseconds, no GPU required
- No proprietary data — the underlying dataset is publicly maintained and updated after every match

**Deployment recommendations:**
1. **API endpoint** — Wrap match prediction in a FastAPI endpoint returning win/draw/loss probabilities with an Elo snapshot timestamp
2. **Automated Elo refresh** — Schedule a post-match job to recompute Elo after each international window
3. **Prediction monitoring** — Log predicted vs actual outcomes; alert if rolling Brier Score exceeds 0.20
4. **Annual retrain** — Refit logistic regression on the full updated dataset each year
5. **Injury module (v2)** — The largest remaining signal; a "star player absent" flag would meaningfully improve accuracy for high-stakes matches

**Known limitations:**
- Injuries, suspensions, and squad rotation not captured
- Elo doesn't react to coaching changes or tactical shifts between matches
- WC psychology feature is a correlation, not causal; squad transformation effects (a team peaking faster than Elo updates) can be missed, as seen with France 2018

---

## Potential Applications

| Domain | Application |
|--------|-------------|
| **Sports media** | Automated tournament preview and live-update content ("Spain's title odds dropped from 22% to 18% after their group draw") |
| **Sports betting** | Identify mispriced lines where model probability diverges from implied bookmaker odds by >5% |
| **Fantasy sports** | Weight player selection by team's probability of advancing to later rounds |
| **National federations** | Benchmark competitive position; quantify Elo gap to the next tier |
| **Brand & sponsorship** | Model expected tournament reach of a sponsor team before signing deals |
| **Research baseline** | Published Elo + logistic regression benchmark for researchers testing transformer or RL-based football models |

---

## Repository Structure

```
├── notebooks/
│   ├── world_cup_prediction.ipynb      # Full analysis: Elo, model, simulation, visualisations
│   └── backtest_historical_wc.ipynb    # Historical backtest: 2010–2022, feature comparison
├── app/
│   ├── streamlit_app.py                # Overview page
│   ├── pages/
│   │   ├── 1_Live_Tracker.py
│   │   ├── 2_Match_Predictor.py
│   │   ├── 3_Bracket_Builder.py
│   │   └── 4_Knockout_Explorer.py
│   └── model/
│       ├── core.py                     # Session state initialisation
│       ├── data.py                     # Data loading, group draw, WC dates
│       ├── elo.py                      # Elo engine + feature enrichment
│       ├── predict.py                  # Logistic regression training + inference
│       └── simulate.py                 # Monte Carlo tournament simulation
└── HANDOUT.md                          # Technical project documentation
```

---

## Data Source

[International football results 1872–present](https://github.com/martj42/international_results) — maintained by Mart Jürisoo. Updated after every international match.

## Stack

Python 3 · pandas · numpy · scikit-learn · matplotlib · Streamlit

---

## Author

Jadiel Montero — Data Scientist specializing in predictive models for sports and financial analytics.  
[LinkedIn](https://www.linkedin.com/in/jadiel-montero-230814142/) | [Upwork](https://www.upwork.com/freelancers/~01d2e510be92ca3e00?mp_source=share)
