# FIFA World Cup 2026 Prediction Model

A data-driven prediction system for international football match outcomes, built on a custom Elo rating engine and trained on 150+ years of international results. Applied to simulate the 2026 FIFA World Cup and estimate each team's probability of winning.

## Business Question

Which team is most likely to win the 2026 FIFA World Cup, and what probability does each contender have of lifting the trophy?

## Approach

### 1. Elo Rating System (built from scratch)
Elo ratings are updated after every international match since 1872. Weights account for:
- Tournament importance (World Cup > friendly)
- Goal difference (margin of victory)
- Home advantage

### 2. Match Outcome Model
Logistic regression trained on Elo rating differences to predict win probability for any matchup. Validated on matches from 2018 onward (time-based split, no look-ahead bias).

Four model families were tested — Logistic Regression, XGBoost, Random Forest, and Gradient Boosting — and all scored within 0.0007 AUC of each other. The relationship between Elo difference and win probability is smooth and monotonic; a linear model captures it fully.

### 3. Feature Enrichment (Simulation Inputs)
Two features enrich the simulation inputs — no external data required, both derived from the existing match dataset:
- **Recent form** — weighted points-per-game in each team's last 10 matches (2025+), mapped to ±75 Elo adjustment
- **World Cup psychology proxy** — each team's WC win rate vs overall international win rate since 1990; captures teams that historically over/under-perform under tournament pressure

### 4. Full Tournament Simulation (10,000 runs)
- **Official groups:** Uses the actual FIFA draw from December 5, 2025 — no randomised group assignment
- **Group stage:** 12 groups of 4. Win/Draw/Loss probabilities derived from enriched Elo difference. Top 2 per group + 8 best 3rd-place teams advance to Round of 32
- **Knockout stage:** R32 → R16 → QF → SF → 3rd Place match + Final
- **Output:** Probability of finishing 1st, 2nd, 3rd, and 4th for all 48 teams; full stage-by-stage progression for every team

## Key Results

**Model performance** (tested on 2018–present matches):
- ROC-AUC: **0.853**
- Brier Score: **0.150** (lower is better; 0.25 = random)

**2026 World Cup Final Standings Probabilities (10,000 simulations, enriched inputs):**

| Team | Champion | Runner-up | 3rd Place | 4th Place |
|------|----------|-----------|-----------|----------|
| Spain | 28.0% | 10.7% | 10.8% | 2.8% |
| Argentina | 19.1% | 11.2% | 9.8% | 3.5% |
| France | 12.0% | 9.2% | 8.4% | 3.9% |
| England | 7.0% | 6.8% | 6.9% | 4.5% |
| Colombia | 4.1% | 5.3% | 5.3% | 4.6% |
| Brazil | 4.0% | 5.2% | 5.3% | 4.8% |
| Portugal | 3.9% | 5.2% | 5.1% | 4.5% |
| Ecuador | 3.3% | 4.6% | 4.7% | 4.6% |

**Head-to-head examples:**
- Spain beats Argentina 51.9% of the time
- France beats Brazil 58.6% of the time
- Argentina beats England 69.2% of the time
- Mexico beats USA 67.9% of the time

## Deployment Analysis

**Verdict: Production-ready with defined limitations.**

**Strengths for deployment:**
- ROC-AUC of 0.853 is strong for binary match outcome prediction in a noisy, low-scoring sport
- Logistic regression outputs well-calibrated probabilities — essential for any downstream application that prices outcomes
- Time-based validation correctly mirrors production conditions
- Simple, interpretable, fast — inference is milliseconds; no GPU or complex infrastructure needed
- No proprietary data required — the underlying dataset is publicly maintained and updated after every match

**Deployment recommendations:**
1. **API endpoint** — Wrap match prediction in a REST API (FastAPI) returning win/draw/loss probabilities with an Elo snapshot timestamp
2. **Automated Elo refresh** — Schedule a post-match job that recomputes Elo ratings after each international window
3. **Prediction monitoring** — Log predicted vs actual outcomes after each match; alert if Brier Score exceeds 0.20
4. **Annual retrain** — Refit logistic regression on the full updated dataset each year
5. **Injury module (v2)** — The largest remaining signal; a "star player absent" flag would meaningfully improve accuracy for high-stakes matches

## Potential Applications

| Domain | Application |
|--------|-------------|
| **Sports media** | Automated tournament preview and live-update content |
| **Sports betting** | Identify mispriced odds where model probability diverges from bookmaker lines by >5% |
| **Fantasy sports** | Weight player selection by team's probability of advancing to later rounds |
| **National federations** | Benchmark competitive position; quantify Elo gap to the next tier |
| **Brand & sponsorship** | Model expected tournament reach of a sponsor team before signing deals |
| **Research baseline** | Published benchmark for researchers testing more complex models |

## Data Source

[International football results 1872–present](https://github.com/martj42/international_results) — maintained by Mart Jürisoo. Updated after every international match.

## Stack

- Python 3.x
- pandas, numpy, scikit-learn, xgboost
- matplotlib, seaborn
- Jupyter Notebook

## Author

Jadiel Montero — Freelance Data Scientist specializing in predictive models for financial and sports analytics.
[LinkedIn](https://www.linkedin.com/in/jadiel-montero-230814142/) | [Upwork](https://www.upwork.com/freelancers/~01d2e510be92ca3e00?mp_source=share)
