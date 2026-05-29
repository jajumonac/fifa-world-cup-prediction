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

### 3. Full Tournament Simulation (10,000 runs)
- **Group stage:** 12 groups of 4. Win/Draw/Loss probabilities derived from Elo difference. Top 2 per group + 8 best 3rd-place teams advance to Round of 32.
- **Knockout stage:** R32 → R16 → QF → SF → 3rd Place match + Final.
- **Output:** Probability of finishing 1st, 2nd, 3rd, and 4th for all 48 teams.

## Key Results

**Model performance** (tested on 2018–present matches):
- ROC-AUC: **0.853**
- Brier Score: **0.150** (lower is better; 0.25 = random)

**2026 World Cup Final Standings Probabilities (10,000 simulations):**

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

## Data Source

[International football results 1872–present](https://github.com/martj42/international_results) — maintained by Mart Jürisoo. Updated after every international match.

## Stack

- Python 3.x
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## Author

Jadiel Montero — Freelance Data Scientist specializing in predictive models for financial and sports analytics.
[LinkedIn](https://www.linkedin.com/in/jadiel-montero-230814142/) | [Upwork](https://www.upwork.com/freelancers/~01d2e510be92ca3e00?mp_source=share)
