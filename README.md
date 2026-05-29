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
Logistic regression trained on Elo rating differences to predict win probability for any matchup. Validated using backtesting on previous World Cups.

### 3. 2026 Tournament Simulation
Monte Carlo simulation (10,000 runs) of the full tournament bracket using current Elo ratings. Output: each team's probability of reaching each stage and winning the tournament.

## Key Results

*(To be updated after model is run)*

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
