# FIFA World Cup 2026 Prediction Model — Project Handout

**Author:** Jadiel Montero — Freelance Data Scientist
**Contact:** [LinkedIn](https://www.linkedin.com/in/jadiel-montero-230814142/) | [Upwork](https://www.upwork.com/freelancers/~01d2e510be92ca3e00?mp_source=share)

---

## 1. Business Question

> Which team is most likely to win the 2026 FIFA World Cup, and what is each team's probability of reaching each stage of the tournament?

The goal was not just to pick a winner but to quantify uncertainty: attach a probability to every outcome so the results are actionable for media, betting analysis, or fan engagement.

---

## 2. Dataset

**Source:** [International football results 1872–present](https://github.com/martj42/international_results), maintained by Mart Jürisoo. Updated after every international match, publicly available.

| Property | Value |
|---|---|
| Matches | 48,000+ |
| Date range | 1872 – present |
| Features | Home team, away team, score, tournament, neutral venue flag |
| Target | Match outcome (home win / draw / away win) |

**Cleaning steps:**
- Dropped rows where scores were missing (future scheduled matches)
- Cast scores to integers
- Filtered training data to 1970–present (pre-modern era Elo too noisy)
- Time-based train/test split at January 2018 — no look-ahead bias

---

## 3. The Elo Rating System

Elo is a method for calculating relative skill levels between competitors. Originally designed for chess, it has been widely adopted in football analytics.

### How it works

After every match, ratings are updated:

```
new_rating = old_rating + K × weight × (actual − expected)
```

Where:
- **K** is the tournament importance factor
- **weight** is the goal difference multiplier
- **actual** is 1 (win), 0.5 (draw), or 0 (loss)
- **expected** = 1 / (1 + 10^(−Elo_diff / 400))

### Design decisions

| Factor | Implementation |
|---|---|
| K-factor (FIFA World Cup) | 60 |
| K-factor (Continental championships) | 50 |
| K-factor (Qualifiers) | 40 |
| K-factor (Friendlies) | 20 |
| Home advantage | +100 Elo to home team's expected rating |
| Goal diff multiplier | 1× (1 goal), 1.5× (2 goals), 1.75× (3+), scaling above that |

Every team starts at 1500 and drifts toward their true level over time. After 150 years of matches, the ratings are highly stable and informative.

**Top teams at tournament start:**
Argentina (2160), Spain (2151), France (2138), Germany (2036), Turkey (2025)

---

## 4. Match Outcome Model

### Feature engineering

The model predicts P(home team wins) given:

| Feature | Description |
|---|---|
| `elo_diff` | Home Elo minus away Elo |
| `home_adv` | 1 if match is not on neutral ground, 0 otherwise |
| `elo_diff_sq` | Squared Elo difference (captures non-linearity at extremes) |
| `elo_diff_abs` | Absolute Elo difference |

Draws were excluded — the model predicts win/loss probability. Draw probability is modeled separately in the group stage simulation using a declining exponential formula.

### Model selection

Four algorithms were trained and compared on the same time-based split:

| Model | ROC-AUC | Brier Score |
|---|---|---|
| **Logistic Regression** | **0.8534** | **0.1499** |
| Random Forest | 0.8532 | 0.1506 |
| XGBoost | 0.8528 | 0.1502 |
| Gradient Boosting | 0.8527 | 0.1504 |

All four models scored within **0.0007 AUC** of each other. This is not a coincidence — the relationship between Elo difference and win probability is smooth and monotonic. A logistic curve is the correct shape; tree-based models offer no advantage. More complex does not mean better in this setting.

**Logistic Regression was selected** for three reasons:
1. Best performance on both metrics
2. Outputs naturally calibrated probabilities — critical for downstream applications that price outcomes
3. Interpretable and auditable — one coefficient per feature

### Validation methodology

Training: all matches before January 2018
Testing: all matches from January 2018 onward

This mirrors real production conditions: the model is always predicting future events from past data. Cross-validation was avoided here because it would mix future information into training.

---

## 5. Feature Enrichment

The base model captures long-run team strength via Elo. For the 2026 simulation, two short-run signals were added to the simulation inputs — not to the model — using only data already available in the dataset.

### Feature 1: Recent Form

For each of the 48 tournament teams, weighted points-per-game in their last 10 international matches (2025 onward):
- Win = 3 pts, Draw = 1 pt, Loss = 0 pts
- Linear recency weighting: most recent match = highest weight
- Normalized to an Elo adjustment of ±75 points

### Feature 2: World Cup Psychology Proxy

For each team, compare their World Cup win rate (1990–present) to their overall international win rate in the same period:
- Positive difference = historically over-performs under World Cup pressure
- Negative difference = historically under-performs in World Cups relative to Elo
- Scaled to an Elo adjustment of ±40 points

This is a proxy for the psychological dimension of tournament football — "big-game mentality." It is a correlation, not a proven causal mechanism, and is documented as such.

### Impact on results

| Team | Base Elo | Adj. Elo | Change |
|---|---|---|---|
| Argentina | ~2080 | 2160 | +80 |
| France | ~2060 | 2138 | +78 |
| Spain | ~2200 | 2151 | −49 |
| Germany | ~1990 | 2036 | +46 |

France and Argentina moved up significantly due to strong WC track records. Spain's base Elo was highest, but their relative WC psychology adjustment was smaller.

---

## 6. Tournament Simulation

### Group stage

- 12 official groups (from the real FIFA draw, December 5 2025)
- Each team plays all 3 others in their group (round-robin)
- For each match, Win/Draw/Loss probabilities are computed from enriched Elo difference:
  - `P(draw) = 0.25 × exp(−|elo_diff| / 500)`
  - `P(win) = logistic(elo_diff) × (1 − P(draw))`
- Top 2 from each group qualify automatically (24 teams)
- Best 8 of 12 third-place finishers also qualify (8 teams)
- Total: 32 teams advance to the knockout stage

### Knockout stage

Round of 32 → Round of 16 → Quarter-finals → Semi-finals → 3rd Place + Final

For each knockout match, win probability comes from the trained logistic regression model using the four-feature input. No draws — extra time and penalties treated as an Elo-weighted coin flip.

### Simulation

The full tournament is simulated **10,000 times**. Each run is independent with a fresh random seed. The output is the empirical distribution of outcomes: the fraction of runs in which each team finishes 1st, reaches the final, reaches the semi-final, and so on.

10,000 runs gives probabilities stable to within ±0.5% for most teams.

---

## 7. Key Results

**Championship probabilities (10,000 simulations, enriched inputs):**

| Team | Adj. Elo | Champion | Runner-up | 3rd Place |
|---|---|---|---|---|
| Argentina | 2160 | 23.2% | 11.9% | 10.2% |
| Spain | 2151 | 22.0% | 10.9% | 10.7% |
| France | 2138 | 18.0% | 11.6% | 10.1% |
| Germany | 2036 | 5.2% | 6.9% | 6.3% |
| Turkey | 2025 | 4.2% | 5.2% | 5.7% |
| Brazil | 2019 | 3.7% | 5.6% | 5.8% |
| Colombia | 2017 | 3.6% | 5.5% | 5.7% |
| England | 2019 | 3.6% | 6.3% | 6.0% |

**Head-to-head examples:**
- Argentina beats Spain 51.3% of the time
- France beats Brazil 59.2% of the time
- Argentina beats England 70.1% of the time

---

## 8. Deployment Analysis

### Is this model production-ready?

**Verdict: Yes, with defined limitations.**

| Criterion | Assessment |
|---|---|
| Predictive accuracy | ROC-AUC 0.853 — strong for football, a notoriously noisy sport |
| Probability calibration | Logistic regression is well-calibrated by design |
| Inference speed | Milliseconds — no GPU required |
| Data dependency | One public CSV, updated automatically |
| Interpretability | One coefficient per feature — auditable |
| Staleness risk | Requires re-run after each international window |

### Limitations
- Injuries and suspensions not incorporated
- Elo does not react to coaching changes
- WC psychology proxy has limited sample for new nations
- Knockout bracket seeding is randomised in simulation

---

## 9. Web Application

### Overview

A four-feature interactive web application built in Python (Streamlit), deployable for free on Streamlit Community Cloud.

### Features

**1. Tournament Overview**
Pre-tournament predictions: championship probabilities, group strength charts, full tournament heatmap (all 48 teams × 6 stages), and road-to-the-final curves for the top 12 contenders.

**2. Live Tournament Tracker**
As the World Cup unfolds, users enter match results. The app updates each team's Elo rating using the same K-factor formula, re-runs 10,000 simulations with the updated ratings, and shows how each result shifts the championship probabilities — including a before/after comparison.

**3. Match Predictor**
Any two teams can be selected from a dropdown. The app returns win/draw/loss probabilities for that specific matchup based on current enriched Elo ratings, with a breakdown of what the Elo difference means.

**4. Bracket Builder**
Users select which 32 teams they believe will advance to the knockout stage, then pick winners round by round. For each user-defined bracket state, the app runs the forward simulation and shows championship probabilities for the remaining teams.

### Architecture

```
app/
  streamlit_app.py          Main page (Overview)
  pages/
    1_Live_Tracker.py
    2_Match_Predictor.py
    3_Bracket_Builder.py
  model/
    data.py                 Data loading, group definitions
    elo.py                  Elo computation and enrichment
    predict.py              Model training and inference
    simulate.py             Tournament simulation
  requirements.txt
```

### Tech stack

| Layer | Technology | Reason |
|---|---|---|
| Frontend + Backend | Streamlit | Pure Python, no frontend expertise required, free hosting |
| ML model | scikit-learn LogisticRegression | Lightweight, calibrated, already trained |
| Data | pandas + numpy | Standard data manipulation |
| Visualizations | matplotlib + seaborn | Consistent with the research notebook |
| Hosting | Streamlit Community Cloud | Free, deploys directly from GitHub |

### Running locally

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deployment on Streamlit Cloud

1. Push the repository to GitHub
2. Go to share.streamlit.io and connect the repository
3. Set the main file path to `app/streamlit_app.py`
4. Deploy — Streamlit handles the rest

---

## 10. Potential Applications

| Domain | Use Case |
|---|---|
| Sports media | Automated tournament previews and live probability updates |
| Sports betting | Identify mispriced lines where model diverges from bookmaker odds by >5% |
| Fantasy sports | Inform player selection based on team advancement probabilities |
| National federations | Benchmark competitive position; quantify the Elo gap to the next tier |
| Brand & sponsorship | Quantify expected tournament reach of a potential sponsor team |
| Research | Reproducible baseline for testing more complex football prediction models |
