# FIFA World Cup 2026 Prediction Model — Project Handout

**Author:** Jadiel Montero — Data Scientist  
**Contact:** [LinkedIn](https://www.linkedin.com/in/jadiel-montero-230814142/) | [Upwork](https://www.upwork.com/freelancers/~01d2e510be92ca3e00?mp_source=share)

---

## 1. Business Question

> Which team is most likely to win the 2026 FIFA World Cup, and what is each team's probability of reaching each stage of the tournament?

The goal was not just to pick a winner but to quantify uncertainty: attach a calibrated probability to every outcome so the results are actionable for media, betting analysis, or fan engagement.

---

## 2. Dataset

**Source:** [International football results 1872–present](https://github.com/martj42/international_results), maintained by Mart Jürisoo. Updated after every international match.

| Property | Value |
|---|---|
| Matches | 49,000+ |
| Date range | 1872 – present |
| Features used | Home team, away team, score, tournament, neutral venue flag |
| Target | Match outcome (home win / draw / away win) |

**Cleaning:**
- Dropped rows with missing scores (future scheduled matches)
- Scores cast to integers
- Training data filtered to 1970–present (pre-modern era Elo too noisy)
- Time-based train/test split at January 2018 — no look-ahead bias

---

## 3. The Elo Rating System

Elo is a method for calculating relative skill levels between competitors. Originally designed for chess, it has been widely adopted in football analytics.

### Update rule

```
new_rating = old_rating + K × weight × (actual − expected)
```

Where:
- **K** = tournament importance factor
- **weight** = goal difference multiplier
- **actual** = 1.0 (win), 0.5 (draw), 0.0 (loss)
- **expected** = 1 / (1 + 10^(−Elo_diff / 400))

### Design decisions

| Factor | Value |
|---|---|
| K-factor (FIFA World Cup) | 60 |
| K-factor (Continental championships) | 50 |
| K-factor (Qualifiers) | 40 |
| K-factor (Friendlies) | 20 |
| Home advantage | +100 Elo added to home team's effective rating |
| Goal diff multiplier | 1× (1 goal) · 1.5× (2) · 1.75× (3) · +0.05 per goal above 3 |

Every team starts at 1500 and drifts toward their true level. After 150 years of matches the ratings are highly stable.

---

## 4. Match Outcome Model

### Features

The model predicts P(home team wins) given:

| Feature | Description |
|---|---|
| `elo_diff` | Home Elo minus away Elo |
| `home_adv` | 1 if match is not on neutral ground |
| `elo_diff_sq` | Squared Elo difference (non-linearity at extremes) |
| `elo_diff_abs` | Absolute Elo difference |

Draws excluded — draw probability is modeled separately in the group stage simulation using a declining exponential formula.

### Model selection

Four algorithms trained on the same time-based split:

| Model | ROC-AUC | Brier Score |
|---|---|---|
| **Logistic Regression** | **0.8545** | **0.1492** |
| Random Forest | 0.8541 | 0.1498 |
| XGBoost | 0.8537 | 0.1495 |
| Gradient Boosting | 0.8535 | 0.1497 |

All within 0.001 AUC. The correct model is a logistic curve — the Elo-difference → win-probability relationship is smooth and monotonic. Tree-based models add no value here.

**Logistic Regression selected:** best metrics, well-calibrated probabilities, interpretable coefficients.

### Validation

- Train: all matches before January 2018
- Test: all matches from January 2018 onward

Time-based split mirrors real production conditions. Cross-validation avoided to prevent future-data leakage.

---

## 5. Feature Enrichment

The base model captures long-run team strength via Elo. Three short-run signals are added to the simulation inputs — not retrained into the model — using only existing match history.

### Feature 1: Recent Form (±75 Elo)

Weighted points-per-game in each team's last 10 international matches (2025 onward):
- Win = 3 pts, Draw = 1 pt, Loss = 0 pts
- Linear recency weighting (most recent = highest weight)
- Normalized to Elo adjustment: `(wPPG − 1.5) × 50`

### Feature 2: World Cup Psychology Proxy (±40 Elo)

For each team: compare WC win rate vs overall win rate across the **last 3 World Cups (2014, 2018, 2022)**.

Using a recent sliding window — rather than all WC history since 1990 — prevents teams with old titles from dominating modern predictions. Brazil's 1994 and 2002 wins previously inflated their WC psychology score, causing the model to rank them #1 in every historical backtest despite not winning since 2002. The window fix resolved this: Argentina was correctly predicted as the 2022 champion.

Formula: `clip((wc_win_rate − overall_win_rate) × 300, −40, +40)`

### Feature 3: Squad Trajectory / Age Proxy (±30 Elo)

The annual rate of Elo improvement over the past 2 years:

```
velocity = (Elo_now − Elo_2yr_ago) / 2
adjustment = clip(velocity × 0.15, −30, +30)
```

- **Positive** = improving team (often young squad coming into form)
- **Negative** = declining team (often aging squad past peak)
- A ±200 Elo/year change maps to the ±30 cap

This is a data-driven proxy for squad age dynamics. A squad that has been consistently improving over two years — like Spain after their Euro 2024 win — gets a meaningful boost, even if their base Elo hasn't fully caught up yet.

*Note: true squad-level age data would require a player roster dataset. This feature captures the same underlying dynamic from match results alone.*

### Enrichment impact for 2026

| Feature | Range across 48 teams |
|---------|----------------------|
| Recent form | −47.7 to +66.8 |
| WC psychology | −40.0 to +40.0 |
| Squad trajectory | −15.8 to +13.8 |

Notable 2026 trajectory movers:
- **Spain +11.1** — Euro 2024 winners, strong upward trajectory
- **Norway +13.8** — consistent improvement in qualifiers
- **Qatar −15.8** — catastrophic 2022 host WC performance
- **Belgium −7.2** — aging golden generation in decline

---

## 6. Feature Development Iterations

The enrichment layer was developed through four iterations, each validated against a historical backtest across the 2010, 2014, 2018, and 2022 World Cups. All Elo, model training, and enrichment used only pre-tournament data (zero look-ahead bias, 10,000 simulations per tournament).

| Version | Change | Top-1 | Top-3 | Avg rank | Avg champion prob |
|---------|--------|-------|-------|----------|------------------|
| **v1** | Baseline: WC psychology uses all history since 1990 | 0/4 | 3/4 | 3.5 | 16.8% |
| **v2** | WC psychology → last 3 tournaments only | 1/4 | 3/4 | 3.5 | 17.5% |
| **v3** | Time-decayed Elo (10%/yr reversion) + v2 | 1/4 | 3/4 | 3.8 | 16.9% |
| **v4** ✓ | v2 + squad trajectory (2-yr Elo velocity) | 1/4 | 3/4 | 3.5 | **18.4%** |

**v1 → v2:** Brazil ranked #1 in all four historical simulations. Root cause: the WC psychology feature used all history since 1990, permanently encoding Brazil's 1994 and 2002 titles. Switching to a 3-tournament window removed the bias. Result: Argentina correctly predicted as 2022 champion (rank #1, 29.9% probability).

**v2 → v3:** Time-decayed Elo applies proportional reversion to all teams. Since all ratings decay together, relative rankings barely change — the Brazil bias lives in the base Elo too, not just the uplift. v3 performed worse than v2. Lesson: equal proportional decay is not the right tool for this problem.

**v3 → v4:** Squad trajectory adds a meaningful forward signal. The 2018 France miss persists (rank #8, 2.9%) — the trajectory feature showed France had near-zero Elo change from 2016 to 2018 (+3 points). Mbappé's emergence and Kanté's form are invisible to match-result metrics. Fixing the 2018 case requires player-level data: a potential v5 extension.

**v4 for 2026:** Spain's Euro 2024 win produces a +11.1 trajectory boost, pushing them neck-and-neck with France. Argentina's 2022 WC probability increases from 29.9% → 32.1%, correctly capturing their unbeaten-run momentum going into Qatar.

---

## 7. Tournament Simulation

### Group stage

- 12 official groups (real FIFA draw, December 5, 2025)
- Round-robin: each team plays the 3 others in their group
- W/D/L probabilities from enriched Elo difference:
  - `P(draw) = 0.25 × exp(−|elo_diff| / 500)`
  - `P(win) = logistic(elo_diff) × (1 − P(draw))`
- Top 2 per group qualify (24 teams); best 8 third-place finishers also qualify (8 teams)

### Knockout stage

R32 → R16 → QF → SF → 3rd Place match + Final. Win probability from the trained logistic regression on four-feature input. No draws — extra time and penalties treated as an Elo-weighted coin flip.

### Simulation

Full tournament simulated **10,000 times** with independent random seeds. Output is the empirical distribution of outcomes. Probabilities stable to ±0.5% for most teams.

---

## 8. Key Results

**Model performance (test set: 2018–present):**

| Metric | Score | Benchmark |
|---|---|---|
| ROC-AUC | **0.855** | 0.5 = random, 1.0 = perfect |
| Brier Score | **0.149** | 0.25 = random, 0.0 = perfect |

**2026 Championship Probabilities (10,000 simulations, v4 enrichment):**

| Team | Group | Adj. Elo | Champion | Runner-up | 3rd Place | Reach Final | Reach SF |
|---|---|---|---|---|---|---|---|
| France | I | 2222 | **25.7%** | 10.4% | 12.1% | 36.1% | 53.7% |
| Spain | H | 2227 | **23.8%** | 10.6% | 14.1% | 34.4% | 53.7% |
| Argentina | J | 2196 | **17.6%** | 15.3% | 9.6% | 32.9% | 48.0% |
| Colombia | K | 2103 | 5.0% | 7.6% | 6.5% | 12.6% | 24.1% |
| Germany | E | 2075 | 4.4% | 5.8% | 5.9% | 10.2% | 21.5% |
| Netherlands | F | 2085 | 4.3% | 5.4% | 5.9% | 9.8% | 20.4% |
| England | L | 2065 | 3.5% | 6.8% | 5.7% | 10.3% | 22.2% |
| Brazil | C | 2039 | 2.2% | 4.8% | 3.9% | 7.0% | 15.9% |

**Historical backtest (v4):**

| Year | Actual Champion | Model Rank | Prob. Assigned |
|---|---|---|---|
| 2010 South Africa | Spain | #2 | 24.9% |
| 2014 Brazil | Germany | #3 | 13.6% |
| 2018 Russia | France | #8 | 2.9% |
| 2022 Qatar | **Argentina** | **#1 ✓** | **32.1%** |

Top-3 accuracy: **75% (3/4)**. Average champion probability assigned: **18.4%**.

---

## 9. Deployment Analysis

**Verdict: Production-ready with defined limitations.**

| Criterion | Assessment |
|---|---|
| Predictive accuracy | ROC-AUC 0.855 — strong for football, a noisy low-scoring sport |
| Probability calibration | Logistic regression well-calibrated by design |
| Inference speed | Milliseconds — no GPU required |
| Data dependency | One public CSV, updated automatically after each match |
| Interpretability | One coefficient per feature — fully auditable |
| Staleness risk | Re-run after each international window |

**Deployment recommendations:**
1. **API endpoint** — FastAPI endpoint returning win/draw/loss probabilities with Elo snapshot timestamp
2. **Automated Elo refresh** — post-match pipeline job after each international window
3. **Prediction monitoring** — log predicted vs actual; alert if rolling Brier Score exceeds 0.20
4. **Annual retrain** — refit logistic regression on full updated dataset each year
5. **Player-level module (v5)** — squad rosters + player ratings would address the 2018-type miss where squad transformation is invisible to team-level Elo

**Known limitations:**
- Injuries and suspensions not captured
- Squad transformation (young talent peaking) only visible after results accumulate
- WC psychology feature has limited statistical power for nations with few WC appearances
- Knockout bracket third-place seeding approximated (FIFA uses a 495-entry lookup table)

---

## 10. Web Application

An interactive Streamlit app with five pages, deployable on Streamlit Community Cloud.

### Pages

| Page | Function |
|---|---|
| **Overview** | Pre-tournament predictions: championship table, group heatmaps, tournament progression for all 48 teams |
| **Live Tracker** | Enter real match results; app updates Elo ratings and re-runs 10,000 simulations live |
| **Match Predictor** | Win/draw/loss probabilities for any head-to-head matchup |
| **Bracket Builder** | Build a custom Round of 32; simulate or pick winners manually round by round |
| **Knockout Explorer** | Who each team is most likely to face at each stage, plus full opponent probability heatmap |

### Architecture

```
app/
  streamlit_app.py          Main page (Overview)
  pages/
    1_Live_Tracker.py
    2_Match_Predictor.py
    3_Bracket_Builder.py
    4_Knockout_Explorer.py
  model/
    data.py                 Data loading, group definitions, WC start dates
    elo.py                  Elo engine + all enrichment (form, WC proxy, trajectory)
    predict.py              Model training and inference
    simulate.py             Monte Carlo tournament simulation
    core.py                 Streamlit session state initialisation
  requirements.txt
```

### Running locally

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Tech stack

| Layer | Technology |
|---|---|
| Frontend + Backend | Streamlit |
| ML model | scikit-learn LogisticRegression |
| Data | pandas, numpy |
| Visualizations | matplotlib |
| Hosting | Streamlit Community Cloud (free) |

---

## 11. Potential Applications

| Domain | Use Case |
|---|---|
| Sports media | Automated tournament previews and live probability updates after each match |
| Sports betting | Identify mispriced lines where model probability diverges from bookmaker implied odds by >5% |
| Fantasy sports | Inform player selection based on team advancement probabilities |
| National federations | Benchmark competitive position; quantify Elo gap to the next tier |
| Brand & sponsorship | Quantify expected tournament reach of a potential sponsor team before signing |
| Research | Reproducible Elo + logistic regression baseline for testing more complex models |
