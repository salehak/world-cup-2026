# World Cup 2026 Match Predictions

I built a machine learning pipeline that predicts every match of the 2026 FIFA World Cup (72 group matches and 32 knockout matches). It was my entry to DataCamp's [Predict the FIFA World Cup 2026](https://app.datacamp.com/learn/competitions/world-cup-prediction) competition. The pipeline combines recent international results, per-match statistics, Elo ratings, FIFA rankings and head-to-head records. XGBoost models are trained on these features and then used to simulate the tournament bracket.

## Objective

All predictions had to be submitted before the tournament started. For every match the competition asked for:

- the exact score (after extra time in knockout matches);
- the number of corners;
- the number of yellow cards and red cards;
- **group stage:** the winner (home / draw / away);
- **knockout stage:** which two teams meet in each slot, the winner, and whether the match goes to penalties.

Points were awarded per prediction. Later rounds carried multipliers (×2 in the Round of 16, rising to ×16 for the final), so getting the bracket right mattered as much as individual scorelines.

## Data

The datasets are **not included in this repository**. The table below explains how to obtain each one. All files go in `data/`.

| File | Source | How it was collected | Contents |
|---|---|---|---|
| `results_clean.csv`, `shootouts_clean.csv` | [martj42/international_results](https://github.com/martj42/international_results) (also on [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)) | Downloaded by `01_data_collection.ipynb` | Results and penalty shootouts from 2024-01-01 to 2026-06-04 |
| `matches_stats.csv` | `results_clean.csv` + [FotMob](https://www.fotmob.com) | FotMob match statistics added manually | Each match plus possession, xG, shots, shots on target, corners, yellow cards and red cards for both teams (2,458 matches) |
| `team_records.csv` | [FBref](https://fbref.com) | Head-to-head records | Each qualified team's record against every opponent: matches, W/D/L, goals, goal difference, points, PPG, win ratio (2,359 rows) |
| `elo_ratings.csv` | [World Football Elo Ratings](https://www.eloratings.net) | Transcribed from the site with Claude (the site's `.tsv` export has no column headers) | Elo rank, rating, all-time matches, wins/draws/losses (244 teams) |
| `fifa_rankings.csv` | [FIFA/Coca-Cola Men's World Ranking](https://inside.fifa.com/fifa-world-ranking/men) | Transcribed with Gemini (FIFA has no public API) | Rank, team, confederation, points (211 teams) |
| `group_fixtures.csv`, `knockout_slots.csv` | DataCamp competition page | Provided to participants | 72 group fixtures; 32 knockout slots with round multipliers |

The ratings and head-to-head tables are single snapshots taken in June 2026, before the tournament.

**Expected columns.** `matches_stats.csv` contains every column of `results_clean.csv` plus:
`home_team_possession, away_team_possession, home_team_xG, away_team_xG, home_team_shots, away_team_shots, home_SoT, away_SoT, home_corners, away_corners, home_YC, away_YC, home_RC, away_RC`.
Leave a value blank when FotMob has no data for that match.

The other files use these headers:

- `team_records.csv`: `Team, Country, Matches, W, D, L, Goals (e.g. 21:03), +/-, Points, PPG, Win ratio (e.g. 84.60%), Average attendance`
- `elo_ratings.csv`: `Rank, Team, Elo Rating, …, Total Matches, …, Wins, …`
- `fifa_rankings.csv`: `rank, team, confederation, points`

## Approach

**Data preparation** (`02_modeling.ipynb`, sections 1–4)

- Country names are standardised across all sources (e.g. *Korea Republic* → *South Korea*).
- Match statistics are often missing. Before imputation, corners and cards are observed in about 41% of matches and xG in about 17%. Missing values are filled with team medians, falling back to the global median.
- Ratings and head-to-head records are joined onto each match. Matches involving a team without Elo or FIFA data are dropped, which leaves 2,368 matches for modelling.

**Feature engineering** (section 5)

- **Strength gaps:** Elo rating gap, FIFA rank gap, each team's FIFA points, and a hand-assigned confederation strength gap.
- **Head-to-head:** win, draw and loss rates; a dominance score; and a flag for pairs with at least 4 previous meetings.
- **Rolling form:** mean of the previous 5 matches on the same side (home or away), shifted so the current match is never included. Goals are capped at 3 before averaging.
- **Context:** neutral venue; competitive vs. friendly.

**Feature selection** (section 6)

- Collinearity check: the raw Elo and FIFA rank/rating columns correlate at |r| ≥ 0.96, so they are reduced to gap features.
- xG is dropped because most of its values are imputed.
- Permutation importance from a proxy random forest. The final feature set was chosen by hand; the notebook lists where it departs from the importance scores.

**Models** (sections 8–10): eight XGBoost models.

| Target | Model |
|---|---|
| Result (home / draw / away) | `XGBClassifier` |
| Home goals, away goals | `XGBRegressor` × 2 |
| Total corners | `XGBRegressor` |
| Home and away yellow cards | `XGBRegressor` × 2 |
| Home and away red card (yes/no) | `XGBClassifier` × 2 |

- **Tuning:** the result and goals models are tuned with a grid search using 5-fold `TimeSeriesSplit` cross-validation on the training data. The corners and cards models reuse the result model's parameters.
- **Validation:** matches are split chronologically, 80% for training (1,894 matches, 2024-01 to 2025-10) and 20% for validation (474 matches, 2025-10 to 2026-06).
- **Final fit:** after validation, all models are refit on every match.

**Simulation** (sections 11–13)

- **Winner:** the result classifier picks each winner. In knockout matches the probabilities are first adjusted by the FIFA points gap.
- **Scoreline:** sampled from Poisson distributions whose means come from the goals models, and resampled until it agrees with the chosen winner.
- **Penalties:** a knockout match is predicted to go to penalties when the home and away probabilities are within 0.12 of each other.
- **Bracket:** groups are ranked by points, goal difference and goals scored. Group positions fill the knockout slots, and winners advance round by round.
- **Reproducibility:** NumPy is seeded, so a re-run produces identical predictions.

## Results

Validation metrics come from models trained on the training split only:

| Metric | All validation matches (474) | Competitive only (110) |
|---|---|---|
| Result accuracy, XGBoost | **0.719** | **0.700** |
| Baseline: always predict home win | 0.492 | 0.491 |
| Baseline: team with higher Elo wins | 0.637 | 0.700 |
| Draw recall | 0.39 | 0.10 |
| Home / away goals MAE | 0.94 / 0.85 | 1.07 / 0.86 |
| Total corners MAE (matches with observed corners) | 2.79 (n = 228) | 3.33 (n = 53) |
| Total yellow cards MAE (matches with observed cards) | 1.74 (n = 207) | 1.50 (n = 53) |

The best cross-validated result accuracy on the training split was 0.636.

**Findings**

- **The Elo gap carries most of the signal.** Its permutation importance (0.073) is about seven times that of any other feature.
- **The model beats the Elo baseline only on the full validation set.** On all validation matches it improves on "higher Elo wins" by about 8 points. On competitive matches it only matches that baseline (0.700 vs 0.700).
- **Draws are the weak spot.** Draw recall is 0.39 overall and 0.10 in competitive matches, so most draws are predicted as wins.
- **Corners and cards are hard to predict.** The errors above are measured only on matches where those statistics were actually recorded.

**Tournament predictions.** Both the submitted entry (`notebooks/archive/datacamp_submission.ipynb`) and the cleaned, seeded pipeline predict **Spain beating Argentina 2–1 in the final**. The other bracket picks differ between the two runs: the submitted notebook sampled scorelines without a fixed seed, and the cleaned pipeline evaluates and seeds differently. The predictions have not been scored against the actual tournament results in this repository.

## Repository structure

```
├── notebooks/
│   ├── 01_data_collection.ipynb    # download and prepare international results (martj42)
│   ├── 02_modeling.ipynb           # features, models, validation, tournament simulation
│   └── archive/
│       └── datacamp_submission.ipynb  # notebook as submitted to DataCamp (kept for reference, not maintained)
├── data/        # input data, not tracked (see Data)
├── outputs/     # generated predictions, not tracked
├── requirements.txt
└── README.md
```

## How to reproduce

```bash
git clone https://github.com/salehak/world-cup-2026.git
cd world-cup-2026
pip install -r requirements.txt jupyterlab
```

1. Run `notebooks/01_data_collection.ipynb` to create `data/results_clean.csv` and `data/shootouts_clean.csv`.
2. Build `data/matches_stats.csv` by adding the FotMob statistics to `results_clean.csv` (see [Data](#data)).
3. Add `team_records.csv`, `elo_ratings.csv`, `fifa_rankings.csv`, `group_fixtures.csv` and `knockout_slots.csv` to `data/`.
4. Run `notebooks/02_modeling.ipynb` to write `outputs/group_predictions.csv` and `outputs/knockout_predictions.csv`.

The notebooks work from either the repo root or `notebooks/`.

Re-running step 1 today returns slightly different rows, because the upstream martj42 dataset is still being revised (for example, *China PR* was renamed to *China*). Results also depend on the package versions pinned in `requirements.txt`.

## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, requests, Jupyter.

## Limitations

- **Point-in-time leakage.** The Elo, FIFA and Elo win-rate features are June 2026 snapshots attached to matches from 2024–2026. Head-to-head totals are not dated. Historical matches are therefore described with information from after they were played, so the validation scores are likely optimistic.
- **Imputation.** About 59% of corner and card values, and about 83% of xG values, are imputed. The team medians are computed over all matches, including the validation period, and the corners and cards models are trained partly on imputed targets.
- **Manual and LLM-assisted data.** The Elo and FIFA tables were transcribed with LLM chatbots, and the FotMob statistics were collected by hand.
- **Small evaluation set.** The competitive validation set has only 110 matches.
- **Heuristics:**
  - confederation strengths are hand-assigned;
  - the FIFA points adjustment and the 0.12 penalty threshold were not tuned;
  - goal means are clipped to 0.1–2.5 and winning margins capped at 3;
  - the best third-placed teams are assigned to slots greedily rather than with FIFA's official allocation table.
- **Team profiles.** Each team is represented by its features from its most recent match only.
- **Single simulation.** The bracket comes from one simulation run, not from probabilities over many runs.

## Future improvements

- Use point-in-time features: Elo ratings as of each match date, and head-to-head records built from earlier matches only.
- Fit imputation statistics on the training period only.
- Score the predictions against the actual 2026 results using the competition's points system.
- Run Monte Carlo simulations of the bracket to estimate each team's probability of advancing.
- Implement FIFA's official third-place allocation table.
- Take specific players & conditions into count.
- Things that differed in 2026: teams were often flying for their next match & there were more matches than usual due to having more teams in the tournament. These are some factors that could not be included.
