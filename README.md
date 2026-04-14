# ⚽ FIFA World Cup 2026 Prediction Engine

An ML pipeline that predicts the winner of the 2026 FIFA World Cup using real FIFA ranking data, XGBoost classification, and Monte Carlo tournament simulation.

![Predictions](https://github.com/user-attachments/assets/6bdd6192-560e-46bc-a8cb-1cf673caf070|https://github.com/user-attachments/assets/6bdd6192-560e-46bc-a8cb-1cf673caf070)

## Results

After 2,000 simulated tournaments:

| Rank | Team | Win % |
|------|------|-------|
| 🥇 | Brazil | 18.3% |
| 🥈 | Belgium | 10.0% |
| 🥉 | Netherlands | 9.3% |
| 4 | France | 8.8% |
| 5 | Spain | 8.4% |
| 6 | Argentina | 6.9% |
| 7 | USA | 4.1% |
| 8 | Portugal | 3.6% |
| 9 | England | 2.6% |
| 10 | Mexico | 2.5% |

## How It Works

The pipeline has 4 stages:

**1. Team Strength Ratings**
FIFA ranking points (211 teams) are used as Elo-equivalent strength ratings. These feed directly into the match prediction model.

**2. Calibrated Match Generation**
20,000 synthetic match results are generated using a Poisson goal model calibrated to FIFA points. The strength differential between two teams determines each side's expected goals (lambda). World Cup matches apply a 0.82x tightness multiplier, reflecting the more defensive nature of tournament football.

```
home_lambda = max(0.4, 1.25 + (strength_diff / 700) + 0.3)
away_lambda = max(0.4, 1.25 - (strength_diff / 700))
```

**3. XGBoost Training**
An XGBoost classifier is trained on 5 features to predict match outcomes (home win / draw / away win):

| Feature | Importance |
|---------|-----------|
| `elo_diff` | 32.1% |
| `elo_away` | 20.0% |
| `elo_home` | 19.4% |
| `neutral` | 15.1% |
| `is_wc` | 13.5% |

5-fold cross-validation accuracy: **50.2%** (random baseline: 33%, theoretical ceiling for football: ~55-58%)

**4. Monte Carlo Tournament Simulation**
The trained model predicts win/draw/loss probabilities for every possible matchup. The full 48-team tournament bracket is then simulated thousands of times: group stage (round-robin), best third-place advancement, Round of 32, Round of 16, quarter-finals, semi-finals, and final. Host nations (USA, Mexico, Canada) receive a 12% win probability boost.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/wc2026-predictor.git
cd wc2026-predictor
pip install -r requirements.txt
python predict.py
```

## Configuration

```bash
# Run with more simulations for stable results
python predict.py --sims 10000

# Use a different ranking file
python predict.py --data data/my_rankings.csv

# Change random seed
python predict.py --seed 123

# Generate more training matches
python predict.py --matches 50000
```

## Project Structure

```
wc2026-predictor/
├── predict.py                        # Main pipeline
├── requirements.txt                  # Python dependencies
├── data/
│   └── fifa_ranking_2022-10-06.csv   # FIFA ranking dataset (211 teams)
└── output/
    ├── predictions.csv               # Full results table
    └── predictions.png               # Visualization
```

## Limitations & Future Work

This model uses a single FIFA ranking snapshot as the strength foundation. Some ways to improve it:

- **Recency weighting**: Use the latest FIFA rankings (March 2026) or apply time decay so recent results matter more
- **Player-level features**: Incorporate squad age, key player availability, and FIFA video game ratings as squad strength proxies
- **Historical match data**: Train on the real [international results dataset](https://github.com/martj42/international_results) (49K+ matches since 1872) instead of generated data
- **Head-to-head records**: Some matchups have historical patterns (e.g., Germany vs Italy) that pure strength ratings miss
- **Tactical matchup modeling**: Certain playing styles counter others, which a more sophisticated model could capture

## Dataset

FIFA ranking data from October 6, 2022 (pre-Qatar World Cup). Contains 211 teams with columns:

| Column | Description |
|--------|-------------|
| `team` | Country name |
| `team_code` | FIFA 3-letter code |
| `association` | Confederation (UEFA, CONMEBOL, etc.) |
| `rank` | Current FIFA rank |
| `points` | FIFA ranking points (used as strength rating) |

## Tech Stack

Python, XGBoost, scikit-learn, pandas, NumPy, Matplotlib

## License

MIT
