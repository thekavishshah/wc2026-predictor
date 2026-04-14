"""
FIFA World Cup 2026 Prediction Engine
======================================
ML pipeline using real FIFA Rankings, XGBoost, and Monte Carlo simulation
to predict the winner of the 2026 FIFA World Cup.

Pipeline:
    1. Load FIFA ranking points as team strength ratings
    2. Generate calibrated match history using a Poisson goal model
    3. Train XGBoost + Logistic Regression classifiers (5-fold CV)
    4. Predict group stage match outcomes
    5. Monte Carlo simulate 2,000+ full tournament brackets
    6. Output win probabilities for all 48 teams

Usage:
    python predict.py
    python predict.py --sims 5000 --data data/fifa_ranking_2022-10-06.csv
"""

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# WORLD CUP 2026 CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Groups (confirmed draw, Dec 5 2025, Kennedy Center, Washington D.C.)
WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

DISPLAY_NAMES = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Curaçao": "Curacao",
    "Türkiye": "Turkiye",
}

HOST_NATIONS = ["USA", "Mexico", "Canada"]
HOST_BOOST = 1.12  # 12% win probability boost for host nations

ALL_WC_TEAMS = []
TEAM_TO_GROUP = {}
for grp, teams in WC2026_GROUPS.items():
    for t in teams:
        ALL_WC_TEAMS.append(t)
        TEAM_TO_GROUP[t] = grp


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_rankings(path: str) -> dict:
    """
    Load FIFA ranking CSV and return a {team_name: points} dictionary.

    Expected columns: team, team_code, association, rank, previous_rank,
                      points, previous_points
    """
    df = pd.read_csv(path)
    print(f"[DATA] Loaded FIFA Rankings: {len(df)} teams")

    strength = {}
    for _, row in df.iterrows():
        strength[row["team"]] = row["points"]

    # Fill in any WC2026 teams missing from the ranking file
    missing = [t for t in ALL_WC_TEAMS if t not in strength]
    if missing:
        print(f"[WARN] Missing teams (assigning default 1300): {missing}")
        for t in missing:
            strength[t] = 1300.0

    # Display WC2026 teams sorted by strength
    wc_ratings = sorted(
        [(t, strength[t]) for t in ALL_WC_TEAMS],
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n{'='*60}")
    print("WORLD CUP 2026 TEAMS BY FIFA POINTS")
    print(f"{'='*60}")
    for i, (team, pts) in enumerate(wc_ratings):
        display = DISPLAY_NAMES.get(team, team)
        host = " *" if team in HOST_NATIONS else ""
        print(f"  {i+1:2d}. {display:25s} {pts:>8.1f}{host}")
    print(f"\n  * = host nation")

    return strength


# ══════════════════════════════════════════════════════════════
# 2. MATCH HISTORY GENERATION (Poisson Goal Model)
# ══════════════════════════════════════════════════════════════

def generate_match_history(strength: dict, n_matches: int = 20_000,
                           seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic match results calibrated to FIFA ranking points.

    Uses a Poisson goal model where each team's expected goals (lambda)
    is derived from the strength differential:
        home_lambda = max(0.4, 1.25 + (diff / 700) + home_advantage)
        away_lambda = max(0.4, 1.25 - (diff / 700))

    World Cup matches apply a 0.82x multiplier to both lambdas,
    reflecting tighter, more defensive play in tournament settings.
    """
    np.random.seed(seed)
    print(f"\n[DATA] Generating {n_matches:,} calibrated match results...")

    # Use top 120 teams by ranking to avoid noise from very weak teams
    teams = sorted(strength.keys(), key=lambda t: strength[t], reverse=True)[:120]

    # Weighted tournament distribution (approximates real fixture mix)
    tournaments = (
        ["Friendly"] * 35 +
        ["FIFA World Cup qualification"] * 25 +
        ["FIFA World Cup"] * 8 +
        ["Continental championship"] * 15 +
        ["Nations League"] * 12 +
        ["Confederations Cup"] * 5
    )

    records = []
    for _ in range(n_matches):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        tournament = np.random.choice(tournaments)

        diff = strength[home] - strength[away]
        home_advantage = 0.3
        home_lambda = max(0.4, 1.25 + (diff / 700) + home_advantage)
        away_lambda = max(0.4, 1.25 - (diff / 700))

        if "World Cup" in tournament and "qualification" not in tournament:
            home_lambda *= 0.82
            away_lambda *= 0.82

        home_score = np.random.poisson(home_lambda)
        away_score = np.random.poisson(away_lambda)
        neutral = 1 if (tournament == "FIFA World Cup" and np.random.random() > 0.3) else 0
        is_wc = 1 if "World Cup" in tournament else 0

        records.append({
            "home_team": home, "away_team": away,
            "home_score": home_score, "away_score": away_score,
            "tournament": tournament, "neutral": neutral, "is_wc": is_wc,
            "elo_home": strength[home], "elo_away": strength[away],
            "elo_diff": strength[home] - strength[away],
        })

    df = pd.DataFrame(records)
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 2,    # home win
        np.where(df["home_score"] < df["away_score"], 0, 1)  # away win / draw
    )

    hw = (df["result"] == 2).sum()
    dr = (df["result"] == 1).sum()
    aw = (df["result"] == 0).sum()
    print(f"[DATA] Results: {hw:,} home wins ({hw/len(df):.1%}), "
          f"{dr:,} draws ({dr/len(df):.1%}), "
          f"{aw:,} away wins ({aw/len(df):.1%})")

    return df


# ══════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════

def train_models(df: pd.DataFrame, seed: int = 42):
    """
    Train XGBoost and Logistic Regression on match data.

    Features:
        elo_diff  - strength gap between home and away team
        elo_home  - home team's FIFA points
        elo_away  - away team's FIFA points
        neutral   - 1 if neutral venue, 0 if home
        is_wc     - 1 if World Cup match, 0 otherwise

    Target: 0 = away win, 1 = draw, 2 = home win
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    feature_cols = ["elo_diff", "elo_home", "elo_away", "neutral", "is_wc"]
    X = df[feature_cols]
    y = df["result"]

    print(f"\n{'='*60}")
    print("MODEL TRAINING")
    print(f"{'='*60}")
    print(f"  Features : {feature_cols}")
    print(f"  Samples  : {len(X):,}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # XGBoost
    print("\n  [XGBoost]")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=seed, verbosity=0,
    )
    xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")
    xgb.fit(X, y)

    importances = xgb.feature_importances_
    for i in np.argsort(importances)[::-1]:
        print(f"    {feature_cols[i]:20s} {importances[i]:.4f}")

    # Logistic Regression (baseline)
    print("\n  [Logistic Regression]")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=seed))
    ])
    lr_scores = cross_val_score(lr_pipe, X, y, cv=cv, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")
    lr_pipe.fit(X, y)

    return xgb, lr_pipe


# ══════════════════════════════════════════════════════════════
# 4. MATCH PREDICTION
# ══════════════════════════════════════════════════════════════

_pred_cache = {}

def predict_match(model, strength: dict, team_a: str, team_b: str,
                  neutral: int = 1, is_wc: int = 1) -> tuple:
    """
    Predict P(team_a wins), P(draw), P(team_b wins).
    Results are cached to avoid redundant model inference during simulation.
    """
    key = (team_a, team_b, neutral, is_wc)
    if key in _pred_cache:
        return _pred_cache[key]

    elo_a = strength.get(team_a, 1400)
    elo_b = strength.get(team_b, 1400)

    feat = pd.DataFrame([{
        "elo_diff": elo_a - elo_b,
        "elo_home": elo_a,
        "elo_away": elo_b,
        "neutral": neutral,
        "is_wc": is_wc,
    }])

    probs = model.predict_proba(feat)[0]
    result = (probs[2], probs[1], probs[0])  # (a_win, draw, b_win)
    _pred_cache[key] = result
    return result


# ══════════════════════════════════════════════════════════════
# 5. TOURNAMENT SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_group(model, strength, group_teams, rng):
    """Simulate round-robin group stage for a single group."""
    pts = {t: 0 for t in group_teams}
    gd  = {t: 0 for t in group_teams}
    gf  = {t: 0 for t in group_teams}

    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            ta, tb = group_teams[i], group_teams[j]
            pa, pd_val, pb = predict_match(model, strength, ta, tb)

            # Apply host nation boost
            if ta in HOST_NATIONS:
                pa *= HOST_BOOST; pb *= 0.92
            elif tb in HOST_NATIONS:
                pb *= HOST_BOOST; pa *= 0.92

            total = pa + pd_val + pb
            pa, pd_val, pb = pa / total, pd_val / total, pb / total

            roll = rng.random()
            if roll < pa:
                pts[ta] += 3
                g = rng.choice([1, 1, 1, 2, 2, 3])
                gd[ta] += g; gd[tb] -= g; gf[ta] += g
            elif roll < pa + pd_val:
                pts[ta] += 1; pts[tb] += 1
                g = rng.choice([0, 1, 1, 2])
                gf[ta] += g; gf[tb] += g
            else:
                pts[tb] += 3
                g = rng.choice([1, 1, 1, 2, 2, 3])
                gd[tb] += g; gd[ta] -= g; gf[tb] += g

    ranked = sorted(group_teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return ranked, pts[ranked[2]], gd[ranked[2]]


def simulate_knockout(model, strength, ta, tb, rng):
    """Simulate a single knockout match (no draws, winner advances)."""
    pa, _, pb = predict_match(model, strength, ta, tb)
    if ta in HOST_NATIONS: pa *= HOST_BOOST
    elif tb in HOST_NATIONS: pb *= HOST_BOOST
    total = pa + pb
    return ta if rng.random() < (pa / total) else tb


def _advance_round(model, strength, teams, rng):
    """Helper: pair teams and advance winners."""
    winners = []
    for i in range(0, len(teams), 2):
        if i + 1 < len(teams):
            winners.append(simulate_knockout(model, strength, teams[i], teams[i + 1], rng))
        else:
            winners.append(teams[i])
    return winners


def simulate_tournament(model, strength, rng):
    """
    Simulate a full World Cup 2026 tournament.

    Format: 12 groups of 4 -> top 2 + best 8 third-place teams advance
            -> Round of 32 -> Round of 16 -> QF -> SF -> Final
    """
    group_results = {}
    third_place = []

    # Group stage
    for grp, teams in WC2026_GROUPS.items():
        ranked, third_pts, third_gd = simulate_group(model, strength, teams, rng)
        group_results[grp] = ranked
        third_place.append((ranked[2], third_pts, third_gd, grp))

    # Best 8 third-place teams
    third_place.sort(key=lambda x: (x[1], x[2]), reverse=True)
    advancing_thirds = [t[0] for t in third_place[:8]]

    # Build bracket
    winners = [group_results[g][0] for g in sorted(WC2026_GROUPS.keys())]
    runners = [group_results[g][1] for g in sorted(WC2026_GROUPS.keys())]

    # Round of 32
    r32 = []
    for i in range(min(len(winners), len(advancing_thirds))):
        r32.append(simulate_knockout(model, strength, winners[i], advancing_thirds[7 - i], rng))
    for i in range(0, len(runners), 2):
        if i + 1 < len(runners):
            r32.append(simulate_knockout(model, strength, runners[i], runners[i + 1], rng))

    # Knockout rounds
    r16 = _advance_round(model, strength, r32, rng)
    qf  = _advance_round(model, strength, r16, rng)
    sf  = _advance_round(model, strength, qf, rng)
    finalists = _advance_round(model, strength, sf, rng)

    champion = (
        simulate_knockout(model, strength, finalists[0], finalists[1], rng)
        if len(finalists) >= 2 else finalists[0]
    )

    return {"champion": champion, "finalists": finalists, "sf": sf, "qf": qf}


def run_monte_carlo(model, strength, n_sims: int = 2_000,
                    seed: int = 42) -> pd.DataFrame:
    """
    Run N full tournament simulations and aggregate results.
    Returns a DataFrame with win/final/semi/QF probabilities per team.
    """
    print(f"\n{'='*60}")
    print(f"MONTE CARLO SIMULATION ({n_sims:,} tournaments)")
    print(f"{'='*60}")

    counts = {t: {"champ": 0, "final": 0, "semi": 0, "qf": 0}
              for t in ALL_WC_TEAMS}

    for i in range(n_sims):
        if (i + 1) % 500 == 0:
            print(f"  Simulated {i + 1:,}/{n_sims:,}...")

        rng = np.random.RandomState(seed + i)
        res = simulate_tournament(model, strength, rng)

        counts[res["champion"]]["champ"] += 1
        for t in res["finalists"]:
            counts[t]["final"] += 1
        for t in res["sf"]:
            counts[t]["semi"] += 1
        for t in res["qf"]:
            counts[t]["qf"] += 1

    rows = []
    for team in ALL_WC_TEAMS:
        display = DISPLAY_NAMES.get(team, team)
        c = counts[team]
        rows.append({
            "Team": display,
            "Group": TEAM_TO_GROUP[team],
            "FIFA Pts": round(strength.get(team, 1300), 1),
            "Win %": round(c["champ"] / n_sims * 100, 2),
            "Final %": round(c["final"] / n_sims * 100, 2),
            "Semi %": round(c["semi"] / n_sims * 100, 2),
            "QF %": round(c["qf"] / n_sims * 100, 2),
        })

    results_df = (pd.DataFrame(rows)
                  .sort_values("Win %", ascending=False)
                  .reset_index(drop=True))
    results_df.index += 1
    return results_df


# ══════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ══════════════════════════════════════════════════════════════

def plot_results(results_df: pd.DataFrame, output_path: str = "output/predictions.png"):
    """Generate a dark-themed prediction chart."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    fig, axes = plt.subplots(1, 2, figsize=(20, 11),
                             gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor("#0a0e17")

    # Left: Top 20 win probabilities
    ax = axes[0]
    ax.set_facecolor("#0a0e17")
    top20 = results_df.head(20).iloc[::-1]

    colors = []
    for _, row in top20.iterrows():
        if row["Win %"] >= 10:   colors.append("#f59e0b")
        elif row["Win %"] >= 5:  colors.append("#3b82f6")
        elif row["Win %"] >= 2:  colors.append("#10b981")
        else:                    colors.append("#64748b")

    ax.barh(range(len(top20)), top20["Win %"], color=colors, height=0.7)
    for i, (_, row) in enumerate(top20.iterrows()):
        ax.text(row["Win %"] + 0.2, i, f'{row["Win %"]:.1f}%',
                va="center", fontsize=10, color="#e2e8f0", fontweight="bold")

    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(
        [f'{row["Team"]} (Grp {row["Group"]})' for _, row in top20.iterrows()],
        fontsize=11, color="#e2e8f0",
    )
    ax.set_xlabel("Win Probability (%)", fontsize=12, color="#94a3b8")
    ax.set_title("FIFA World Cup 2026 Predictions\nXGBoost + Monte Carlo Simulation",
                 fontsize=16, fontweight="bold", color="#e2e8f0", pad=20)
    ax.tick_params(colors="#94a3b8")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color("#1e293b")
    ax.set_xlim(0, top20["Win %"].max() * 1.3)

    # Right: stage probabilities for top 10
    ax2 = axes[1]
    ax2.set_facecolor("#0a0e17")
    top10 = results_df.head(10)
    x = np.arange(len(top10))
    w = 0.2

    ax2.bar(x - 1.5*w, top10["QF %"],   w, label="Quarter-Final", color="#64748b")
    ax2.bar(x - 0.5*w, top10["Semi %"], w, label="Semi-Final",    color="#10b981")
    ax2.bar(x + 0.5*w, top10["Final %"],w, label="Final",         color="#3b82f6")
    ax2.bar(x + 1.5*w, top10["Win %"],  w, label="Champion",      color="#f59e0b")

    ax2.set_xticks(x)
    ax2.set_xticklabels(top10["Team"], rotation=45, ha="right",
                        fontsize=10, color="#e2e8f0")
    ax2.set_ylabel("Probability (%)", fontsize=12, color="#94a3b8")
    ax2.set_title("Tournament Stage Probabilities",
                  fontsize=14, fontweight="bold", color="#e2e8f0", pad=20)
    ax2.legend(loc="upper right", fontsize=9, facecolor="#111827",
               edgecolor="#1e293b", labelcolor="#e2e8f0")
    ax2.tick_params(colors="#94a3b8")
    for s in ["top", "right"]: ax2.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax2.spines[s].set_color("#1e293b")

    plt.tight_layout(pad=3)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0a0e17")
    print(f"[VIZ] Saved: {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FIFA World Cup 2026 ML Predictor")
    parser.add_argument("--data", type=str, default="data/fifa_ranking_2022-10-06.csv",
                        help="Path to FIFA ranking CSV")
    parser.add_argument("--sims", type=int, default=2000,
                        help="Number of Monte Carlo tournament simulations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--matches", type=int, default=20000,
                        help="Number of calibrated matches to generate for training")
    args = parser.parse_args()

    print("=" * 60)
    print("  FIFA WORLD CUP 2026 — ML PREDICTION ENGINE")
    print("  FIFA Rankings + XGBoost + Monte Carlo")
    print("=" * 60)

    # 1. Load rankings
    strength = load_rankings(args.data)

    # 2. Generate training data
    match_df = generate_match_history(strength, n_matches=args.matches, seed=args.seed)

    # 3. Train models
    xgb_model, lr_model = train_models(match_df, seed=args.seed)

    # 4. Group stage predictions
    print(f"\n{'='*60}")
    print("KEY GROUP STAGE PREDICTIONS")
    print(f"{'='*60}")
    matchups = [
        ("Brazil", "Morocco", "C"),
        ("Argentina", "Algeria", "J"),
        ("France", "Senegal", "I"),
        ("England", "Croatia", "L"),
        ("Spain", "Uruguay", "H"),
        ("USA", "Türkiye", "D"),
        ("Germany", "Ecuador", "E"),
        ("Netherlands", "Japan", "F"),
        ("Portugal", "Colombia", "K"),
        ("Belgium", "Egypt", "G"),
    ]
    for ta, tb, grp in matchups:
        pa, pd_val, pb = predict_match(xgb_model, strength, ta, tb)
        da = DISPLAY_NAMES.get(ta, ta)
        db = DISPLAY_NAMES.get(tb, tb)
        print(f"  Grp {grp}: {da:20s} vs {db:20s}  |  "
              f"{da[:3].upper()}: {pa:.1%}  Draw: {pd_val:.1%}  "
              f"{db[:3].upper()}: {pb:.1%}")

    # 5. Monte Carlo simulation
    results_df = run_monte_carlo(xgb_model, strength, n_sims=args.sims, seed=args.seed)

    # 6. Results
    print(f"\n{'='*60}")
    print("FINAL PREDICTIONS")
    print(f"{'='*60}")
    print(results_df.head(25).to_string())

    # 7. Visualize + save
    plot_results(results_df, output_path="output/predictions.png")
    results_df.to_csv("output/predictions.csv", index=True)
    print(f"[OUTPUT] Saved: output/predictions.csv")

    # Top 3
    top3 = results_df.head(3)
    print(f"\n{'='*60}")
    for i, medal in enumerate(["🥇", "🥈", "🥉"]):
        row = top3.iloc[i]
        print(f"  {medal} {row['Team']:20s} {row['Win %']:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
