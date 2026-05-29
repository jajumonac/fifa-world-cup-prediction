import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


FEATURES = ["elo_diff", "home_adv", "elo_diff_sq", "elo_diff_abs"]
TRAIN_CUTOFF = pd.Timestamp("2018-01-01")


def _build_features(match_df: pd.DataFrame) -> pd.DataFrame:
    mdf = match_df[
        (match_df["date"].dt.year >= 1970) & (match_df["result"] != 0.5)
    ].copy()
    mdf["home_win"] = (mdf["result"] == 1.0).astype(int)
    mdf["home_adv"] = (~mdf["neutral"]).astype(int)
    mdf["elo_diff_sq"] = mdf["elo_diff"] ** 2
    mdf["elo_diff_abs"] = mdf["elo_diff"].abs()
    return mdf


def train_model(match_df: pd.DataFrame) -> tuple:
    """Train logistic regression. Returns (model, metrics_dict)."""
    mdf = _build_features(match_df)
    train = mdf[mdf["date"] < TRAIN_CUTOFF]
    test = mdf[mdf["date"] >= TRAIN_CUTOFF]

    X_train, y_train = train[FEATURES], train["home_win"]
    X_test, y_test = test[FEATURES], test["home_win"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "brier": round(brier_score_loss(y_test, proba), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    return model, metrics


def win_prob(model, elo_a: float, elo_b: float, home: bool = False) -> float:
    """P(team A beats team B) from trained model."""
    d = elo_a - elo_b
    X = [[d, int(home), d ** 2, abs(d)]]
    return float(model.predict_proba(X)[0][1])


def wdl_probs(elo_a: float, elo_b: float) -> tuple[float, float, float]:
    """Win / Draw / Loss probabilities on neutral ground."""
    d = elo_a - elo_b
    p_draw = 0.25 * np.exp(-abs(d) / 500)
    p_win = (1 / (1 + 10 ** (-d / 400))) * (1 - p_draw)
    p_loss = 1 - p_win - p_draw
    return p_win, p_draw, p_loss
