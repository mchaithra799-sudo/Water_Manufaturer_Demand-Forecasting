"""
Model training + evaluation.

Trains two candidate regressors (Random Forest and Histogram Gradient
Boosting), evaluates them with a *time-based* hold-out (train on 2021-2022,
validate on 2023) so the score reflects true forecasting skill, picks the
winner, then refits it on the full history and saves the artefacts.

Run with:  python -m src.train_model
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from . import config
from .features import make_model_matrix


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _scores(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _candidates() -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=config.RANDOM_SEED,
        ),
    }


def train(save: bool = True) -> dict:
    df = pd.read_csv(config.ENRICHED_FILE, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    X_all, feature_columns = make_model_matrix(df)
    y_all = df[config.TARGET_COL].values

    # Time-based split: last calendar year is the validation set.
    split_date = pd.Timestamp("2023-01-01")
    train_mask = (df["Date"] < split_date).values
    val_mask = ~train_mask

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]

    results = {}
    fitted = {}
    for name, model in _candidates().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        results[name] = _scores(y_val, pred)
        fitted[name] = model
        print(f"[train] {name:22s} "
              f"MAE={results[name]['MAE']:8.1f}  "
              f"RMSE={results[name]['RMSE']:8.1f}  "
              f"MAPE={results[name]['MAPE']:5.2f}%  "
              f"R2={results[name]['R2']:.4f}")

    best_name = min(results, key=lambda k: results[k]["RMSE"])
    print(f"[train] best model on hold-out: {best_name}")

    # Rolling time-series cross-validation on the whole series (fairer average).
    cv_scores = _timeseries_cv(best_name, X_all, y_all, df["Date"])
    print(f"[train] 5-fold time-series CV  "
          f"MAE={cv_scores['MAE']:8.1f}  MAPE={cv_scores['MAPE']:5.2f}%  "
          f"R2={cv_scores['R2']:.4f}")

    # Residual std on the hold-out -> used for forecast prediction intervals.
    best_val_pred = fitted[best_name].predict(X_val)
    residual_std = float(np.std(y_val - best_val_pred))

    # Refit the winner on ALL available history for deployment.
    final_model = _candidates()[best_name]
    final_model.fit(X_all, y_all)

    # Feature importance (permutation-free, model-native where available).
    importance = _feature_importance(final_model, feature_columns, X_all, y_all)

    metrics = {
        "best_model": best_name,
        "validation": results,
        "cross_validation": cv_scores,
        "holdout_split_date": str(split_date.date()),
        "residual_std": residual_std,
        "n_train_rows": int(train_mask.sum()),
        "n_val_rows": int(val_mask.sum()),
        "n_features": len(feature_columns),
        "target": config.TARGET_COL,
        "feature_importance": importance,
    }

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, config.MODEL_FILE)
        config.FEATURE_COLUMNS_FILE.write_text(json.dumps(feature_columns, indent=2))
        config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
        print(f"[train] saved model      -> {config.MODEL_FILE.name}")
        print(f"[train] saved features   -> {config.FEATURE_COLUMNS_FILE.name}")
        print(f"[train] saved metrics    -> {config.METRICS_FILE.name}")

    return metrics


def _timeseries_cv(model_name: str, X, y, dates, n_splits: int = 5) -> dict:
    """Rolling-origin cross-validation, chronologically ordered."""
    order = np.argsort(dates.values)
    Xo, yo = X.iloc[order].reset_index(drop=True), y[order]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold = []
    for tr_idx, te_idx in tscv.split(Xo):
        model = _candidates()[model_name]
        model.fit(Xo.iloc[tr_idx], yo[tr_idx])
        pred = model.predict(Xo.iloc[te_idx])
        fold.append(_scores(yo[te_idx], pred))
    return {k: float(np.mean([f[k] for f in fold])) for k in fold[0]}


def _feature_importance(model, feature_columns, X, y) -> list[dict]:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        # HistGradientBoosting has no native importances -> permutation.
        from sklearn.inspection import permutation_importance
        r = permutation_importance(
            model, X, y, n_repeats=5, random_state=config.RANDOM_SEED, n_jobs=-1
        )
        imp = r.importances_mean
    pairs = sorted(
        zip(feature_columns, imp), key=lambda t: t[1], reverse=True
    )
    return [{"feature": f, "importance": float(v)} for f, v in pairs]


if __name__ == "__main__":
    train(save=True)
