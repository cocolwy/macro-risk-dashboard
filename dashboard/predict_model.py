"""
Predictive model: estimates probability of >5% drawdown in next 20 trading days.
Uses rate-of-change features from existing macro indicators.

This script is called by the daily update workflow after fetch_macro_data.py finishes.
Outputs: crash_prediction.json, model_metrics.json
"""

import json
import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc

warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA_DIR = Path(__file__).parent / 'data'

INDICATOR_PARSERS = {
    'term_spread': lambda d: d.get('term_spread_10y2y'),
    'credit_spread': lambda d: d.get('high_yield_spread'),
    'vix': lambda d: d.get('vix'),
    'sp500': lambda d: d.get('sp500'),
    'absorption_ratio': lambda d: d.get('absorption_ratio'),
    'turbulence': lambda d: d.get('turbulence'),
    'breadth': lambda d: d.get('pct_above_200ma'),
}


def load_indicators() -> pd.DataFrame:
    data = {}
    for name, parser in INDICATOR_PARSERS.items():
        path = DATA_DIR / f'{name}.json'
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            raw = json.load(f)
        data[name] = {d['date']: parser(d) for d in raw if 'date' in d}

    all_dates = sorted(set(data.get('sp500', {}).keys()) &
                       set(data.get('vix', {}).keys()) &
                       set(data.get('turbulence', {}).keys()))

    df = pd.DataFrame(index=all_dates)
    for name, series in data.items():
        df[name] = df.index.map(lambda d, s=series: s.get(d))

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.ffill().dropna(subset=['sp500', 'vix'])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    for col in ['vix', 'turbulence', 'absorption_ratio']:
        if col in df.columns:
            features[f'{col}_5d_chg'] = df[col].pct_change(5)
            features[f'{col}_10d_chg'] = df[col].pct_change(10)
            features[f'{col}_20d_chg'] = df[col].pct_change(20)

    if 'term_spread' in df.columns:
        features['term_spread_level'] = df['term_spread']
        features['term_spread_5d_chg'] = df['term_spread'].diff(5)
        features['term_spread_20d_chg'] = df['term_spread'].diff(20)

    if 'credit_spread' in df.columns:
        features['credit_spread_level'] = df['credit_spread']
        features['credit_spread_5d_chg'] = df['credit_spread'].diff(5)
        features['credit_spread_10d_chg'] = df['credit_spread'].diff(10)

    if 'breadth' in df.columns:
        features['breadth_level'] = df['breadth']
        features['breadth_10d_chg'] = df['breadth'].diff(10)

    if 'vix' in df.columns:
        features['vix_level'] = df['vix']
        features['vix_vs_20d_avg'] = df['vix'] / df['vix'].rolling(20).mean()
        features['vix_volatility'] = df['vix'].rolling(10).std()

    if 'sp500' in df.columns:
        features['sp500_20d_ret'] = df['sp500'].pct_change(20)
        features['sp500_50d_ret'] = df['sp500'].pct_change(50)
        features['sp500_vs_50ma'] = df['sp500'] / df['sp500'].rolling(50).mean() - 1

    return features


def build_features_slim(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated feature set: 1 best window per indicator, 11 features total."""
    features = pd.DataFrame(index=df.index)

    if 'vix' in df.columns:
        features['vix_level'] = df['vix']
        features['vix_20d_chg'] = df['vix'].pct_change(20)

    if 'turbulence' in df.columns:
        features['turbulence_20d_chg'] = df['turbulence'].pct_change(20)

    if 'absorption_ratio' in df.columns:
        features['absorption_ratio_20d_chg'] = df['absorption_ratio'].pct_change(20)

    if 'term_spread' in df.columns:
        features['term_spread_level'] = df['term_spread']
        features['term_spread_20d_chg'] = df['term_spread'].diff(20)

    if 'credit_spread' in df.columns:
        features['credit_spread_10d_chg'] = df['credit_spread'].diff(10)

    if 'breadth' in df.columns:
        features['breadth_level'] = df['breadth']
        features['breadth_10d_chg'] = df['breadth'].diff(10)

    if 'sp500' in df.columns:
        features['sp500_vs_50ma'] = df['sp500'] / df['sp500'].rolling(50).mean() - 1

    return features


def fetch_regime_data() -> pd.DataFrame:
    """Fetch Fed Funds Rate and CPI from FRED for regime features."""
    start = '2016-01-01'
    regime = pd.DataFrame()

    try:
        url_dff = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF&cosd={start}&fq=Daily"
        dff = pd.read_csv(url_dff)
        date_col = [c for c in dff.columns if 'date' in c.lower()][0]
        dff[date_col] = pd.to_datetime(dff[date_col])
        dff = dff.set_index(date_col)
        col = [c for c in dff.columns if c != date_col][0]
        regime['fed_funds'] = pd.to_numeric(dff[col], errors='coerce')
        print(f"  Fed Funds Rate: {len(regime['fed_funds'].dropna())} points")
    except Exception as e:
        print(f"  Warning: Failed to fetch DFF: {e}")

    try:
        url_cpi = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&cosd={start}&fq=Monthly"
        cpi = pd.read_csv(url_cpi)
        date_col = [c for c in cpi.columns if 'date' in c.lower()][0]
        cpi[date_col] = pd.to_datetime(cpi[date_col])
        cpi = cpi.set_index(date_col)
        col = [c for c in cpi.columns if c != date_col][0]
        cpi_s = pd.to_numeric(cpi[col], errors='coerce').dropna()
        cpi_yoy = cpi_s.pct_change(12) * 100
        regime['cpi_yoy'] = cpi_yoy
        print(f"  CPI YoY: {len(regime['cpi_yoy'].dropna())} points")
    except Exception as e:
        print(f"  Warning: Failed to fetch CPI: {e}")

    return regime


def build_features_regime(df: pd.DataFrame, regime_df: pd.DataFrame = None) -> pd.DataFrame:
    """Slim features + regime features (fed funds direction, CPI trend, curve state).

    Regime features use fillna(0) for binary flags and ffill+fillna for levels,
    so they never introduce NaN rows that would be dropped by prep().
    """
    features = build_features_slim(df)

    if 'term_spread' in df.columns:
        ts = df['term_spread']
        features['curve_inverted'] = (ts < 0).astype(float)
        features['curve_flat'] = ((ts >= 0) & (ts < 0.5)).astype(float)

    if regime_df is not None and 'fed_funds' in regime_df.columns:
        ff_idx = pd.to_datetime(regime_df.index)
        ff_series = regime_df['fed_funds'].copy()
        ff_series.index = ff_idx
        df_idx = pd.to_datetime(df.index)
        ff = ff_series.reindex(df_idx, method='ffill').fillna(method='bfill').fillna(0)
        ff.index = df.index
        features['fed_rate_level'] = ff
        chg63 = ff.diff(63).fillna(0)
        features['fed_rate_chg_63d'] = chg63
        features['fed_hiking'] = (chg63 > 0.25).astype(float)
        features['fed_cutting'] = (chg63 < -0.25).astype(float)

    if regime_df is not None and 'cpi_yoy' in regime_df.columns:
        cpi_idx = pd.to_datetime(regime_df.index)
        cpi_series = regime_df['cpi_yoy'].copy()
        cpi_series.index = cpi_idx
        df_idx = pd.to_datetime(df.index)
        cpi = cpi_series.reindex(df_idx, method='ffill').fillna(method='bfill').fillna(0)
        cpi.index = df.index
        features['cpi_yoy'] = cpi
        features['cpi_accelerating'] = (cpi.diff(1).fillna(0) > 0).astype(float)
        features['cpi_above_3'] = (cpi > 3.0).astype(float)

    return features


def get_event_dates():
    """Known FOMC meeting dates, CPI release dates (2022-2026). NFP = 1st Friday of month."""
    fomc = [
        '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27',
        '2022-09-21', '2022-11-02', '2022-12-14',
        '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26',
        '2023-09-20', '2023-11-01', '2023-12-13',
        '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31',
        '2024-09-18', '2024-11-07', '2024-12-18',
        '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30',
        '2025-09-17', '2025-10-29', '2025-12-17',
        '2026-01-28', '2026-03-18', '2026-05-06', '2026-06-17', '2026-07-29',
    ]
    cpi_releases = [
        '2022-01-12', '2022-02-10', '2022-03-10', '2022-04-12', '2022-05-11',
        '2022-06-10', '2022-07-13', '2022-08-10', '2022-09-13', '2022-10-13',
        '2022-11-10', '2022-12-13',
        '2023-01-12', '2023-02-14', '2023-03-14', '2023-04-12', '2023-05-10',
        '2023-06-13', '2023-07-12', '2023-08-10', '2023-09-13', '2023-10-12',
        '2023-11-14', '2023-12-12',
        '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15',
        '2024-06-12', '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10',
        '2024-11-13', '2024-12-11',
        '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10', '2025-05-13',
        '2025-06-11', '2025-07-10', '2025-08-12', '2025-09-10', '2025-10-14',
        '2025-11-12', '2025-12-10',
        '2026-01-13', '2026-02-11', '2026-03-11', '2026-04-14', '2026-05-12',
        '2026-06-10', '2026-07-14',
    ]
    return {
        'fomc': [pd.Timestamp(d) for d in fomc],
        'cpi': [pd.Timestamp(d) for d in cpi_releases],
    }


def build_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Days to/from next FOMC and CPI release. NFP = 1st Friday of each month."""
    events = get_event_dates()
    dates = pd.to_datetime(df.index)
    features = pd.DataFrame(index=df.index)

    for event_name, event_dates in events.items():
        event_set = set(event_dates)
        days_to = []
        days_since = []
        for d in dates:
            future = [e for e in event_dates if e >= d]
            past = [e for e in event_dates if e <= d]
            days_to.append((future[0] - d).days if future else 30)
            days_since.append((d - past[-1]).days if past else 30)

        features[f'{event_name}_days_to'] = days_to
        features[f'{event_name}_days_since'] = days_since
        features[f'{event_name}_within_3d'] = (pd.Series(days_to, index=df.index) <= 3).astype(float)
        features[f'{event_name}_within_7d'] = (pd.Series(days_to, index=df.index) <= 7).astype(float)

    nfp_dates = set()
    for year in range(2022, 2027):
        for month in range(1, 13):
            first = pd.Timestamp(year, month, 1)
            offset = (4 - first.weekday()) % 7
            nfp_dates.add(first + pd.Timedelta(days=offset))

    nfp_list = sorted(nfp_dates)
    nfp_days_to = []
    for d in dates:
        future = [e for e in nfp_list if e >= d]
        nfp_days_to.append((future[0] - d).days if future else 30)
    features['nfp_within_3d'] = (pd.Series(nfp_days_to, index=df.index) <= 3).astype(float)

    return features


def build_features_with_events(df: pd.DataFrame) -> pd.DataFrame:
    """Slim features + event calendar features."""
    features = build_features_slim(df)
    event_feats = build_event_features(df)
    for col in event_feats.columns:
        features[col] = event_feats[col]
    return features


def build_features_kitchen_sink(df: pd.DataFrame, regime_df: pd.DataFrame = None) -> pd.DataFrame:
    """Slim + minimal regime + event features — the full combined feature set."""
    features = build_features_regime_minimal(df, regime_df)
    event_feats = build_event_features(df)
    for col in event_feats.columns:
        features[col] = event_feats[col]
    return features


def build_features_regime_minimal(df: pd.DataFrame, regime_df: pd.DataFrame = None) -> pd.DataFrame:
    """Slim + only 2 key regime features: curve_inverted + fed_rate_direction."""
    features = build_features_slim(df)

    if 'term_spread' in df.columns:
        features['curve_inverted'] = (df['term_spread'] < 0).astype(float)

    if regime_df is not None and 'fed_funds' in regime_df.columns:
        ff_idx = pd.to_datetime(regime_df.index)
        ff_series = regime_df['fed_funds'].copy()
        ff_series.index = ff_idx
        df_idx = pd.to_datetime(df.index)
        ff = ff_series.reindex(df_idx, method='ffill').fillna(method='bfill').fillna(0)
        ff.index = df.index
        chg63 = ff.diff(63).fillna(0)
        features['fed_hiking'] = (chg63 > 0.25).astype(float)

    return features


def compute_target(sp500: pd.Series, horizon: int = 20, threshold: float = -0.05) -> pd.Series:
    """Binary target: 1 if max drawdown in next `horizon` days exceeds `threshold`."""
    future_max_dd = pd.Series(index=sp500.index, dtype=float)
    for i in range(len(sp500) - horizon):
        current = sp500.iloc[i]
        future_window = sp500.iloc[i + 1:i + horizon + 1]
        dd = future_window.min() / current - 1
        future_max_dd.iloc[i] = dd
    return (future_max_dd < threshold).astype(int)


HUMAN_WEIGHTS = {
    'vix_5d_chg': +0.20, 'vix_10d_chg': +0.15, 'vix_20d_chg': +0.15,
    'turbulence_5d_chg': +0.25, 'turbulence_10d_chg': +0.15, 'turbulence_20d_chg': +0.10,
    'absorption_ratio_5d_chg': +0.40, 'absorption_ratio_10d_chg': +0.20, 'absorption_ratio_20d_chg': +0.10,
    'term_spread_level': -0.50, 'term_spread_5d_chg': -0.15, 'term_spread_20d_chg': -0.10,
    'credit_spread_level': +0.20, 'credit_spread_5d_chg': +0.30, 'credit_spread_10d_chg': +0.80,
    'breadth_level': -0.35, 'breadth_10d_chg': -0.40,
    'vix_level': +0.25, 'vix_vs_20d_avg': +0.20,
    'sp500_20d_ret': -0.50, 'sp500_50d_ret': -0.40, 'sp500_vs_50ma': -0.70,
    'vix_volatility': +0.15,
}


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, split_ratio: float = 0.7,
                       embargo: int = 0, subsample_step: int = 1):
    split = int(len(X) * split_ratio)

    train_end = max(split - embargo, 1)
    X_train_raw = X.iloc[:train_end]
    y_train_raw = y.iloc[:train_end]

    test_start = min(split + embargo, len(X))
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    if subsample_step > 1:
        idx = list(range(0, len(X_train_raw), subsample_step))
        X_train = X_train_raw.iloc[idx]
        y_train = y_train_raw.iloc[idx]
    else:
        X_train = X_train_raw
        y_train = y_train_raw

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_test_s)[:, 1]

    return model, scaler, X_train, X_test, y_train, y_test, y_prob


def human_model_probs(X: pd.DataFrame, scaler: StandardScaler, ml_model=None, y_train=None) -> np.ndarray:
    """Compute crash probabilities using manually designed weights.
    Auto-calibrates scale + bias so mean probability matches base rate.
    Zeros out weights for features that are >50% zeros (unreliable data)."""
    weights = np.array([HUMAN_WEIGHTS.get(col, 0.0) for col in X.columns])

    zero_rates = (X == 0).mean()
    for i, col in enumerate(X.columns):
        if zero_rates[col] > 0.5:
            weights[i] = 0.0

    raw_scores = scaler.transform(X) @ weights

    if y_train is not None and len(y_train) > 0:
        base_rate = float(y_train.mean())
        target_logit = np.log(base_rate / (1 - base_rate))
        median_score = np.median(raw_scores)
        std_score = np.std(raw_scores)
        if std_score > 0:
            scale = 1.5 / std_score
            bias = target_logit - median_score * scale
            calibrated = raw_scores * scale + bias
            calibrated = np.clip(calibrated, -6, 6)
            return 1 / (1 + np.exp(-calibrated))

    raw_scores = np.clip(raw_scores, -6, 6)
    return 1 / (1 + np.exp(-raw_scores))


def build_metrics(model, scaler, X, y, X_train, X_test, y_test, y_prob, sp500):
    # PR curve
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)
    pr_curve = [{"threshold": round(float(t), 3), "precision": round(float(p), 3), "recall": round(float(r), 3)}
                for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds_pr)]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_data = [{"fpr": round(float(f), 3), "tpr": round(float(t), 3)} for f, t in zip(fpr, tpr)]

    # Feature importance
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    feature_importance = [{"feature": k, "weight": round(float(v), 4)} for k, v in coefs.items()]

    # Full probability timeline
    X_all_s = scaler.transform(X)
    all_probs = model.predict_proba(X_all_s)[:, 1]
    prob_timeline = [{"date": d, "probability": round(float(p), 4)} for d, p in zip(X.index, all_probs)]

    # SP500 timeline
    sp500_timeline = [{"date": d, "sp500": round(float(sp500.loc[d]), 2)}
                      for d in X.index if d in sp500.index]

    # Threshold analysis
    threshold_analysis = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (y_prob > thresh).astype(int)
        tp = ((preds == 1) & (y_test.values == 1)).sum()
        fp = ((preds == 1) & (y_test.values == 0)).sum()
        fn = ((preds == 0) & (y_test.values == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        alert_days = int(preds.sum())
        threshold_analysis.append({
            "threshold": thresh,
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "f1": round(float(f1), 3),
            "alert_days": alert_days,
            "total_days": len(preds),
            "alert_pct": round(alert_days / len(preds) * 100, 1),
        })

    # Backtest events (only those in test window)
    events_backtest = []
    key_events = [
        {"name": "2024.8 日元套利平仓", "start": "2024-07-15", "peak": "2024-08-05", "drop_pct": -8.5},
        {"name": "2025.4 关税冲击", "start": "2025-03-15", "peak": "2025-04-08", "drop_pct": -12.1},
    ]
    for evt in key_events:
        window = [(d, p) for d, p in zip(X.index, all_probs) if evt["start"] <= d <= evt["peak"]]
        if window:
            first_alert = next(((d, p) for d, p in window if p > 0.5), None)
            max_prob_item = max(window, key=lambda x: x[1])
            events_backtest.append({
                "name": evt["name"],
                "event_date": evt["peak"],
                "drop_pct": evt["drop_pct"],
                "first_alert_date": first_alert[0] if first_alert else None,
                "lead_days": (pd.Timestamp(evt["peak"]) - pd.Timestamp(first_alert[0])).days if first_alert else None,
                "max_probability": round(float(max_prob_item[1]), 3),
                "max_prob_date": max_prob_item[0],
            })

    latest_prob = float(all_probs[-1])
    signal = "elevated" if latest_prob > 0.5 else ("watch" if latest_prob > 0.3 else "low")

    return {
        "model_info": {
            "name": "Logistic Regression v1",
            "target": "Max drawdown > 5% in next 20 trading days",
            "features_count": len(X.columns),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "positive_rate": round(float(y.mean()), 3),
            "train_period": f"{X_train.index[0]} ~ {X_train.index[-1]}",
            "test_period": f"{X_test.index[0]} ~ {X_test.index[-1]}",
            "roc_auc": round(float(roc_auc), 3),
            "last_updated": X.index[-1],
        },
        "current_prediction": {
            "date": X.index[-1],
            "probability": round(latest_prob, 4),
            "signal": signal,
        },
        "pr_curve": pr_curve,
        "roc_curve": roc_data,
        "feature_importance": feature_importance,
        "threshold_analysis": threshold_analysis,
        "events_backtest": events_backtest,
        "probability_timeline": prob_timeline,
        "sp500_timeline": sp500_timeline,
    }, prob_timeline


def build_comparison_metrics(y_test, y_prob, all_probs, X, sp500, model_name, key_events):
    """Build a lightweight metrics dict for a comparison model (no feature importance / ROC)."""
    roc_auc = auc(*roc_curve(y_test, y_prob)[:2])

    threshold_analysis = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (y_prob > thresh).astype(int)
        tp = ((preds == 1) & (y_test.values == 1)).sum()
        fp = ((preds == 1) & (y_test.values == 0)).sum()
        fn = ((preds == 0) & (y_test.values == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        threshold_analysis.append({
            "threshold": thresh, "precision": round(float(prec), 3),
            "recall": round(float(rec), 3), "f1": round(float(f1), 3),
            "alert_days": int(preds.sum()), "total_days": len(preds),
            "alert_pct": round(int(preds.sum()) / len(preds) * 100, 1),
        })

    events_backtest = []
    for evt in key_events:
        window = [(d, p) for d, p in zip(X.index, all_probs) if evt["start"] <= d <= evt["peak"]]
        if window:
            first_alert = next(((d, p) for d, p in window if p > 0.5), None)
            max_prob_item = max(window, key=lambda x: x[1])
            events_backtest.append({
                "name": evt["name"], "event_date": evt["peak"], "drop_pct": evt["drop_pct"],
                "first_alert_date": first_alert[0] if first_alert else None,
                "lead_days": (pd.Timestamp(evt["peak"]) - pd.Timestamp(first_alert[0])).days if first_alert else None,
                "max_probability": round(float(max_prob_item[1]), 3),
                "max_prob_date": max_prob_item[0],
            })

    prob_timeline = [{"date": d, "probability": round(float(p), 4)} for d, p in zip(X.index, all_probs)]
    latest = float(all_probs[-1])

    return {
        "name": model_name,
        "auc": round(float(roc_auc), 3),
        "current_probability": round(latest, 4),
        "current_signal": "elevated" if latest > 0.5 else ("watch" if latest > 0.3 else "low"),
        "threshold_analysis": threshold_analysis,
        "events_backtest": events_backtest,
        "probability_timeline": prob_timeline,
    }


KEY_EVENTS = [
    {"name": "2024.8 日元套利平仓", "start": "2024-07-01", "peak": "2024-08-05", "drop_pct": -8.5},
    {"name": "2025.4 关税冲击", "start": "2025-03-01", "peak": "2025-04-08", "drop_pct": -12.1},
]


def main():
    print("=== Prediction Model: Daily Update ===")

    print("Loading indicators...")
    df = load_indicators()
    print(f"  {len(df)} days of data")

    print("Building features...")
    features = build_features(df)

    print("Computing target...")
    target = compute_target(df['sp500'])

    combined = features.copy()
    combined['target'] = target
    combined = combined.dropna()

    X = combined.drop('target', axis=1)
    y = combined['target']
    print(f"  {len(X)} usable samples, {y.sum()} positive ({y.mean()*100:.1f}%)")

    print("Training ML model...")
    X_clipped = X.clip(-10, 10)
    model, scaler, X_train, X_test, y_train, y_test, y_prob = train_and_evaluate(X_clipped, y)

    print("Building ML metrics...")
    metrics, prob_timeline = build_metrics(model, scaler, X_clipped, y, X_train, X_test, y_test, y_prob, df['sp500'])

    print("Building Human Logic model...")
    split = int(len(X_clipped) * 0.7)
    human_probs_all = human_model_probs(X_clipped, scaler, model, y_train)
    human_probs_test = human_probs_all[split:]
    human_comparison = build_comparison_metrics(
        y_test, human_probs_test, human_probs_all,
        X_clipped, df['sp500'], "Human Logic v1", KEY_EVENTS,
    )

    # Attach AB comparison data
    metrics["experiments"] = [
        {
            "name": "ML (Logistic Regression)",
            "auc": metrics["model_info"]["roc_auc"],
            "current_probability": metrics["current_prediction"]["probability"],
            "current_signal": metrics["current_prediction"]["signal"],
            "threshold_analysis": metrics["threshold_analysis"],
            "events_backtest": metrics["events_backtest"],
            "probability_timeline": metrics["probability_timeline"],
        },
        human_comparison,
    ]

    # Weight comparison
    ml_coefs = pd.Series(model.coef_[0], index=X_clipped.columns)
    weight_comparison = []
    for col in X_clipped.columns:
        mc = float(ml_coefs[col])
        hc = HUMAN_WEIGHTS.get(col, 0.0)
        agree = "same" if (mc > 0.01 and hc > 0) or (mc < -0.01 and hc < 0) else ("zero" if abs(mc) < 0.01 else "diff")
        weight_comparison.append({"feature": col, "ml_weight": round(mc, 4), "human_weight": hc, "agree": agree})
    metrics["weight_comparison"] = weight_comparison

    # --- Slim features experiment (deduplicated, 11 features) ---
    print("Building Slim ML model (dedup features)...")
    features_slim = build_features_slim(df)
    combined_slim = features_slim.copy()
    combined_slim['target'] = target
    combined_slim = combined_slim.dropna()
    X_slim = combined_slim.drop('target', axis=1).clip(-10, 10)
    y_slim = combined_slim['target']

    split_slim = int(len(X_slim) * 0.7)
    X_train_s, X_test_s = X_slim.iloc[:split_slim], X_slim.iloc[split_slim:]
    y_train_s, y_test_s = y_slim.iloc[:split_slim], y_slim.iloc[split_slim:]

    scaler_slim = StandardScaler()
    X_train_ss = scaler_slim.fit_transform(X_train_s)
    X_test_ss = scaler_slim.transform(X_test_s)

    model_slim = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    model_slim.fit(X_train_ss, y_train_s)
    slim_probs_test = model_slim.predict_proba(X_test_ss)[:, 1]
    slim_probs_all = model_slim.predict_proba(scaler_slim.transform(X_slim))[:, 1]

    slim_comparison = build_comparison_metrics(
        y_test_s, slim_probs_test, slim_probs_all,
        X_slim, df['sp500'], f"ML Slim ({len(X_slim.columns)}feat)", KEY_EVENTS,
    )
    metrics["experiments"].append(slim_comparison)
    print(f"  Slim AUC: {slim_comparison['auc']} ({len(X_slim.columns)} features vs {len(X_clipped.columns)} original)")

    # --- D1: Slim + Embargo (20-day gap between train/test) ---
    print("Building D1: Slim + Embargo(20d)...")
    EMBARGO = 20
    model_d1, scaler_d1, X_train_d1, X_test_d1, y_train_d1, y_test_d1, y_prob_d1 = \
        train_and_evaluate(X_slim, y_slim, embargo=EMBARGO)
    d1_probs_all = model_d1.predict_proba(scaler_d1.transform(X_slim))[:, 1]
    d1_comparison = build_comparison_metrics(
        y_test_d1, y_prob_d1, d1_probs_all,
        X_slim, df['sp500'], f"D1 Slim+Embargo({EMBARGO}d)", KEY_EVENTS,
    )
    metrics["experiments"].append(d1_comparison)
    print(f"  D1 AUC: {d1_comparison['auc']} (embargo={EMBARGO}d, train={len(X_train_d1)}, test={len(X_test_d1)})")

    # --- AND Ensembles: both MIN and logical AND ---
    print("Building AND Ensembles (D1 x Human)...")
    d1_series = pd.Series(d1_probs_all, index=X_slim.index)
    human_series = pd.Series(human_probs_all, index=X_clipped.index)
    common_dates = d1_series.index.intersection(human_series.index)

    d1_test_start = min(int(len(X_slim) * 0.7) + EMBARGO, len(X_slim))
    d1_test_dates = X_slim.index[d1_test_start:]
    test_mask = common_dates.isin(d1_test_dates)
    y_common = y_slim.reindex(common_dates)
    y_and_test = y_common[test_mask].values
    and_ref_X = pd.DataFrame(index=common_dates)

    # MIN version: continuous probability = min(D1, Human)
    min_probs_all = np.minimum(d1_series[common_dates].values, human_series[common_dates].values)
    min_comparison = build_comparison_metrics(
        pd.Series(y_and_test), min_probs_all[test_mask], min_probs_all,
        and_ref_X, df['sp500'], "MIN (D1, Human)", KEY_EVENTS,
    )
    metrics["experiments"].append(min_comparison)
    print(f"  MIN: continuous prob, AUC={min_comparison['auc']}")

    # Logical AND version: binary 0/1, both > 50% = 1
    d1_binary = (d1_series[common_dates].values > 0.5).astype(float)
    human_binary = (human_series[common_dates].values > 0.5).astype(float)
    and_probs_all = d1_binary * human_binary
    and_comparison = build_comparison_metrics(
        pd.Series(y_and_test), and_probs_all[test_mask], and_probs_all,
        and_ref_X, df['sp500'], "AND (D1 & Human)", KEY_EVENTS,
    )
    metrics["experiments"].append(and_comparison)
    print(f"  AND: {int(and_probs_all.sum())} alert days out of {len(and_probs_all)}")

    # Preserve extended experiments from previous runs
    metrics_path = DATA_DIR / 'model_metrics.json'
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                old_metrics = json.load(f)
            ext_experiments = [e for e in old_metrics.get('experiments', []) if 'Ext' in e.get('name', '')]
            if ext_experiments:
                metrics['experiments'].extend(ext_experiments)
                print(f"  Preserved {len(ext_experiments)} extended experiments from previous run")
            if 'experiment_a_info' in old_metrics:
                metrics['experiment_a_info'] = old_metrics['experiment_a_info']
        except (json.JSONDecodeError, KeyError):
            pass

    # Save outputs
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(f"  Saved model_metrics.json")

    with open(DATA_DIR / 'crash_prediction.json', 'w') as f:
        json.dump(prob_timeline, f)
    print(f"  Saved crash_prediction.json ({len(prob_timeline)} points)")

    print(f"\n  ML prediction:    {metrics['current_prediction']['probability']*100:.1f}% ({metrics['current_prediction']['signal']})")
    print(f"  Human prediction: {human_comparison['current_probability']*100:.1f}% ({human_comparison['current_signal']})")
    print(f"  ML AUC: {metrics['model_info']['roc_auc']}  Human AUC: {human_comparison['auc']}  Slim AUC: {slim_comparison['auc']}")
    print("=== Done ===")


if __name__ == '__main__':
    main()
