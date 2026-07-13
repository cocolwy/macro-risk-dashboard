"""
Experiment A: Extended History (2005-present)
Fetches 20 years of data (vs default 4 years) and re-runs ML + Human Logic models
to test if more training data improves prediction quality.

Outputs experiment results into model_metrics.json as a third experiment.
"""

import json
import warnings
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import yfinance as yf
except ImportError:
    yf = None

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from predict_model import (
    build_features, build_features_slim, compute_target, HUMAN_WEIGHTS, KEY_EVENTS,
    build_comparison_metrics, human_model_probs,
)

DATA_DIR = Path(__file__).parent / 'data'

START_DATE = '2005-01-01'

FRED_SERIES = {
    'term_spread': 'T10Y2Y',
    'credit_spread': 'BAA10Y',
}

SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']


def fetch_fred_series(series_id: str) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={START_DATE}&fq=Daily"
    )
    df = pd.read_csv(url)
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    col = [c for c in df.columns if c != date_col][0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.index = s.index.strftime('%Y-%m-%d')
    return s


def fetch_yf_series(ticker: str) -> pd.Series:
    if yf is None:
        raise ImportError("yfinance not installed")
    data = yf.download(ticker, start=START_DATE, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = close.index.strftime('%Y-%m-%d')
    return close


def compute_absorption_ratio(sector_returns: pd.DataFrame, window: int = 60, n_components: int = 3) -> pd.Series:
    from sklearn.decomposition import PCA
    result = pd.Series(index=sector_returns.index, dtype=float)
    for i in range(window, len(sector_returns)):
        chunk = sector_returns.iloc[i - window:i].dropna(axis=1, how='any')
        if chunk.shape[1] < n_components:
            continue
        pca = PCA(n_components=n_components)
        pca.fit(chunk)
        result.iloc[i] = pca.explained_variance_ratio_.sum()
    return result.dropna()


def compute_turbulence(sector_returns: pd.DataFrame, window: int = 60) -> pd.Series:
    result = pd.Series(index=sector_returns.index, dtype=float)
    for i in range(window, len(sector_returns)):
        chunk = sector_returns.iloc[i - window:i]
        mu = chunk.mean().values
        cov = chunk.cov().values
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        delta = sector_returns.iloc[i].values - mu
        turb = float(delta @ inv_cov @ delta)
        if np.isfinite(turb):
            result.iloc[i] = turb
    return result.dropna()


def compute_breadth(sector_closes: dict[str, pd.Series]) -> pd.Series:
    above_200ma = pd.DataFrame()
    for sym, s in sector_closes.items():
        ma200 = s.rolling(200).mean()
        above_200ma[sym] = (s > ma200).astype(float)
    return above_200ma.mean(axis=1) * 100


def load_extended_data() -> pd.DataFrame:
    print("  Fetching FRED data...")
    fred_data = {}
    for name, sid in FRED_SERIES.items():
        try:
            fred_data[name] = fetch_fred_series(sid)
            print(f"    {name}: {len(fred_data[name])} points")
        except Exception as e:
            print(f"    {name}: FAILED ({e})")

    print("  Fetching VIX & S&P 500...")
    vix = fetch_yf_series('^VIX')
    sp500 = fetch_yf_series('^GSPC')
    print(f"    VIX: {len(vix)}, S&P500: {len(sp500)}")

    print("  Fetching sector ETFs...")
    sector_closes = {}
    for sym in SECTOR_ETFS:
        try:
            s = fetch_yf_series(sym)
            if len(s) > 0:
                sector_closes[sym] = s
        except Exception as e:
            print(f"    {sym}: FAILED ({e})")
    print(f"    Got {len(sector_closes)} sectors")

    sector_df = pd.DataFrame(sector_closes)
    sector_returns = sector_df.pct_change().dropna()

    print("  Computing Absorption Ratio...")
    abs_ratio = compute_absorption_ratio(sector_returns)
    print(f"    {len(abs_ratio)} points")

    print("  Computing Turbulence Index...")
    turbulence = compute_turbulence(sector_returns)
    print(f"    {len(turbulence)} points")

    print("  Computing Market Breadth...")
    breadth = compute_breadth(sector_closes)
    print(f"    {len(breadth)} points")

    all_dates = sorted(set(sp500.index) & set(vix.index))
    df = pd.DataFrame(index=all_dates)
    df['sp500'] = df.index.map(lambda d: sp500.get(d))
    df['vix'] = df.index.map(lambda d: vix.get(d))
    df['term_spread'] = df.index.map(lambda d: fred_data.get('term_spread', pd.Series(dtype=float)).get(d))
    df['credit_spread'] = df.index.map(lambda d: fred_data.get('credit_spread', pd.Series(dtype=float)).get(d))
    df['absorption_ratio'] = df.index.map(lambda d: abs_ratio.get(d))
    df['turbulence'] = df.index.map(lambda d: turbulence.get(d))
    df['breadth'] = df.index.map(lambda d: breadth.get(d))

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.ffill().dropna(subset=['sp500', 'vix'])
    return df


EXTENDED_KEY_EVENTS = [
    {"name": "2008 金融危机", "start": "2008-06-01", "peak": "2008-10-10", "drop_pct": -56.8},
    {"name": "2011 欧债危机", "start": "2011-07-01", "peak": "2011-10-03", "drop_pct": -19.4},
    {"name": "2015.8 中国股灾", "start": "2015-08-01", "peak": "2015-08-25", "drop_pct": -12.4},
    {"name": "2018.Q4 加息恐慌", "start": "2018-09-01", "peak": "2018-12-24", "drop_pct": -19.8},
    {"name": "2020.3 COVID", "start": "2020-02-01", "peak": "2020-03-23", "drop_pct": -33.9},
    {"name": "2022.1 通胀/加息", "start": "2021-12-01", "peak": "2022-06-16", "drop_pct": -23.6},
    {"name": "2024.8 日元套利平仓", "start": "2024-07-01", "peak": "2024-08-05", "drop_pct": -8.5},
    {"name": "2025.4 关税冲击", "start": "2025-03-01", "peak": "2025-04-08", "drop_pct": -12.1},
]


def main():
    print("=" * 60)
    print("EXPERIMENT A: Extended History (2005-present)")
    print("=" * 60)

    print("\n[1/4] Loading extended data from 2005...")
    df = load_extended_data()
    print(f"  Total: {len(df)} trading days ({df.index[0]} ~ {df.index[-1]})")

    print("\n[2/4] Building features & target...")
    features = build_features(df)
    target = compute_target(df['sp500'])
    combined = features.copy()
    combined['target'] = target

    core_cols = [c for c in combined.columns if c.startswith(('vix_', 'sp500_'))]
    core_cols.append('target')
    combined = combined.dropna(subset=core_cols)
    combined = combined.fillna(0)

    X = combined.drop('target', axis=1).clip(-10, 10)
    y = combined['target']
    print(f"  {len(X)} usable samples, {int(y.sum())} positive ({y.mean()*100:.1f}%)")

    non_zero_counts = (X != 0).sum()
    for col in X.columns:
        if non_zero_counts[col] < len(X) * 0.5:
            print(f"    Warning: {col} has {non_zero_counts[col]}/{len(X)} non-zero values")

    EMBARGO = 20

    print(f"\n[3/4] Training models (70/30 split, embargo={EMBARGO}d)...")
    split = int(len(X) * 0.7)
    train_end = max(split - EMBARGO, 1)
    test_start = min(split + EMBARGO, len(X))
    X_train, X_test = X.iloc[:train_end], X.iloc[test_start:]
    y_train, y_test = y.iloc[:train_end], y.iloc[test_start:]
    print(f"  Train: {len(X_train)} days ({X_train.index[0]} ~ {X_train.index[-1]})")
    print(f"  Embargo gap: {EMBARGO*2} days ({X.index[train_end]} ~ {X.index[test_start-1]})")
    print(f"  Test:  {len(X_test)} days ({X_test.index[0]} ~ {X_test.index[-1]})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ml_model = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    ml_model.fit(X_train_s, y_train)
    ml_probs_test = ml_model.predict_proba(X_test_s)[:, 1]
    ml_probs_all = ml_model.predict_proba(scaler.transform(X))[:, 1]

    human_probs_all = human_model_probs(X, scaler, ml_model, y_train)
    human_probs_test = human_probs_all[len(X) - len(X_test):]

    ml_auc = auc(*roc_curve(y_test, ml_probs_test)[:2])
    human_auc = auc(*roc_curve(y_test, human_probs_test)[:2])
    print(f"  ML AUC: {ml_auc:.3f}")
    print(f"  Human AUC: {human_auc:.3f}")

    # D1: Slim features + Embargo on extended data
    print("\n  Training D1 (Slim+Embargo) on extended data...")
    features_slim = build_features_slim(df)
    combined_slim = features_slim.copy()
    combined_slim['target'] = target
    core_slim = [c for c in combined_slim.columns if c.startswith(('vix_', 'sp500_'))]
    core_slim.append('target')
    combined_slim = combined_slim.dropna(subset=core_slim).fillna(0)
    X_slim = combined_slim.drop('target', axis=1).clip(-10, 10)
    y_slim = combined_slim['target']

    split_d1 = int(len(X_slim) * 0.7)
    train_end_d1 = max(split_d1 - EMBARGO, 1)
    test_start_d1 = min(split_d1 + EMBARGO, len(X_slim))
    X_train_d1, X_test_d1 = X_slim.iloc[:train_end_d1], X_slim.iloc[test_start_d1:]
    y_train_d1, y_test_d1 = y_slim.iloc[:train_end_d1], y_slim.iloc[test_start_d1:]

    scaler_d1 = StandardScaler()
    X_train_d1s = scaler_d1.fit_transform(X_train_d1)
    X_test_d1s = scaler_d1.transform(X_test_d1)

    model_d1 = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    model_d1.fit(X_train_d1s, y_train_d1)
    d1_probs_test = model_d1.predict_proba(X_test_d1s)[:, 1]
    d1_probs_all = model_d1.predict_proba(scaler_d1.transform(X_slim))[:, 1]

    d1_auc = auc(*roc_curve(y_test_d1, d1_probs_test)[:2])
    print(f"  D1 Extended AUC: {d1_auc:.3f} (train={len(X_train_d1)}, embargo={EMBARGO}, test={len(X_test_d1)})")

    print("\n[4/4] Building comparison data (all with embargo)...")
    ml_result = build_comparison_metrics(
        y_test, ml_probs_test, ml_probs_all, X, df['sp500'],
        "ML Extended (2005+)", EXTENDED_KEY_EVENTS,
    )
    human_result = build_comparison_metrics(
        y_test, human_probs_test, human_probs_all, X, df['sp500'],
        "Human Extended (2005+)", EXTENDED_KEY_EVENTS,
    )
    d1_result = build_comparison_metrics(
        y_test_d1, d1_probs_test, d1_probs_all, X_slim, df['sp500'],
        "D1 Ext Slim+Embargo", EXTENDED_KEY_EVENTS,
    )

    # AND Ensemble: D1 Extended x Human Extended (logical AND at 50%)
    print("  Building AND Ensemble (D1 Ext x Human Ext)...")
    d1_ext_series = pd.Series(d1_probs_all, index=X_slim.index)
    human_ext_series = pd.Series(human_probs_all, index=X.index)
    common_ext = d1_ext_series.index.intersection(human_ext_series.index)
    d1_ext_binary = (d1_ext_series[common_ext].values > 0.5).astype(float)
    human_ext_binary = (human_ext_series[common_ext].values > 0.5).astype(float)
    and_ext_all = d1_ext_binary * human_ext_binary

    d1_ext_test_dates = X_slim.index[test_start_d1:]
    test_ext_mask = common_ext.isin(d1_ext_test_dates)
    and_ext_test = and_ext_all[test_ext_mask]
    y_and_ext_test = y_slim.reindex(common_ext)[test_ext_mask].values

    and_ext_ref = pd.DataFrame(index=common_ext)
    and_ext_result = build_comparison_metrics(
        pd.Series(y_and_ext_test), and_ext_test, and_ext_all,
        and_ext_ref, df['sp500'], "AND Ext (D1xHuman)", EXTENDED_KEY_EVENTS,
    )

    experiment_a = {
        "ml": ml_result,
        "human": human_result,
        "data_info": {
            "start": df.index[0],
            "end": df.index[-1],
            "total_days": len(df),
            "train_days": len(X_train),
            "test_days": len(X_test),
            "positive_samples": int(y.sum()),
            "positive_rate": round(float(y.mean()), 3),
            "train_period": f"{X_train.index[0]} ~ {X_train.index[-1]}",
            "test_period": f"{X_test.index[0]} ~ {X_test.index[-1]}",
        },
    }

    metrics_path = DATA_DIR / 'model_metrics.json'
    with open(metrics_path) as f:
        metrics = json.load(f)

    base_experiments = [e for e in metrics.get('experiments', []) if 'Ext' not in e['name']]
    metrics['experiments'] = base_experiments + [ml_result, human_result, d1_result, and_ext_result]
    metrics['experiment_a_info'] = experiment_a['data_info']

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(f"\n  Updated model_metrics.json with {len(metrics['experiments'])} experiments")

    print("\n=== Results Summary ===")
    print(f"  {'Model':<28} {'AUC':>6}  {'P@50%':>6}  {'R@50%':>6}")
    print(f"  {'-'*52}")
    for exp in metrics['experiments']:
        row50 = next((r for r in exp['threshold_analysis'] if r['threshold'] == 0.5), None)
        p50 = f"{row50['precision']*100:.1f}%" if row50 else "N/A"
        r50 = f"{row50['recall']*100:.1f}%" if row50 else "N/A"
        print(f"  {exp['name']:<28} {exp['auc']:>6.3f}  {p50:>6}  {r50:>6}")

    print("\n  Event detection:")
    for exp in metrics['experiments']:
        detected = sum(1 for e in exp['events_backtest'] if e.get('lead_days') is not None)
        total = len(exp['events_backtest'])
        print(f"    {exp['name']}: {detected}/{total} events detected")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
