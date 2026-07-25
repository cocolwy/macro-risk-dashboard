"""
Autoresearch experiment file — THIS IS THE FILE YOU MODIFY.

Everything in this file is fair game:
  - Feature engineering (build_features)
  - Model choice and hyperparameters (train_model)
  - Feature transforms, interactions, windows
  - Anything that improves composite_score

Constraints:
  - Must define build_features(df, extra_data=None) -> pd.DataFrame
  - Must define train_model(X_train, y_train, X_test) -> (model, scaler, probs_test)
  - Must define DESCRIPTION (short string for the experiment log)
  - Do NOT use class_weight='balanced' (inflates probabilities)
  - Do NOT modify evaluate_harness.py or run.py
  - Only use packages already in requirements.txt

Current baseline: LR Slim+Events (the production best-F1 model)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Experiment description (logged to results.tsv)
# ---------------------------------------------------------------------------

DESCRIPTION = "baseline: LR Slim+Events (production best-F1 model)"

# ---------------------------------------------------------------------------
# Feature engineering — modify this to try new features
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, extra_data=None) -> pd.DataFrame:
    """Build feature matrix from raw indicator DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw macro indicators with columns: vix, turbulence, absorption_ratio,
        term_spread, credit_spread, breadth, sp500.
    extra_data : any, optional
        Extra data from config.load_extra() (e.g. regime DataFrame).

    Returns
    -------
    pd.DataFrame
        Feature matrix, one row per trading day. May contain NaN from
        lookback windows — the harness handles dropna.
    """
    features = pd.DataFrame(index=df.index)

    # --- Slim core features (10 features) ---

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

    # --- Event calendar features (9 features) ---
    event_feats = _build_event_features(df)
    for col in event_feats.columns:
        features[col] = event_feats[col]

    return features


# ---------------------------------------------------------------------------
# Event features helper
# ---------------------------------------------------------------------------

def _get_event_dates():
    """FOMC + CPI release dates (2016-2026)."""
    fomc = [
        '2016-01-27', '2016-03-16', '2016-04-27', '2016-06-15', '2016-07-27', '2016-09-21',
        '2016-11-02', '2016-12-14', '2017-02-01', '2017-03-15', '2017-05-03', '2017-06-14',
        '2017-07-26', '2017-09-20', '2017-11-01', '2017-12-13', '2018-01-31', '2018-03-21',
        '2018-05-02', '2018-06-13', '2018-08-01', '2018-09-26', '2018-11-08', '2018-12-19',
        '2019-01-30', '2019-03-20', '2019-05-01', '2019-06-19', '2019-07-31', '2019-09-18',
        '2019-10-04', '2019-10-30', '2019-12-11', '2020-01-29', '2020-03-02', '2020-03-15',
        '2020-04-29', '2020-06-10', '2020-07-29', '2020-09-16', '2020-11-05', '2020-12-16',
        '2021-01-27', '2021-03-17', '2021-04-28', '2021-06-16', '2021-07-28', '2021-09-22',
        '2021-11-03', '2021-12-15', '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15',
        '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14', '2023-02-01', '2023-03-22',
        '2023-05-03', '2023-06-14', '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
        '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31', '2024-09-18',
        '2024-11-07', '2024-12-18', '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
        '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-17', '2026-01-28', '2026-03-18',
        '2026-05-06', '2026-06-17', '2026-07-29',
    ]
    cpi_releases = [
        '2016-01-20', '2016-02-19', '2016-03-16', '2016-04-14', '2016-05-17', '2016-06-16',
        '2016-07-15', '2016-08-16', '2016-09-16', '2016-10-18', '2016-11-17', '2016-12-15',
        '2017-01-18', '2017-02-15', '2017-03-15', '2017-04-14', '2017-05-12', '2017-06-14',
        '2017-07-14', '2017-08-11', '2017-09-14', '2017-10-13', '2017-11-15', '2017-12-13',
        '2018-01-12', '2018-02-14', '2018-03-13', '2018-04-11', '2018-05-10', '2018-06-12',
        '2018-07-12', '2018-08-10', '2018-09-13', '2018-10-11', '2018-11-14', '2018-12-12',
        '2019-01-11', '2019-02-13', '2019-03-12', '2019-04-10', '2019-05-10', '2019-06-12',
        '2019-07-11', '2019-08-13', '2019-09-12', '2019-10-10', '2019-11-13', '2019-12-11',
        '2020-01-14', '2020-02-13', '2020-03-11', '2020-04-10', '2020-05-12', '2020-06-10',
        '2020-07-14', '2020-08-12', '2020-09-11', '2020-10-13', '2020-11-12', '2020-12-10',
        '2021-01-13', '2021-02-10', '2021-03-10', '2021-04-13', '2021-05-12', '2021-06-10',
        '2021-07-13', '2021-08-11', '2021-09-14', '2021-10-13', '2021-11-10', '2021-12-10',
        '2022-01-12', '2022-02-10', '2022-03-10', '2022-04-12', '2022-05-11', '2022-06-10',
        '2022-07-13', '2022-08-10', '2022-09-13', '2022-10-13', '2022-11-10', '2022-12-13',
        '2023-01-12', '2023-02-14', '2023-03-14', '2023-04-12', '2023-05-10', '2023-06-13',
        '2023-07-12', '2023-08-10', '2023-09-13', '2023-10-12', '2023-11-14', '2023-12-12',
        '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15', '2024-06-12',
        '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11',
        '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10', '2025-05-13', '2025-06-11',
        '2025-07-10', '2025-08-12', '2025-09-10', '2025-10-14', '2025-11-12', '2025-12-10',
        '2026-01-13', '2026-02-11', '2026-03-11', '2026-04-14', '2026-05-12', '2026-06-10',
        '2026-07-14',
    ]
    return {
        'fomc': [pd.Timestamp(d) for d in fomc],
        'cpi': [pd.Timestamp(d) for d in cpi_releases],
    }


def _build_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Days to/from FOMC and CPI releases, plus NFP proximity."""
    events = _get_event_dates()
    dates = pd.to_datetime(df.index)
    features = pd.DataFrame(index=df.index)

    for event_name, event_dates in events.items():
        days_to, days_since = [], []
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
    for year in range(2016, 2027):
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


# ---------------------------------------------------------------------------
# Model definition — modify this to try different models / hyperparameters
# ---------------------------------------------------------------------------

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame):
    """Train model and return predictions on test set.

    Parameters
    ----------
    X_train, y_train : training data (already clipped by harness)
    X_test : test data (already clipped by harness)

    Returns
    -------
    model : trained model object
    scaler : fitted scaler (or None)
    probs_test : np.ndarray of predicted probabilities on X_test
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train_s, y_train)

    probs_test = model.predict_proba(X_test_s)[:, 1]
    return model, scaler, probs_test
