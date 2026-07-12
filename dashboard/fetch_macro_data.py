"""
Macro Risk Dashboard - Data Fetcher
Fetches US-focused macro risk indicators from free data sources:
  - FRED (Federal Reserve Economic Data): term spread, credit spread
  - Yahoo Finance: VIX, S&P 500, market breadth proxies
  - AKShare: supplemental Chinese macro data

Output: JSON files in dashboard/data/ for the frontend to consume.
Designed to run via GitHub Actions on a daily schedule.
"""

import json
import datetime
import os
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
PUBLIC_DATA_DIR = Path(__file__).parent / "frontend" / "public" / "data"
DATA_DIR.mkdir(exist_ok=True)

LOOKBACK_YEARS = 5


def sync_public_data():
    """Mirror dashboard/data into frontend/public/data for Vite dev/build."""
    if PUBLIC_DATA_DIR.exists():
        shutil.rmtree(PUBLIC_DATA_DIR)
    shutil.copytree(DATA_DIR, PUBLIC_DATA_DIR)
    print("  Synced data to frontend/public/data")


def fetch_fred_series(series_id: str, start: str = None) -> pd.Series:
    """Fetch a FRED series via their public CSV endpoint (no API key needed)."""
    if start is None:
        start = (datetime.date.today() - datetime.timedelta(days=LOOKBACK_YEARS * 365)).isoformat()
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&fq=Daily"
    )
    df = pd.read_csv(url)
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    col = [c for c in df.columns if c != date_col][0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.name = series_id
    return s


def fetch_yfinance_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch price history from Yahoo Finance."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    df = t.history(period=period, auto_adjust=True)
    return df


def compute_market_breadth(period="5y") -> pd.DataFrame:
    """
    Compute market breadth: % of sector ETFs trading above their 200-day MA.
    Uses 11 SPDR sector ETFs as a proxy for market internal health.
    """
    import yfinance as yf

    sector_etfs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    data = yf.download(sector_etfs, period=period, auto_adjust=True)["Close"]

    breadth_values = []
    dates = []

    for i in range(200, len(data)):
        row = data.iloc[i]
        ma200 = data.iloc[i - 200:i].mean()
        above_count = (row > ma200).sum()
        total = row.notna().sum()
        if total > 0:
            pct = (above_count / total) * 100
            breadth_values.append(pct)
            dates.append(data.index[i])

    df = pd.DataFrame({"pct_above_200ma": breadth_values}, index=pd.DatetimeIndex(dates))
    return df


def compute_sector_vs_ma(period="1y") -> list:
    """
    Compute each sector ETF's % distance from its 200-day MA.
    Returns a time series so the frontend can show historical sector health.
    """
    import yfinance as yf

    sector_etfs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    data = yf.download(sector_etfs, period="2y", auto_adjust=True)["Close"]

    ma200 = data.rolling(200).mean()
    pct_diff = ((data / ma200) - 1) * 100
    pct_diff = pct_diff.dropna()

    # Keep last 1 year of data
    pct_diff = pct_diff.iloc[-252:]

    records = []
    for date, row in pct_diff.iterrows():
        entry = {"date": date.strftime("%Y-%m-%d")}
        for col in sector_etfs:
            if pd.notna(row[col]):
                entry[col] = round(float(row[col]), 2)
        records.append(entry)

    return records


def compute_absorption_ratio(period="5y", n_components=5, window=60) -> pd.DataFrame:
    """
    Compute Absorption Ratio on a basket of sector ETFs.
    AR = variance explained by top N PCs / total variance
    Uses 11 SPDR sector ETFs as proxy for market structure.
    """
    import yfinance as yf
    from sklearn.decomposition import PCA

    sector_etfs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    data = yf.download(sector_etfs, period=period, auto_adjust=True)["Close"]
    returns = data.pct_change().dropna()

    ar_values = []
    dates = []

    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window:i].dropna(axis=1)
        if chunk.shape[1] < n_components + 1:
            continue
        pca = PCA(n_components=n_components)
        pca.fit(chunk)
        ar = pca.explained_variance_ratio_.sum()
        ar_values.append(ar)
        dates.append(returns.index[i])

    df = pd.DataFrame({"absorption_ratio": ar_values}, index=pd.DatetimeIndex(dates))
    return df


def compute_turbulence_index(period="5y", lookback=252) -> pd.DataFrame:
    """
    Compute Turbulence Index (Mahalanobis distance) on sector ETFs.
    High values = market behavior is abnormal relative to history.
    """
    import yfinance as yf

    sector_etfs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    data = yf.download(sector_etfs, period=period, auto_adjust=True)["Close"]
    returns = data.pct_change().dropna()

    turb_values = []
    dates = []

    for i in range(lookback, len(returns)):
        historical = returns.iloc[i - lookback:i]
        mu = historical.mean().values
        cov = historical.cov().values

        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            continue

        r_t = returns.iloc[i].values
        diff = r_t - mu
        mahal = diff @ cov_inv @ diff.T
        turb = float(mahal)
        if not np.isfinite(turb):
            continue
        turb_values.append(turb)
        dates.append(returns.index[i])

    df = pd.DataFrame({"turbulence": turb_values}, index=pd.DatetimeIndex(dates))
    return df


def series_to_json(s: pd.Series, name: str) -> list:
    """Convert a pandas Series to a list of {date, value} dicts."""
    records = []
    for date, value in s.items():
        if pd.notna(value):
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                name: round(float(value), 4)
            })
    return records


def df_to_json(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of dicts with date field."""
    records = []
    for date, row in df.iterrows():
        entry = {"date": date.strftime("%Y-%m-%d")}
        for col in df.columns:
            if pd.notna(row[col]):
                entry[col] = round(float(row[col]), 4)
        records.append(entry)
    return records


def compute_momentum(all_data: dict) -> dict:
    """Compute weekly and monthly rate of change for each indicator."""
    key_fields = {
        "term_spread": "term_spread_10y2y",
        "credit_spread": "high_yield_spread",
        "vix": "vix",
        "absorption_ratio": "absorption_ratio",
        "turbulence": "turbulence",
        "breadth": "pct_above_200ma",
    }
    momentum = {}
    for name, field in key_fields.items():
        data = all_data.get(name, [])
        if len(data) < 22:
            continue
        latest = data[-1].get(field)
        week_ago = data[-5].get(field) if len(data) > 5 else None
        month_ago = data[-22].get(field) if len(data) > 22 else None
        if latest is not None and week_ago is not None and month_ago is not None:
            momentum[name] = {
                "current": round(float(latest), 2),
                "week_change": round(float(latest - week_ago), 2),
                "month_change": round(float(latest - month_ago), 2),
            }
    return momentum


def compute_alert_status(data: dict) -> dict:
    """Compute current alert levels for each indicator."""
    alerts = {}

    if "term_spread" in data and data["term_spread"]:
        latest = data["term_spread"][-1]["term_spread_10y2y"]
        if latest < 0:
            alerts["term_spread"] = {"level": "danger", "message": f"Yield curve INVERTED: {latest:.2f}%"}
        elif latest < 0.3:
            alerts["term_spread"] = {"level": "warning", "message": f"Yield curve nearly flat: {latest:.2f}%"}
        else:
            alerts["term_spread"] = {"level": "ok", "message": f"Normal: {latest:.2f}%"}

    if "credit_spread" in data and data["credit_spread"]:
        latest_hy = data["credit_spread"][-1].get("high_yield_spread")
        if latest_hy:
            if latest_hy > 6:
                alerts["credit_spread"] = {"level": "danger", "message": f"HY spread extreme: {latest_hy:.0f}bps"}
            elif latest_hy > 4:
                alerts["credit_spread"] = {"level": "warning", "message": f"HY spread elevated: {latest_hy:.0f}bps"}
            else:
                alerts["credit_spread"] = {"level": "ok", "message": f"Normal: {latest_hy:.0f}bps"}

    if "vix" in data and data["vix"]:
        latest_vix = data["vix"][-1]["vix"]
        if latest_vix > 30:
            alerts["vix"] = {"level": "danger", "message": f"VIX extreme fear: {latest_vix:.1f}"}
        elif latest_vix > 20:
            alerts["vix"] = {"level": "warning", "message": f"VIX elevated: {latest_vix:.1f}"}
        else:
            alerts["vix"] = {"level": "ok", "message": f"Normal: {latest_vix:.1f}"}

    if "absorption_ratio" in data and data["absorption_ratio"]:
        latest_ar = data["absorption_ratio"][-1]["absorption_ratio"]
        ar_values = [d["absorption_ratio"] for d in data["absorption_ratio"][-252:]]
        ar_mean = np.mean(ar_values)
        ar_std = np.std(ar_values)
        z_score = (latest_ar - ar_mean) / ar_std if ar_std > 0 else 0
        if z_score > 1.5:
            alerts["absorption_ratio"] = {"level": "danger", "message": f"AR={latest_ar:.3f}，偏离均值 {z_score:.1f} 个标准差，市场高度耦合"}
        elif z_score > 1.0:
            alerts["absorption_ratio"] = {"level": "warning", "message": f"AR={latest_ar:.3f}，偏离均值 {z_score:.1f} 个标准差，耦合度上升"}
        else:
            alerts["absorption_ratio"] = {"level": "ok", "message": f"AR={latest_ar:.3f}，正常范围（偏离 {z_score:.1f} 个标准差）"}

    if "turbulence" in data and data["turbulence"]:
        latest_turb = data["turbulence"][-1]["turbulence"]
        turb_values = [d["turbulence"] for d in data["turbulence"][-252:]]
        turb_p90 = np.percentile(turb_values, 90)
        turb_p99 = np.percentile(turb_values, 99)
        if latest_turb > turb_p99:
            alerts["turbulence"] = {"level": "danger", "message": f"Extreme turbulence: {latest_turb:.1f} (>P99)"}
        elif latest_turb > turb_p90:
            alerts["turbulence"] = {"level": "warning", "message": f"Elevated turbulence: {latest_turb:.1f} (>P90)"}
        else:
            alerts["turbulence"] = {"level": "ok", "message": f"Normal: {latest_turb:.1f}"}

    return alerts


def main():
    print("=" * 60)
    print("Macro Risk Dashboard - Data Fetch")
    print(f"Time: {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    all_data = {}

    # 1. US Term Spread (10Y - 2Y)
    print("\n[1/7] Fetching US Term Spread (10Y-2Y)...")
    try:
        spread = fetch_fred_series("T10Y2Y")
        all_data["term_spread"] = series_to_json(spread, "term_spread_10y2y")
        print(f"  ✓ {len(all_data['term_spread'])} data points")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["term_spread"] = []

    # 2. Credit Spread (ICE BofA US High Yield)
    print("\n[2/7] Fetching Credit Spreads...")
    try:
        hy_spread = fetch_fred_series("BAMLH0A0HYM2")
        ig_spread = fetch_fred_series("BAMLC0A0CM")
        hy_df = hy_spread.to_frame("high_yield_spread")
        ig_df = ig_spread.to_frame("investment_grade_spread")
        merged = hy_df.join(ig_df, how="outer").ffill()
        all_data["credit_spread"] = df_to_json(merged)
        print(f"  ✓ {len(all_data['credit_spread'])} data points")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["credit_spread"] = []

    # 3. VIX
    print("\n[3/7] Fetching VIX...")
    try:
        vix_df = fetch_yfinance_history("^VIX")
        vix_series = vix_df["Close"]
        vix_series.name = "vix"
        all_data["vix"] = series_to_json(vix_series, "vix")
        print(f"  ✓ {len(all_data['vix'])} data points")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["vix"] = []

    # 4. S&P 500 (for context)
    print("\n[4/7] Fetching S&P 500...")
    try:
        spx_df = fetch_yfinance_history("^GSPC")
        spx_series = spx_df["Close"]
        spx_series.name = "sp500"
        all_data["sp500"] = series_to_json(spx_series, "sp500")
        print(f"  ✓ {len(all_data['sp500'])} data points")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["sp500"] = []

    # 5. Market Breadth
    print("\n[5/7] Computing Market Breadth...")
    try:
        breadth_df = compute_market_breadth()
        all_data["breadth"] = df_to_json(breadth_df)
        print(f"  ✓ {len(all_data['breadth'])} data points")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["breadth"] = []

    # 5b. Sector vs 200MA (latest snapshot)
    print("\n[5b/7] Computing Sector vs 200MA...")
    try:
        sector_data = compute_sector_vs_ma()
        all_data["sectors"] = sector_data
        print(f"  ✓ {len(sector_data)} data points")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["sectors"] = []

    # 6. Absorption Ratio
    print("\n[6/7] Computing Absorption Ratio...")
    try:
        ar_df = compute_absorption_ratio()
        all_data["absorption_ratio"] = df_to_json(ar_df)
        print(f"  ✓ {len(all_data['absorption_ratio'])} data points")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["absorption_ratio"] = []

    # 7. Turbulence Index
    print("\n[7/7] Computing Turbulence Index...")
    try:
        turb_df = compute_turbulence_index()
        all_data["turbulence"] = df_to_json(turb_df)
        print(f"  ✓ {len(all_data['turbulence'])} data points")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["turbulence"] = []

    # Compute alerts
    print("\n[*] Computing alert status...")
    alerts = compute_alert_status(all_data)
    all_data["alerts"] = alerts
    all_data["last_updated"] = datetime.datetime.now().isoformat()

    # Compute composite score
    print("[*] Computing composite risk score...")
    composite_summary = None
    try:
        from composite_score import compute_composite_score, get_score_label
        scores = compute_composite_score(all_data)
        all_data["composite_score"] = scores
        if scores:
            latest_score = scores[-1]
            label = get_score_label(latest_score["composite_score"])
            composite_summary = {
                "score": latest_score["composite_score"],
                "label": label["label"],
                "level": label["level"],
                "color": label["color"],
                "action": label["action"],
                "components": latest_score["components"],
                "date": latest_score["date"],
            }
            print(f"  ✓ Score: {latest_score['composite_score']:.0f}/100 [{label['label']}]")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_data["composite_score"] = []

    # Compute momentum
    print("[*] Computing momentum...")
    momentum = compute_momentum(all_data)
    all_data["momentum"] = momentum

    # Save individual JSON files for each indicator
    for key in ["term_spread", "credit_spread", "vix", "sp500", "breadth", "sectors", "absorption_ratio", "turbulence", "composite_score", "momentum"]:
        filepath = DATA_DIR / f"{key}.json"
        with open(filepath, "w") as f:
            json.dump(all_data[key], f)
        print(f"  Saved {filepath.name}")

    # Save summary with alerts
    summary = {
        "alerts": alerts,
        "last_updated": all_data["last_updated"],
        "indicators_available": [k for k in all_data if k not in ("alerts", "last_updated") and all_data[k]],
    }
    if composite_summary:
        summary["composite_score"] = composite_summary
    with open(DATA_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")

    sync_public_data()

    print("\n" + "=" * 60)
    print("Done! Alert Summary:")
    for k, v in alerts.items():
        icon = {"ok": "🟢", "warning": "🟡", "danger": "🔴"}.get(v["level"], "⚪")
        print(f"  {icon} {k}: {v['message']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
