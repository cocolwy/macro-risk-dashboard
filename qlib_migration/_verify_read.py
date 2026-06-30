"""验证 Qlib 能正确读取我们 dump 的二进制数据，并和 yfinance 原值核对。"""
from pathlib import Path
import qlib
from qlib.data import D

ROOT = Path(__file__).resolve().parent.parent


def main():
    qlib.init(provider_uri=str(ROOT / "qlib_data" / "sp500"), region="us",
              expression_cache=None, dataset_cache=None)
    insts = D.instruments(market="all")
    df = D.features(insts, ["$close", "$open", "$volume", "$factor"],
                    start_time="2024-12-20", end_time="2024-12-30",
                    freq="day")
    print("loaded shape:", df.shape,
          "| n instruments:", df.index.get_level_values(0).nunique())
    aapl = df.xs("AAPL", level=0)
    print("\nAAPL tail:\n", aapl.tail(3))

    import yfinance as yf
    raw = yf.download("AAPL", start="2024-12-26", end="2024-12-31",
                      auto_adjust=True, progress=False)
    yf_close = round(float(raw["Close"].loc["2024-12-27"].iloc[0]), 2)
    qlib_close = round(float(aapl.loc["2024-12-27", "$close"]), 2)
    print(f"\nyfinance AAPL 2024-12-27 close = {yf_close}")
    print(f"qlib     AAPL 2024-12-27 close = {qlib_close}")
    print("MATCH ✓" if abs(yf_close - qlib_close) < 0.01 else "MISMATCH ✗")


if __name__ == "__main__":
    main()
