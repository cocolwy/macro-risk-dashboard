"""
Step 1/3 — 把 Russell-1000 的日线 OHLCV 写入 Qlib 二进制格式。

相比 v1 (SP500-107): 使用 RUSSELL1000 全量股票池,
数据写入 qlib_data/russell1000/ (不覆盖 sp500)。

价格用 auto_adjust=True 的复权价 → $factor 设为 1.0。
"""
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import importlib.util as _u
_uni_path = Path(__file__).resolve().parent.parent / "agents" / "universe.py"
_spec = _u.spec_from_file_location("universe", _uni_path)
_uni = _u.module_from_spec(_spec)
_spec.loader.exec_module(_uni)
RUSSELL1000 = _uni.RUSSELL1000

START, END = "2021-06-01", "2024-12-31"
PROVIDER_URI = Path(__file__).resolve().parent.parent / "qlib_data" / "russell1000"
FIELDS = ["open", "high", "low", "close", "volume", "factor"]


def dump_bin(path: Path, start_idx: int, values: np.ndarray):
    arr = np.hstack([[np.float32(start_idx)], values.astype("<f4")]).astype("<f4")
    arr.tofile(path)


def main():
    print(f"[1] 下载 {len(RUSSELL1000)} 只 OHLCV {START}~{END} ...")
    raw = yf.download(RUSSELL1000, start=START, end=END, group_by="ticker",
                      auto_adjust=True, threads=True, progress=True)

    valid = [t for t in RUSSELL1000 if t in raw.columns.get_level_values(0)]
    panels = {}
    for f in ["Open", "High", "Low", "Close", "Volume"]:
        panels[f] = pd.DataFrame({t: raw[t][f] for t in valid})

    close = panels["Close"]
    close = close.dropna(axis=1, thresh=252)
    valid = list(close.columns)
    print(f"    有效标的: {len(valid)} / {len(RUSSELL1000)}")

    calendar = close.index.sort_values()
    cal_dates = [d.strftime("%Y-%m-%d") for d in calendar]
    cal_index = {d: i for i, d in enumerate(calendar)}

    PROVIDER_URI.mkdir(parents=True, exist_ok=True)
    (PROVIDER_URI / "calendars").mkdir(exist_ok=True)
    (PROVIDER_URI / "instruments").mkdir(exist_ok=True)
    (PROVIDER_URI / "features").mkdir(exist_ok=True)

    (PROVIDER_URI / "calendars" / "day.txt").write_text("\n".join(cal_dates) + "\n")

    inst_lines = []
    for t in valid:
        s = close[t].dropna()
        if s.empty:
            continue
        start_d = s.index.min().strftime("%Y-%m-%d")
        end_d = s.index.max().strftime("%Y-%m-%d")
        inst_lines.append(f"{t.upper()}\t{start_d}\t{end_d}")

        fdir = PROVIDER_URI / "features" / t.lower()
        fdir.mkdir(parents=True, exist_ok=True)

        sub_cal = calendar[(calendar >= s.index.min()) & (calendar <= s.index.max())]
        start_idx = cal_index[sub_cal[0]]

        field_series = {
            "open": panels["Open"][t], "high": panels["High"][t],
            "low": panels["Low"][t], "close": panels["Close"][t],
            "volume": panels["Volume"][t],
        }
        for fname, series in field_series.items():
            vals = series.reindex(sub_cal).values.astype("float32")
            dump_bin(fdir / f"{fname}.day.bin", start_idx, vals)
        dump_bin(fdir / "factor.day.bin", start_idx,
                 np.ones(len(sub_cal), dtype="float32"))

    (PROVIDER_URI / "instruments" / "all.txt").write_text("\n".join(inst_lines) + "\n")

    print(f"[2] 已写入 Qlib 数据 → {PROVIDER_URI}")
    print(f"    日历: {len(cal_dates)} 天 ({cal_dates[0]} ~ {cal_dates[-1]})")
    print(f"    标的: {len(inst_lines)}  字段: {FIELDS}")


if __name__ == "__main__":
    main()
