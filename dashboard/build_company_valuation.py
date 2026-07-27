#!/usr/bin/env python3
"""Build NVDA valuation + fundamentals JSON for the dashboard pages.

Usage (from repo root, with yfinance available):
  .venv/bin/python dashboard/build_company_valuation.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "NVDA"
PEERS = ["AMD", "AVGO", "INTC", "QCOM", "MRVL"]
ROOT = Path(__file__).resolve().parent
OUT_DIRS = [ROOT / "data", ROOT / "frontend" / "public" / "data"]


def row(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def df_table(df, rows, n=4):
    if df is None or df.empty:
        return {"periods": [], "rows": []}
    cols = list(df.columns[:n])
    periods = [str(c.date()) if hasattr(c, "date") else str(c)[:10] for c in cols]
    table = {"periods": periods, "rows": []}
    for label, aliases in rows:
        s = None
        for a in aliases:
            if a in df.index:
                s = df.loc[a]
                break
        if s is None:
            continue
        vals = []
        for c in cols:
            v = s[c]
            try:
                vals.append(None if pd.isna(v) else float(v))
            except Exception:
                vals.append(None)
        table["rows"].append({"label": label, "values": vals})
    return table


def med(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v) and v > 0]
    return float(np.median(vals)) if vals else None


def main():
    t = yf.Ticker(TICKER)
    info = t.info
    income_a, cashflow_a, balance_a = t.income_stmt, t.cashflow, t.balance_sheet
    income_q, cashflow_q = t.quarterly_income_stmt, t.quarterly_cashflow

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap")
    shares_out = info.get("sharesOutstanding")
    total_debt = float(info.get("totalDebt") or 0)
    cash = float(info.get("totalCash") or 0)
    beta = float(info.get("beta") or 1.5)

    hist = t.history(period="1y")
    px_now = float(hist["Close"].iloc[-1]) if len(hist) else float(price)
    px_3m = float(hist["Close"].iloc[-63]) if len(hist) > 63 else None
    px_12m = float(hist["Close"].iloc[0]) if len(hist) else None

    rev = row(income_a, ["Total Revenue"])
    ebit = row(income_a, ["Operating Income", "EBIT"])
    da = row(cashflow_a, ["Depreciation And Amortization"])
    capex = row(cashflow_a, ["Capital Expenditure"])
    nwc = row(cashflow_a, ["Change In Working Capital"])
    tax_exp = row(income_a, ["Tax Provision"])
    pretax = row(income_a, ["Pretax Income", "Income Before Tax"])
    interest = row(income_a, ["Interest Expense"])
    rev_q = row(income_q, ["Total Revenue"])
    ebit_q = row(income_q, ["Operating Income", "EBIT"])
    da_q = row(cashflow_q, ["Depreciation And Amortization"])

    rev_ttm = float(rev_q.iloc[:4].sum()) if rev_q is not None else float(rev.iloc[0])
    ebit_ttm = float(ebit_q.iloc[:4].sum()) if ebit_q is not None else float(ebit.iloc[0])
    da_ttm = float(da_q.iloc[:4].sum()) if da_q is not None else (float(da.iloc[0]) if da is not None else 0.0)
    ebitda_ttm = ebit_ttm + da_ttm
    ltm_rev_growth = (
        rev_ttm / float(rev_q.iloc[4:8].sum()) - 1
        if rev_q is not None and len(rev_q.dropna()) >= 8
        else None
    )

    ebit_margin = float((ebit / rev).iloc[:3].astype(float).median())
    da_pct = float((da.abs() / rev).iloc[:3].median()) if da is not None else 0.02
    capex_pct = float((capex.abs() / rev).iloc[:3].median()) if capex is not None else 0.03
    nwc_pct = float((nwc.abs() / rev).iloc[:3].median()) if nwc is not None else 0.02

    eff_taxes = []
    if tax_exp is not None and pretax is not None:
        for i in range(min(3, len(tax_exp))):
            pt = float(pretax.iloc[i])
            if pt > 0:
                eff_taxes.append(float(tax_exp.iloc[i]) / pt)
    tax_rate = max(0.15, min(0.30, float(np.median(eff_taxes)) if eff_taxes else 0.15))

    y1 = None
    try:
        re = t.revenue_estimate
        if re is not None and not re.empty and "growth" in re.columns:
            for lab in ["+1y", "0y"]:
                if lab in re.index and pd.notna(re.loc[lab, "growth"]):
                    y1 = float(re.loc[lab, "growth"])
                    break
    except Exception:
        pass
    if y1 is None:
        y1 = float(info.get("revenueGrowth") or 0.25)
    y1_model = min(max(float(y1), 0.05), 0.55)

    g_terminal = 0.025
    growth_path = list(np.linspace(y1_model, g_terminal + 0.01, 5))

    rf = 0.0454
    try:
        tnx = yf.Ticker("^TNX").fast_info.last_price
        if tnx:
            rf = float(tnx) / 100.0
    except Exception:
        pass
    erp = 0.055
    if interest is not None and total_debt > 0:
        kd = min(max(abs(float(interest.dropna().iloc[0])) / total_debt, 0.02), 0.10)
    else:
        kd = 0.055
    ke = rf + beta * erp
    e_v = market_cap / (market_cap + total_debt)
    wacc = e_v * ke + (1 - e_v) * kd * (1 - tax_rate)

    rev0 = rev_ttm
    rev_t = [rev0]
    fcff = []
    proj = []
    for i, g in enumerate(growth_path):
        r = rev_t[-1] * (1 + g)
        rev_t.append(r)
        ebit_y = r * ebit_margin
        nopat = ebit_y * (1 - tax_rate)
        da_y, capex_y, nwc_y = r * da_pct, r * capex_pct, r * nwc_pct
        f = nopat + da_y - capex_y - nwc_y
        fcff.append(f)
        proj.append({
            "year": i + 1, "growth": float(g), "revenue": float(r),
            "ebit": float(ebit_y), "nopat": float(nopat),
            "da": float(da_y), "capex": float(capex_y), "nwc": float(nwc_y), "fcff": float(f),
        })

    exit_mult = 20.0
    ebitda_y5 = rev_t[-1] * ebit_margin + rev_t[-1] * da_pct
    tv_gordon = fcff[-1] * (1 + g_terminal) / (wacc - g_terminal)
    tv_exit = ebitda_y5 * exit_mult
    tv_base = 0.5 * (tv_gordon + tv_exit)
    pv_fcff = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(fcff))
    pv_tv = tv_base / (1 + wacc) ** 5
    ev = pv_fcff + pv_tv
    equity = ev + cash - total_debt
    implied_dcf = equity / shares_out

    peer_rows = []
    for p in PEERS:
        pi = yf.Ticker(p).info
        peer_rows.append({
            "ticker": p,
            "pe_fwd": pi.get("forwardPE"),
            "ev_rev": pi.get("enterpriseToRevenue"),
            "ev_ebitda": pi.get("enterpriseToEbitda"),
            "gross_margin": pi.get("grossMargins"),
            "revenue_growth": pi.get("revenueGrowth"),
            "operating_margin": pi.get("operatingMargins"),
        })

    eve_vals = [p["ev_ebitda"] for p in peer_rows if p["ev_ebitda"] and 5 < p["ev_ebitda"] < 80]
    med_pe = med([p["pe_fwd"] for p in peer_rows])
    med_evr = med([p["ev_rev"] for p in peer_rows])
    med_eve = float(np.median(eve_vals)) if eve_vals else med([p["ev_ebitda"] for p in peer_rows])
    premium = 1.15
    fwd_eps = info.get("forwardEps")
    net_debt = total_debt - cash
    implied_pe = med_pe * premium * float(fwd_eps) if med_pe and fwd_eps else None
    implied_evr = (med_evr * premium * rev_ttm - net_debt) / shares_out if med_evr else None
    implied_eve = (med_eve * premium * ebitda_ttm - net_debt) / shares_out if med_eve else None
    rel_parts = [x for x in [implied_pe, implied_evr, implied_eve] if x and x > 0]
    implied_rel = float(np.median(rel_parts))
    blended = 0.5 * implied_dcf + 0.5 * implied_rel
    upside = blended / float(price) - 1

    wacc_grid = [wacc + dx for dx in (-0.01, -0.005, 0, 0.005, 0.01)]
    g_grid = [0.015, 0.020, 0.025, 0.030, 0.035]
    sens_rows = []
    for w in wacc_grid:
        row_vals = []
        for g in g_grid:
            tv = 0.5 * (fcff[-1] * (1 + g) / (w - g) + ebitda_y5 * exit_mult)
            pv = sum(f / (1 + w) ** (i + 1) for i, f in enumerate(fcff)) + tv / (1 + w) ** 5
            row_vals.append(round((pv + cash - total_debt) / shares_out, 1))
        sens_rows.append({"wacc": round(w, 4), "prices": row_vals})

    def scenario(g_shift, m_shift, w_shift, g_term):
        gp = np.linspace(min(max(y1_model + g_shift, 0.03), 0.70), g_term + 0.01, 5)
        rt = [rev0]
        ff = []
        em = ebit_margin + m_shift
        for g in gp:
            r = rt[-1] * (1 + g)
            rt.append(r)
            ff.append(r * em * (1 - tax_rate) + r * da_pct - r * capex_pct - r * nwc_pct)
        w = wacc + w_shift
        tv = 0.5 * (ff[-1] * (1 + g_term) / (w - g_term) + (rt[-1] * em + rt[-1] * da_pct) * exit_mult)
        pv = sum(f / (1 + w) ** (i + 1) for i, f in enumerate(ff)) + tv / (1 + w) ** 5
        return round((pv + cash - total_debt) / shares_out, 1)

    glossary = [
        {"term": "DCF", "zh": "现金流折现", "def": "把公司未来预计能产生的自由现金流，按资金成本折成今天的价值，再扣净债务得到股权价值与每股公允价值。"},
        {"term": "FCFF", "zh": "企业自由现金流", "def": "给所有资本提供者（股权+债权）的现金：税后经营利润 + 折旧 − 资本开支 − 营运资金增加。"},
        {"term": "WACC", "zh": "加权平均资本成本", "def": "股权与债务融资成本的加权平均，用作折现率。越高，未来现金的今天价值越低。"},
        {"term": "rf", "zh": "无风险利率", "def": "通常用 10 年期美债收益率，作为要求回报的底座。"},
        {"term": "Beta (β)", "zh": "贝塔", "def": "相对市场的波动敏感度。β>1 表示比大盘更波动，股权要求回报更高。"},
        {"term": "ERP", "zh": "股权风险溢价", "def": "股票相对无风险资产长期多要求的超额回报（本模型用 5.5%）。"},
        {"term": "ke", "zh": "股权成本", "def": "ke = rf + β × ERP，股东要求的年化回报率。"},
        {"term": "kd", "zh": "债务成本", "def": "公司借钱的有效利率；税后进入 WACC（利息可抵税）。"},
        {"term": "EBIT margin", "zh": "经营利润率", "def": "经营利润 ÷ 收入。衡量主营业务赚钱效率。"},
        {"term": "g / 永久增长", "zh": "终值增长率", "def": "显式预测期结束后，现金流永续增长的假设，通常接近长期 GDP（约 2–3%）。"},
        {"term": "Terminal Value", "zh": "终值", "def": "第 5 年之后所有现金流的打包价值，常用 Gordon 永续增长与退出倍数两种方法取中。"},
        {"term": "Gordon", "zh": "戈登增长模型", "def": "TV = FCFF₅ × (1+g) / (WACC − g)，假设现金流永续按 g 增长。"},
        {"term": "Exit multiple", "zh": "退出倍数", "def": "用第 5 年 EBITDA × 同业合理 EV/EBITDA，模拟未来出售/再估值。"},
        {"term": "NOPAT", "zh": "税后净营业利润", "def": "EBIT × (1 − 税率)，不含利息的税后经营利润。"},
        {"term": "D&A", "zh": "折旧与摊销", "def": "非现金费用；算 FCFF 时加回（因为没真的花现金）。"},
        {"term": "CapEx", "zh": "资本开支", "def": "买设备、建厂等长期投资现金流出；算 FCFF 时减去。"},
        {"term": "ΔNWC", "zh": "营运资金变动", "def": "应收、存货等占用的现金增加；增长时通常是现金流出。"},
        {"term": "EV", "zh": "企业价值", "def": "股权价值 + 净债务（债务 − 现金），对应整家公司的运营价值。"},
    ]

    valuation = {
        "title": "Company Valuation",
        "subtitle": "DCF + 相对估值三角 · NVDA 案例",
        "ticker": TICKER,
        "name": info.get("shortName") or "NVIDIA Corporation",
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "disclaimer": "研究/教育用途，非投资建议。数据来自 yfinance，请以官方财报为准。",
        "snapshot": {
            "price": float(price),
            "market_cap": float(market_cap),
            "shares_out": int(shares_out),
            "cash": cash,
            "total_debt": total_debt,
            "beta": beta,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "chg_3m": None if px_3m is None else px_now / px_3m - 1,
            "chg_12m": None if px_12m is None else px_now / px_12m - 1,
            "ltm_revenue": rev_ttm,
            "ltm_revenue_growth": ltm_rev_growth,
            "ltm_ebit": ebit_ttm,
            "ltm_ebitda": ebitda_ttm,
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "forward_eps": fwd_eps,
            "trailing_eps": info.get("trailingEps"),
            "forward_pe": info.get("forwardPE"),
            "ev_revenue": info.get("enterpriseToRevenue"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
        },
        "verdict": {
            "blended": round(blended, 1),
            "dcf": round(implied_dcf, 1),
            "relative": round(implied_rel, 1),
            "current": float(price),
            "upside": round(upside, 4),
            "headline": (
                f"NVDA 公允价值约 ${blended:.0f}（混合），现价 ${float(price):.1f} → "
                f"{upside*100:+.0f}%；DCF ${implied_dcf:.0f}，相对估值 ${implied_rel:.0f}。"
            ),
        },
        "methods": [
            {"name": "DCF", "price": round(implied_dcf, 1), "weight": 0.5,
             "note": "5 年 FCFF，WACC 折现，终值 = Gordon 与 20× EBITDA 中位"},
            {"name": "Relative", "price": round(implied_rel, 1), "weight": 0.5,
             "note": "同业中位 P/E·EV/Sales·EV/EBITDA × 1.15 成长溢价（剔极端值）"},
            {"name": "SOTP", "price": None, "weight": 0.0, "note": "yfinance 无分部数据，本次跳过"},
        ],
        "dcf": {
            "assumptions": {
                "horizon_years": 5,
                "y1_growth": y1_model,
                "growth_path": [round(g, 4) for g in growth_path],
                "ebit_margin": ebit_margin,
                "da_pct_rev": da_pct,
                "capex_pct_rev": capex_pct,
                "nwc_pct_rev": nwc_pct,
                "tax_rate": tax_rate,
                "rf": rf,
                "beta": beta,
                "erp": erp,
                "ke": ke,
                "kd": kd,
                "equity_weight": e_v,
                "wacc": wacc,
                "terminal_g": g_terminal,
                "exit_ebitda_multiple": exit_mult,
                "tv_method": "midpoint of Gordon and exit multiple",
            },
            "projection": proj,
            "bridge": {
                "pv_fcff": pv_fcff,
                "tv_gordon": tv_gordon,
                "tv_exit": tv_exit,
                "tv_base": tv_base,
                "pv_tv": pv_tv,
                "enterprise_value": ev,
                "cash": cash,
                "debt": total_debt,
                "equity_value": equity,
                "shares": shares_out,
                "implied_price": implied_dcf,
                "tv_share_of_ev": pv_tv / ev,
            },
            "sensitivity": {
                "wacc_grid": [round(w, 4) for w in wacc_grid],
                "g_grid": g_grid,
                "rows": sens_rows,
                "base_wacc": round(wacc, 4),
                "base_g": g_terminal,
            },
            "scenarios": [
                {"name": "Bull", "price": scenario(0.03, 0.02, -0.01, 0.03),
                 "levers": "增速 +3pp · 利润率 +2pp · WACC −1pp · g=3%"},
                {"name": "Base", "price": scenario(0, 0, 0, 0.025), "levers": "上表 Base 假设"},
                {"name": "Bear", "price": scenario(-0.03, -0.02, 0.01, 0.015),
                 "levers": "增速 −3pp · 利润率 −2pp · WACC +1pp · g=1.5%"},
            ],
        },
        "relative": {
            "peers": peer_rows,
            "medians": {"pe_fwd": med_pe, "ev_rev": med_evr, "ev_ebitda": med_eve},
            "premium": premium,
            "implied": {
                "from_pe": implied_pe,
                "from_ev_rev": implied_evr,
                "from_ev_ebitda": implied_eve,
                "median": implied_rel,
            },
            "self_multiples": {
                "pe_fwd": info.get("forwardPE"),
                "ev_rev": info.get("enterpriseToRevenue"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
            },
        },
        "risks": [
            "终值占 EV 比重高：AI 资本开支放缓时倍数与增速可能同时下修",
            "β≈2.2 → WACC 偏高，是 DCF 低于市价/相对估值的主因",
            "客户集中与自研 ASIC / AMD 竞争可能压缩超高利润率",
            "相对估值依赖同业倍数；行业杀估值时锚失效",
            "yfinance 非官方数据，决策前应对照 10-K/10-Q",
        ],
        "glossary": glossary,
        "dcf_explain": {
            "what": "DCF（Discounted Cash Flow，现金流折现）回答的问题是：如果公司未来每年能挣到这些现金，按你要求的回报率折回今天，值多少钱？",
            "steps": [
                "预测未来几年收入与利润（显式预测期，这里 5 年）",
                "把利润调成自由现金流 FCFF（加回折旧、减去投资）",
                "用 WACC 把每年 FCFF 折成现值并加总",
                "估算第 5 年之后的终值并折现",
                "得到企业价值 EV，加上现金、减去债务，再除以股数 → 每股公允价值",
            ],
            "intuition": "钱在今天比在五年后更值钱；风险越高（WACC 越高），同样一笔未来现金的今天出价越低。",
        },
    }

    income_rows = [
        ("Total Revenue", ["Total Revenue"]),
        ("Gross Profit", ["Gross Profit"]),
        ("Operating Income (EBIT)", ["Operating Income", "EBIT"]),
        ("Net Income", ["Net Income"]),
        ("Diluted EPS", ["Diluted EPS"]),
        ("Tax Provision", ["Tax Provision"]),
    ]
    cf_rows = [
        ("Operating Cash Flow", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
        ("Capital Expenditure", ["Capital Expenditure"]),
        ("Free Cash Flow", ["Free Cash Flow"]),
        ("Depreciation And Amortization", ["Depreciation And Amortization"]),
        ("Stock Based Compensation", ["Stock Based Compensation"]),
        ("Change In Working Capital", ["Change In Working Capital"]),
    ]
    bs_rows = [
        ("Cash And Cash Equivalents", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
        ("Total Assets", ["Total Assets"]),
        ("Total Debt", ["Total Debt", "Long Term Debt And Capital Lease Obligation"]),
        ("Stockholders Equity", ["Stockholders Equity", "Common Stock Equity"]),
        ("Current Assets", ["Current Assets"]),
        ("Current Liabilities", ["Current Liabilities"]),
    ]

    fundamentals = {
        "title": "Company Fundamentals",
        "subtitle": "利润表 · 现金流 · 资产负债表 · NVDA",
        "ticker": TICKER,
        "name": info.get("shortName") or "NVIDIA Corporation",
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "disclaimer": "数据来自 yfinance，单位多为 USD；与官方 10-K/10-Q 可能有口径差异。",
        "snapshot": valuation["snapshot"],
        "annual": {
            "income": df_table(income_a, income_rows, 4),
            "cashflow": df_table(cashflow_a, cf_rows, 4),
            "balance": df_table(balance_a, bs_rows, 4),
        },
        "quarterly": {
            "income": df_table(income_q, income_rows, 8),
            "cashflow": df_table(cashflow_q, cf_rows, 8),
        },
        "estimates": {},
        "link_valuation": "#valuation",
    }
    try:
        ee = t.earnings_estimate
        if ee is not None and not ee.empty:
            fundamentals["estimates"]["earnings"] = ee.reset_index().astype(str).to_dict(orient="records")
    except Exception:
        pass
    try:
        re = t.revenue_estimate
        if re is not None and not re.empty:
            fundamentals["estimates"]["revenue"] = re.reset_index().astype(str).to_dict(orient="records")
    except Exception:
        pass

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for name, obj in [("nvda_valuation.json", valuation), ("nvda_fundamentals.json", fundamentals)]:
            path = d / name
            with open(path, "w") as f:
                json.dump(obj, f, indent=2, allow_nan=False)
            print("wrote", path)
    print(valuation["verdict"]["headline"])


if __name__ == "__main__":
    main()
