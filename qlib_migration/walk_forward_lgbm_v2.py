"""
Week4 v2 Step 2 — Walk-forward LightGBM 回归 + 可选 LambdaRank 对比。

相比 v1:
  1. 主模型用 regression (预测 raw forward return), 不是 lambdarank
  2. 同时跑一个 lambdarank 对照组 (同一宽 universe 数据)
  3. 支持 --raw-features 跳过 PCA, 让树直接用原始 5 因子
  4. 更宽 universe → 每折训练样本 ~459×24 ≈ 11k vs 107×24 ≈ 2.6k

窗口: 训练 24 个月 / 测试 3 个月 / 每 3 个月重新训练。

输出:
  oos_scores_v2.parquet   (date, instrument) × [reg_score, rank_score, linear_score, fwd_ret]
  feature_importance_v2.csv
  folds_v2.csv
"""
from pathlib import Path
import argparse
import ctypes, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import sklearn
ctypes.CDLL(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs", "libomp.dylib"),
            mode=ctypes.RTLD_GLOBAL)
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
PANEL_V2 = ROOT / "qlib_migration" / "lgbm_factor_panel_v2.parquet"
PANEL_V1 = ROOT / "qlib_migration" / "lgbm_factor_panel.parquet"
FACTORS = ["momentum", "bp", "log_mcap", "low_vol", "reversal"]
TRAIN_M, TEST_M, STEP_M = 24, 3, 3
N_GRADES = 10
IC_WINDOW = 6

REG_PARAMS = dict(
    objective="regression", metric="l2",
    num_leaves=31, min_data_in_leaf=20, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbosity=-1, num_threads=4,
)
RANK_PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[20],
    num_leaves=15, min_data_in_leaf=30, learning_rate=0.05,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    lambda_l2=1.0, verbosity=-1, num_threads=4,
)
NUM_ROUNDS = 200


def fit_pca(train_df, n_comp=5):
    mean, std = train_df.mean(), train_df.std().replace(0, 1)
    z = (train_df - mean) / std
    pca = PCA(n_components=n_comp).fit(z.fillna(0))
    return mean, std, pca


def transform_pca(df, mean, std, pca, col_names):
    z = ((df - mean) / std).fillna(0)
    return pd.DataFrame(pca.transform(z), index=df.index, columns=col_names)


def linear_weights(feature_panel, fwd, train_months, feature_names, ic_window=IC_WINDOW):
    last = train_months[-ic_window:]
    ics = {f: [] for f in feature_names}
    for m in last:
        if m not in feature_panel.index.get_level_values(0):
            continue
        x = feature_panel.loc[m]
        y = fwd.loc[m].reindex(x.index) if m in fwd.index.get_level_values(0) else None
        if y is None:
            continue
        common = x.dropna().index.intersection(y.dropna().index)
        if len(common) < 10:
            continue
        for f in feature_names:
            ics[f].append(x.loc[common, f].corr(y.loc[common], method="spearman"))
    w = {f: np.nanmean(v) if v else 0.0 for f, v in ics.items()}
    tot = sum(abs(x) for x in w.values()) or 1.0
    return {f: w[f] / tot for f in feature_names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-features", action="store_true",
                        help="Use raw 5 factors instead of PCA")
    parser.add_argument("--panel", default=None,
                        help="Path to factor panel parquet (default: auto-detect v2→v1)")
    args = parser.parse_args()

    panel_path = Path(args.panel) if args.panel else (PANEL_V2 if PANEL_V2.exists() else PANEL_V1)
    use_pca = not args.raw_features
    feature_names = [f"PC{i+1}" for i in range(5)] if use_pca else FACTORS
    tag = "pca" if use_pca else "raw"

    print(f"[v2] panel={panel_path.name}  features={tag}  "
          f"objectives=regression+lambdarank")
    panel = pd.read_parquet(panel_path)
    months = sorted(panel.index.get_level_values(0).unique())
    fwd = panel["fwd_ret"]
    n_inst = panel.index.get_level_values(1).nunique()
    print(f"  {len(months)} months, {n_inst} instruments, {len(panel)} rows")

    oos_rows, fold_rows, imp_acc = [], [], []
    i = TRAIN_M
    fold_num = 0
    while i < len(months):
        train_months = months[i - TRAIN_M:i]
        test_months = months[i:i + TEST_M]
        if not test_months:
            break
        fold_num += 1

        tr = panel.loc[panel.index.get_level_values(0).isin(train_months)]
        te = panel.loc[panel.index.get_level_values(0).isin(test_months)]

        # -- Feature transform --
        if use_pca:
            mean, std, pca = fit_pca(tr[FACTORS])
            feat_tr = transform_pca(tr[FACTORS], mean, std, pca, feature_names)
            feat_te = transform_pca(te[FACTORS], mean, std, pca, feature_names)
        else:
            feat_tr = tr[FACTORS].copy()
            feat_te = te[FACTORS].copy()

        # -- Build training data --
        Xs, ys_reg, ys_rank, groups = [], [], [], []
        for m in train_months:
            if m not in feat_tr.index.get_level_values(0):
                continue
            xm = feat_tr.loc[m]
            ym = fwd.loc[m].reindex(xm.index)
            valid = ym.dropna().index.intersection(xm.dropna(how="any").index)
            if len(valid) < 20:
                continue
            xm = xm.loc[valid]
            ym = ym.loc[valid]
            grade = pd.qcut(ym.rank(method="first"), N_GRADES, labels=False)
            Xs.append(xm.values)
            ys_reg.append(ym.values)
            ys_rank.append(grade.values)
            groups.append(len(valid))

        if not Xs:
            i += STEP_M
            continue

        X = np.vstack(Xs)
        y_reg = np.concatenate(ys_reg)
        y_rank = np.concatenate(ys_rank)

        # -- Regression model --
        dtrain_reg = lgb.Dataset(X, label=y_reg, feature_name=feature_names)
        model_reg = lgb.train(REG_PARAMS, dtrain_reg, num_boost_round=NUM_ROUNDS)

        # -- LambdaRank model (same data, ranking objective) --
        dtrain_rank = lgb.Dataset(X, label=y_rank, group=groups, feature_name=feature_names)
        model_rank = lgb.train(RANK_PARAMS, dtrain_rank, num_boost_round=NUM_ROUNDS)

        imp_acc.append(model_reg.feature_importance(importance_type="gain"))

        # -- Linear baseline weights --
        lw = linear_weights(feat_tr, fwd, train_months, feature_names)

        # -- Score test months --
        for m in test_months:
            if m not in feat_te.index.get_level_values(0):
                continue
            xm = feat_te.loc[m]
            reg_s = pd.Series(model_reg.predict(xm.fillna(0).values), index=xm.index)
            rank_s = pd.Series(model_rank.predict(xm.fillna(0).values), index=xm.index)
            zm = (xm - xm.mean()) / xm.std().replace(0, 1)
            lin_s = sum(lw[f] * zm[f] for f in feature_names)
            ym = (fwd.loc[m].reindex(xm.index)
                  if m in fwd.index.get_level_values(0)
                  else pd.Series(np.nan, index=xm.index))
            for inst in xm.index:
                oos_rows.append((m, inst,
                                 float(reg_s[inst]), float(rank_s[inst]),
                                 float(lin_s[inst]), float(ym[inst])))

        fold_rows.append((train_months[0], train_months[-1],
                          test_months[0], test_months[-1], len(groups)))
        if fold_num % 5 == 0:
            print(f"  fold {fold_num}: train {train_months[0].date()}~{train_months[-1].date()} "
                  f"→ test {test_months[0].date()}~{test_months[-1].date()} "
                  f"({sum(groups)} samples)")
        i += STEP_M

    suffix = f"_v2_{tag}"
    oos = (pd.DataFrame(oos_rows,
                        columns=["date", "instrument",
                                 "reg_score", "rank_score", "linear_score", "fwd_ret"])
           .set_index(["date", "instrument"]))
    oos_path = ROOT / "qlib_migration" / f"oos_scores_v2.parquet"
    oos.to_parquet(oos_path)

    imp = pd.Series(np.mean(imp_acc, axis=0), index=feature_names, name="avg_gain")
    imp.to_csv(ROOT / "qlib_migration" / "feature_importance_v2.csv")

    folds = pd.DataFrame(fold_rows,
                         columns=["train_start", "train_end",
                                  "test_start", "test_end", "train_months"])
    folds.to_csv(ROOT / "qlib_migration" / "folds_v2.csv", index=False)

    print(f"\n[done] {len(folds)} folds | OOS rows={len(oos)} | "
          f"OOS span={oos.index.get_level_values(0).min().date()}"
          f"~{oos.index.get_level_values(0).max().date()}")
    print(f"  instruments in OOS: {oos.index.get_level_values(1).nunique()}")
    print("feature importance (avg gain, regression model):")
    print(imp.round(1).to_string())


if __name__ == "__main__":
    main()
