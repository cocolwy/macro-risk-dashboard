"""
Week4 Step 2 — Walk-forward LambdaRank 训练 (无随机 CV)。

窗口: 训练 24 个月 / 测试 3 个月 / 每 3 个月重新训练。
每折内部:
  1. 在训练窗口上拟合 标准化 + PCA(5)  → PC1..PC5 (无前视)
  2. 用同一 scaler/PCA 变换测试窗口
  3. LightGBM lambdarank: 标签 = 每月 fwd_ret 的十分位 (0..9), group = 每月股票数
  4. 同时算"诚实"线性基线: 训练窗口末 6 个月的 PC rank-IC 均值 → 归一化权重 → 测试月打分
输出:
  oos_scores.parquet   (date, instrument) × [lgbm_score, linear_score, fwd_ret]
  feature_importance.csv   各 PC 的平均增益
  folds.csv                每折训练/测试区间
"""
from pathlib import Path
import ctypes, os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# 预载 sklearn 自带的 arm64 libomp, 保证 lightgbm 能加载 (本机无 homebrew)
import sklearn
ctypes.CDLL(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs", "libomp.dylib"),
            mode=ctypes.RTLD_GLOBAL)
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "qlib_migration" / "lgbm_factor_panel.parquet"
FACTORS = ["momentum", "bp", "log_mcap", "low_vol", "reversal"]
PCS = [f"PC{i+1}" for i in range(5)]
TRAIN_M, TEST_M, STEP_M = 24, 3, 3
N_GRADES = 10          # fwd_ret 十分位作为 LambdaRank 相关度
IC_WINDOW = 6          # 线性基线: 末 6 月 IC 均值 (对齐 baseline)

LGB_PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[20],
    num_leaves=15, min_data_in_leaf=30, learning_rate=0.05,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    lambda_l2=1.0, verbosity=-1, num_threads=4,
)
NUM_ROUNDS = 120


def fit_pca(train_df):
    mean, std = train_df.mean(), train_df.std().replace(0, 1)
    z = (train_df - mean) / std
    pca = PCA(n_components=5).fit(z.fillna(0))
    return mean, std, pca


def transform_pca(df, mean, std, pca):
    z = ((df - mean) / std).fillna(0)
    return pd.DataFrame(pca.transform(z), index=df.index, columns=PCS)


def linear_weights(pc_panel, fwd, train_months):
    """末 IC_WINDOW 个训练月的 PC rank-IC 均值 → 绝对值归一化(保号)权重。"""
    last = train_months[-IC_WINDOW:]
    ics = {pc: [] for pc in PCS}
    for m in last:
        x = pc_panel.loc[m]
        y = fwd.loc[m].reindex(x.index)
        common = x.dropna().index.intersection(y.dropna().index)
        if len(common) < 10:
            continue
        for pc in PCS:
            ics[pc].append(x.loc[common, pc].corr(y.loc[common], method="spearman"))
    w = {pc: np.nanmean(v) if v else 0.0 for pc, v in ics.items()}
    tot = sum(abs(x) for x in w.values()) or 1.0
    return {pc: w[pc] / tot for pc in PCS}


def main():
    panel = pd.read_parquet(PANEL)
    months = sorted(panel.index.get_level_values(0).unique())
    fwd = panel["fwd_ret"]

    oos_rows, fold_rows, imp_acc = [], [], []
    i = TRAIN_M
    while i + TEST_M <= len(months) + TEST_M and i < len(months):
        train_months = months[i - TRAIN_M:i]
        test_months = months[i:i + TEST_M]
        if not test_months:
            break

        tr = panel.loc[panel.index.get_level_values(0).isin(train_months)]
        te = panel.loc[panel.index.get_level_values(0).isin(test_months)]

        # ── per-fold PCA (无前视) ──
        mean, std, pca = fit_pca(tr[FACTORS])
        pc_tr = transform_pca(tr[FACTORS], mean, std, pca)
        pc_te = transform_pca(te[FACTORS], mean, std, pca)

        # ── LightGBM 训练数据: 每月十分位标签 + group ──
        Xs, ys, groups = [], [], []
        for m in train_months:
            if m not in pc_tr.index.get_level_values(0):
                continue
            xm = pc_tr.loc[m]
            ym = fwd.loc[m].reindex(xm.index)
            valid = ym.dropna().index
            if len(valid) < 20:
                continue
            xm = xm.loc[valid]; ym = ym.loc[valid]
            grade = pd.qcut(ym.rank(method="first"), N_GRADES, labels=False)
            Xs.append(xm.values); ys.append(grade.values); groups.append(len(valid))
        if not Xs:
            i += STEP_M; continue
        X = np.vstack(Xs); y = np.concatenate(ys)
        dtrain = lgb.Dataset(X, label=y, group=groups, feature_name=PCS)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)
        imp_acc.append(model.feature_importance(importance_type="gain"))

        # ── 线性基线权重 (同折特征) ──
        lw = linear_weights(pc_tr, fwd, train_months)

        # ── 测试月打分 (两种模型) ──
        for m in test_months:
            if m not in pc_te.index.get_level_values(0):
                continue
            xm = pc_te.loc[m]
            lgbm_s = pd.Series(model.predict(xm.values), index=xm.index)
            zm = (xm - xm.mean()) / xm.std().replace(0, 1)   # 线性用每月 z-score
            lin_s = sum(lw[pc] * zm[pc] for pc in PCS)
            ym = fwd.loc[m].reindex(xm.index) if m in fwd.index.get_level_values(0) else pd.Series(np.nan, index=xm.index)
            for inst in xm.index:
                oos_rows.append((m, inst, float(lgbm_s[inst]), float(lin_s[inst]), float(ym[inst])))

        fold_rows.append((train_months[0], train_months[-1], test_months[0], test_months[-1], len(groups)))
        i += STEP_M

    oos = pd.DataFrame(oos_rows, columns=["date", "instrument", "lgbm_score", "linear_score", "fwd_ret"]).set_index(["date", "instrument"])
    oos.to_parquet(ROOT / "qlib_migration" / "oos_scores.parquet")

    imp = pd.Series(np.mean(imp_acc, axis=0), index=PCS, name="avg_gain")
    imp.to_csv(ROOT / "qlib_migration" / "feature_importance.csv")

    folds = pd.DataFrame(fold_rows, columns=["train_start", "train_end", "test_start", "test_end", "train_months"])
    folds.to_csv(ROOT / "qlib_migration" / "folds.csv", index=False)

    print(f"[done] {len(folds)} folds | OOS rows={len(oos)} | "
          f"OOS span={oos.index.get_level_values(0).min().date()}~{oos.index.get_level_values(0).max().date()}")
    print("feature importance (avg gain):")
    print(imp.round(1).to_string())


if __name__ == "__main__":
    main()
