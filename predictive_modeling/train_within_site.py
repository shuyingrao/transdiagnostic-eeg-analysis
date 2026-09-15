# -*- coding: utf-8 -*-
"""中心内验证: 一次按比例切割 train/test, 在 train 上 CV 搜索超参, test 评估.

REPEAT_N 次重复, 每次重复:
  1) 从两类各下采样 min_len 个样本得到平衡数据集;
  2) StratifiedShuffleSplit(test_size=TEST_SIZE) 得到 train/test (一次性切割);
  3) 在 train 上跑 RandomizedSearchCV (内层 INNER_K 折) 找最优超参 + 全集 refit;
  4) 用 best_estimator 在 test 上预测 -> y_true / y_pred / y_proba;
  5) 记录该 repeat 的 feature_importances_ 与最优超参.

输出 (CSV) 保存到 results/within_site/<demo>/<feature>/<task>/<site>/:
  - feature_weights.csv  (REPEAT_N 行: repeat_i + 各特征列, 单次模型的重要度)
  - predictions.csv       (列: repeat_i, sample_idx, y_true, y_pred, y_proba)
  - best_params.csv       (列: repeat_i, <param列>)
  - meta.json
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from xgboost import XGBClassifier
from tqdm import trange

from config import (
    RESULTS_DIR, REPEAT_N, TEST_SIZE, INNER_K, RANDOM_SEED, N_JOBS,
    N_ITER_SEARCH, PARAM_DIST, ALL_FEATURE_SETS, TASKS, SITES,
)
from data_loader import get_two_class_data

warnings.filterwarnings("ignore")


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_SEED,
            verbosity=0,
        )),
    ])


def _make_search(seed: int) -> RandomizedSearchCV:
    """构造 RandomizedSearchCV (训练集上的内层 CV 超参搜索)."""
    pipe = _build_pipeline()
    param_dist = {f"clf__{k}": v for k, v in PARAM_DIST.items()}
    inner_cv = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=seed)
    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=N_JOBS,
        random_state=seed,
        refit=True,
        verbose=0,
    )


def run_within_site(
    feature_label: str,
    feature_keys: list[str],
    task: tuple[str, str],
    site: str,
    add_demographic: bool,
    repeat_n: int = REPEAT_N,
    test_size: float = TEST_SIZE,
    out_root: Path | None = None,
    seed_base: int = RANDOM_SEED,
) -> Path:
    g_pos, g_neg = task
    X, y, feat_names, sids = get_two_class_data(
        feature_keys, site, g_pos, g_neg, add_demographic=add_demographic
    )
    n_pos = int((y == 0).sum())
    n_neg = int((y == 1).sum())
    min_len = min(n_pos, n_neg)
    n_train_per_class = int(round(min_len * (1 - test_size)))
    n_test_per_class = min_len - n_train_per_class
    if n_train_per_class < INNER_K:
        raise ValueError(
            f"训练集样本太少: {n_train_per_class}/class < INNER_K={INNER_K}; "
            f"min_len={min_len}, test_size={test_size}"
        )
    if n_test_per_class < 1:
        raise ValueError(
            f"测试集样本太少: {n_test_per_class}/class; min_len={min_len}, "
            f"test_size={test_size}"
        )

    demo_tag = "with_demo" if add_demographic else "no_demo"
    out_root = out_root or (RESULTS_DIR / "within_site")
    out_dir = out_root / demo_tag / feature_label / f"{g_pos}_vs_{g_neg}" / site
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "site": site,
        "task": f"{g_pos}_vs_{g_neg}",
        "feature_label": feature_label,
        "feature_keys": feature_keys,
        "add_demographic": add_demographic,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "min_len": min_len,
        "test_size": test_size,
        "n_train_per_class": n_train_per_class,
        "n_test_per_class": n_test_per_class,
        "n_features": X.shape[1],
        "feature_names": feat_names,
        "repeat_n": repeat_n,
        "inner_k": INNER_K,
        "n_iter_search": N_ITER_SEARCH,
        "split_strategy": "single_stratified_split + train_CV_hp_search",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    idx_pos = np.where(y == 0)[0]
    idx_neg = np.where(y == 1)[0]

    weight_rows: list[np.ndarray] = []
    pred_rows: list[dict] = []
    bestparam_rows: list[dict] = []

    rng_master = np.random.RandomState(seed_base)
    for rep in trange(
        repeat_n,
        desc=f"[within] {site} {feature_label} {g_pos}_vs_{g_neg} {demo_tag}",
    ):
        rs = int(rng_master.randint(0, 2**31 - 1))
        sel_pos = resample(idx_pos, replace=False, n_samples=min_len, random_state=rs)
        sel_neg = resample(idx_neg, replace=False, n_samples=min_len, random_state=rs + 1)
        sel = np.concatenate([sel_pos, sel_neg])
        Xb = X[sel]
        yb = y[sel]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=rs + 2,
        )
        tr_idx, te_idx = next(sss.split(Xb, yb))
        Xtr, ytr = Xb[tr_idx], yb[tr_idx]
        Xte, yte = Xb[te_idx], yb[te_idx]

        search = _make_search(seed=rs + 3)
        search.fit(Xtr, ytr)
        est = search.best_estimator_

        yhat = est.predict(Xte)
        yproba = est.predict_proba(Xte)[:, 1]

        for s_i, (yt, yp, yq) in enumerate(zip(yte, yhat, yproba)):
            pred_rows.append({
                "repeat_i": rep,
                "sample_idx": int(te_idx[s_i]),
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_proba": float(yq),
            })
        bp = {"repeat_i": rep}
        bp.update({k: v for k, v in search.best_params_.items()})
        bestparam_rows.append(bp)

        xgb_clf = est.named_steps["clf"]
        weight_rows.append(np.asarray(xgb_clf.feature_importances_, dtype=float))

    weights_arr = np.vstack(weight_rows)
    weights_df = pd.DataFrame(weights_arr, columns=feat_names)
    weights_df.insert(0, "repeat_i", np.arange(repeat_n))
    weights_df.to_csv(out_dir / "feature_weights.csv", index=False)

    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(bestparam_rows).to_csv(out_dir / "best_params.csv", index=False)

    print(f"[within] saved -> {out_dir}")
    return out_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--site", type=str, default=None)
    ap.add_argument("--demo", type=str, default="both",
                    choices=["with", "no", "both"])
    ap.add_argument("--repeat", type=int, default=REPEAT_N)
    ap.add_argument("--test_size", type=float, default=TEST_SIZE)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feat_sets = ALL_FEATURE_SETS if args.feature_set is None \
        else {args.feature_set: ALL_FEATURE_SETS[args.feature_set]}
    tasks = TASKS if args.task is None \
        else [tuple(args.task.split("_vs_"))]
    sites = SITES if args.site is None else [args.site]
    demo_opts = ([True, False] if args.demo == "both"
                 else ([True] if args.demo == "with" else [False]))

    for fl, fk in feat_sets.items():
        for tk in tasks:
            for st in sites:
                for d in demo_opts:
                    try:
                        run_within_site(
                            feature_label=fl, feature_keys=fk,
                            task=tk, site=st, add_demographic=d,
                            repeat_n=args.repeat, test_size=args.test_size,
                        )
                    except Exception as e:
                        print(f"[ERROR] within {st} {fl} {tk[0]}_vs_{tk[1]} demo={d}: {e}")
