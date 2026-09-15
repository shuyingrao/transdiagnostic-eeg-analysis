# -*- coding: utf-8 -*-
"""跨中心验证: 训练中心下采样平衡 + 内层CV超参搜索, 在测试中心(全部样本, 不平衡)上预测.

REPEAT_N 次重复, 每次重复:
  1) 训练中心两类各下采样 min_len 个样本组成平衡训练集;
  2) 在该平衡集上跑 RandomizedSearchCV (3折内层CV, ROC-AUC), 找最优超参 + 全集 refit;
  3) 用最优 estimator 在测试中心(全部, 不平衡)上预测;
  4) 记录该 repeat 的 feature_importances_.

输出 (CSV) 保存到 results/cross_site/<feature_label>/<task>/<train>_to_<test>/:
  - feature_weights.csv  (REPEAT_N 行, 第一列 'repeat_i'; 之后是特征列)
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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from xgboost import XGBClassifier
from tqdm import trange

from config import (
    RESULTS_DIR, REPEAT_N, INNER_K, RANDOM_SEED, N_JOBS,
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


def run_cross_site(
    feature_label: str,
    feature_keys: list[str],
    task: tuple[str, str],
    train_site: str,
    test_site: str,
    add_demographic: bool,
    repeat_n: int = REPEAT_N,
    out_root: Path | None = None,
    seed_base: int = RANDOM_SEED,
) -> Path:
    g_pos, g_neg = task
    Xtr_full, ytr_full, feat_tr, _ = get_two_class_data(
        feature_keys, train_site, g_pos, g_neg, add_demographic=add_demographic
    )
    Xte, yte, feat_te, _ = get_two_class_data(
        feature_keys, test_site, g_pos, g_neg, add_demographic=add_demographic
    )
    if feat_tr != feat_te:
        raise RuntimeError("两个中心特征列不一致.")
    feat_names = feat_tr

    n_pos_tr = int((ytr_full == 0).sum())
    n_neg_tr = int((ytr_full == 1).sum())
    min_len = min(n_pos_tr, n_neg_tr)
    if min_len < INNER_K:
        raise ValueError(f"训练样本太少: min_len={min_len} < inner_k={INNER_K}")

    demo_tag = "with_demo" if add_demographic else "no_demo"
    out_root = out_root or (RESULTS_DIR / "cross_site")
    out_dir = (out_root / demo_tag / feature_label /
               f"{g_pos}_vs_{g_neg}" / f"{train_site}_to_{test_site}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "train_site": train_site,
        "test_site": test_site,
        "task": f"{g_pos}_vs_{g_neg}",
        "feature_label": feature_label,
        "feature_keys": feature_keys,
        "add_demographic": add_demographic,
        "n_pos_train": n_pos_tr,
        "n_neg_train": n_neg_tr,
        "min_len_train": min_len,
        "n_test": int(Xte.shape[0]),
        "n_pos_test": int((yte == 0).sum()),
        "n_neg_test": int((yte == 1).sum()),
        "n_features": Xtr_full.shape[1],
        "feature_names": feat_names,
        "repeat_n": repeat_n,
        "inner_k": INNER_K,
        "n_iter_search": N_ITER_SEARCH,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    idx_pos = np.where(ytr_full == 0)[0]
    idx_neg = np.where(ytr_full == 1)[0]

    weight_rows: list[np.ndarray] = []
    pred_rows: list[dict] = []
    bestparam_rows: list[dict] = []

    rng_master = np.random.RandomState(seed_base)
    for rep in trange(repeat_n,
                      desc=f"[cross] {train_site}->{test_site} {feature_label} {g_pos}_vs_{g_neg} {demo_tag}"):
        rs = int(rng_master.randint(0, 2**31 - 1))
        sel_pos = resample(idx_pos, replace=False, n_samples=min_len, random_state=rs)
        sel_neg = resample(idx_neg, replace=False, n_samples=min_len, random_state=rs + 1)
        sel = np.concatenate([sel_pos, sel_neg])
        Xtr = Xtr_full[sel]
        ytr = ytr_full[sel]

        search = _make_search(seed=rs + 2)
        search.fit(Xtr, ytr)
        est = search.best_estimator_

        yhat = est.predict(Xte)
        yproba = est.predict_proba(Xte)[:, 1]

        for s_i, (yt, yp, yq) in enumerate(zip(yte, yhat, yproba)):
            pred_rows.append({
                "repeat_i": rep,
                "sample_idx": int(s_i),
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_proba": float(yq),
            })

        bp = {"repeat_i": rep}
        bp.update({k: v for k, v in search.best_params_.items()})
        bestparam_rows.append(bp)

        xgb_clf = est.named_steps["clf"]
        weight_rows.append(np.asarray(xgb_clf.feature_importances_, dtype=float))

    # 保存
    weights_arr = np.vstack(weight_rows)
    weights_df = pd.DataFrame(weights_arr, columns=feat_names)
    weights_df.insert(0, "repeat_i", np.arange(repeat_n))
    weights_df.to_csv(out_dir / "feature_weights.csv", index=False)

    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(bestparam_rows).to_csv(out_dir / "best_params.csv", index=False)

    print(f"[cross] saved -> {out_dir}")
    return out_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--direction", type=str, default="both",
                    choices=["HX_to_HQ", "HQ_to_HX", "both"])
    ap.add_argument("--demo", type=str, default="both",
                    choices=["with", "no", "both"])
    ap.add_argument("--repeat", type=int, default=REPEAT_N)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feat_sets = ALL_FEATURE_SETS if args.feature_set is None \
        else {args.feature_set: ALL_FEATURE_SETS[args.feature_set]}
    tasks = TASKS if args.task is None \
        else [tuple(args.task.split("_vs_"))]
    if args.direction == "both":
        directions = [("HX", "HQ"), ("HQ", "HX")]
    else:
        a, b = args.direction.split("_to_")
        directions = [(a, b)]
    demo_opts = ([True, False] if args.demo == "both"
                 else ([True] if args.demo == "with" else [False]))

    for fl, fk in feat_sets.items():
        for tk in tasks:
            for tr_site, te_site in directions:
                for d in demo_opts:
                    try:
                        run_cross_site(
                            feature_label=fl, feature_keys=fk,
                            task=tk,
                            train_site=tr_site, test_site=te_site,
                            add_demographic=d,
                            repeat_n=args.repeat,
                        )
                    except Exception as e:
                        print(f"[ERROR] cross {tr_site}->{te_site} {fl} "
                              f"{tk[0]}_vs_{tk[1]} demo={d}: {e}")
