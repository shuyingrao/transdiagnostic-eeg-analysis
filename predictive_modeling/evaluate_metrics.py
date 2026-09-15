# -*- coding: utf-8 -*-
"""遍历 results/ 下所有 predictions.csv, 逐 repeat 计算分类性能, 汇总输出 summary.

指标定义 (二分类, y=1 视为阳性):
  - AUC          : roc_auc_score(y_true, y_proba)
  - ACC          : within_site / merge_site -> accuracy_score (测试集已平衡)
                   cross_site                -> balanced_accuracy_score (测试集不平衡)
  - Precision    : tp / (tp + fp)
  - F1           : 2 * P * R / (P + R)  (按 y=1 计算)
  - Sensitivity  : tp / (tp + fn)
  - Specificity  : tn / (tn + fp)

聚合规则: 每个 repeat 的所有测试样本算一次 -> n_repeat 个指标 -> mean ± std
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, f1_score, recall_score, confusion_matrix,
)

from config import RESULTS_DIR

METRIC_COLS = ["AUC", "ACC", "Precision", "F1", "Sensitivity", "Specificity"]


def _safe_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def _spec(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def _compute_metrics(y_true, y_pred, y_proba, balanced_acc=False) -> dict:
    acc = (float(balanced_accuracy_score(y_true, y_pred)) if balanced_acc
           else float(accuracy_score(y_true, y_pred)))
    return {
        "AUC":         _safe_auc(y_true, y_proba),
        "ACC":         acc,
        "Precision":   float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1":          float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Specificity": _spec(y_true, y_pred),
    }


def _per_repeat(df, balanced_acc=False) -> pd.DataFrame:
    rows = []
    for rep, sub in df.groupby("repeat_i", sort=True):
        m = _compute_metrics(
            sub["y_true"].to_numpy(int),
            sub["y_pred"].to_numpy(int),
            sub["y_proba"].to_numpy(float),
            balanced_acc=balanced_acc,
        )
        m["repeat_i"] = int(rep)
        m["n_samples"] = int(len(sub))
        rows.append(m)
    out = pd.DataFrame(rows)
    return out[["repeat_i", "n_samples"] + METRIC_COLS]


def _summary_row(per_rep) -> dict:
    out = {}
    for m in METRIC_COLS:
        v = per_rep[m].to_numpy(dtype=float)
        out[f"{m}_mean"] = float(np.nanmean(v)) if len(v) else float("nan")
        out[f"{m}_std"]  = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
    out["n_repeat"] = int(per_rep["repeat_i"].nunique())
    return out


def _load_meta(d):
    p = d / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _evaluate_dir(results_dir: Path, mode: str, leaf_label: str,
                  balanced_acc: bool, meta_extras: list[str]):
    """通用遍历器.

    Args:
      mode: 'within_site' / 'cross_site' / 'merge_site' (子目录名)
      leaf_label: predictions.csv 所在叶子目录的语义列名 (site / direction / sites)
      balanced_acc: 是否对 ACC 用 balanced accuracy
      meta_extras: 从 meta.json 取的额外字段
    """
    root = results_dir / mode
    per_rep_rows, summary_rows = [], []
    if not root.exists():
        return pd.DataFrame(), pd.DataFrame()
    for pred_path in sorted(root.rglob("predictions.csv")):
        d = pred_path.parent
        try:
            leaf = d.name
            task = d.parent.name
            feat = d.parent.parent.name
            demo = d.parent.parent.parent.name
        except Exception:
            leaf = task = feat = demo = "?"
        meta = _load_meta(d)
        df = pd.read_csv(pred_path)
        if df.empty:
            continue
        per_rep = _per_repeat(df, balanced_acc=balanced_acc)
        per_rep.insert(0, leaf_label, leaf)
        per_rep.insert(0, "task", task)
        per_rep.insert(0, "feature", feat)
        per_rep.insert(0, "demo", demo)
        per_rep_rows.append(per_rep)

        srow = {"demo": demo, "feature": feat, "task": task, leaf_label: leaf}
        for k in meta_extras:
            srow[k] = meta.get(k)
        srow.update(_summary_row(per_rep))
        summary_rows.append(srow)
    per_rep_all = pd.concat(per_rep_rows, ignore_index=True) if per_rep_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["demo", "feature", "task", leaf_label]).reset_index(drop=True)
    return per_rep_all, summary


def evaluate_within_site(results_dir):
    return _evaluate_dir(
        results_dir, "within_site", "site", balanced_acc=False,
        meta_extras=["n_pos", "n_neg", "min_len", "n_features"],
    )


def evaluate_cross_site(results_dir):
    return _evaluate_dir(
        results_dir, "cross_site", "direction", balanced_acc=True,
        meta_extras=["n_pos_train", "n_neg_train",
                     "n_pos_test", "n_neg_test", "n_features"],
    )


def evaluate_merge_site(results_dir):
    """merge: 测试集为合并下采样平衡后的 stratified split, ACC 为 accuracy."""
    return _evaluate_dir(
        results_dir, "merge_site", "sites", balanced_acc=False,
        meta_extras=["n_pos", "n_neg", "min_len", "n_features"],
    )


def _format_summary_for_paper(summary, group_cols, extra_cols):
    if summary.empty:
        return summary
    out = summary[group_cols + extra_cols].copy()
    out["n_repeat"] = summary["n_repeat"]
    for m in METRIC_COLS:
        mean = summary[f"{m}_mean"]
        std  = summary[f"{m}_std"]
        out[m] = [f"{mu:.4f}±{sd:.4f}" if pd.notna(mu) and pd.notna(sd)
                  else (f"{mu:.4f}" if pd.notna(mu) else "")
                  for mu, sd in zip(mean, std)]
    return out


def main(results_dir: Optional[Path] = None):
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    out_dir = results_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rep_w, summary_w = evaluate_within_site(results_dir)
    per_rep_c, summary_c = evaluate_cross_site(results_dir)
    per_rep_m, summary_m = evaluate_merge_site(results_dir)

    # 长表
    if not per_rep_w.empty:
        per_rep_w.to_csv(out_dir / "per_repeat_within.csv", index=False)
        print(f"[saved] {out_dir/'per_repeat_within.csv'}  rows={len(per_rep_w)}")
    if not per_rep_c.empty:
        per_rep_c.to_csv(out_dir / "per_repeat_cross.csv", index=False)
        print(f"[saved] {out_dir/'per_repeat_cross.csv'}  rows={len(per_rep_c)}")
    if not per_rep_m.empty:
        per_rep_m.to_csv(out_dir / "per_repeat_merge.csv", index=False)
        print(f"[saved] {out_dir/'per_repeat_merge.csv'}  rows={len(per_rep_m)}")

    # 数值 summary
    if not summary_w.empty:
        summary_w.to_csv(out_dir / "summary_within.csv", index=False)
        print(f"[saved] {out_dir/'summary_within.csv'}  rows={len(summary_w)}")
    if not summary_c.empty:
        summary_c.to_csv(out_dir / "summary_cross.csv", index=False)
        print(f"[saved] {out_dir/'summary_cross.csv'}  rows={len(summary_c)}")
    if not summary_m.empty:
        summary_m.to_csv(out_dir / "summary_merge.csv", index=False)
        print(f"[saved] {out_dir/'summary_merge.csv'}  rows={len(summary_m)}")

    # 论文风格 mean±std
    paper_w = _format_summary_for_paper(
        summary_w, ["demo", "feature", "task", "site"],
        ["n_pos", "n_neg", "min_len", "n_features"],
    )
    paper_c = _format_summary_for_paper(
        summary_c, ["demo", "feature", "task", "direction"],
        ["n_pos_train", "n_neg_train", "n_pos_test", "n_neg_test", "n_features"],
    )
    paper_m = _format_summary_for_paper(
        summary_m, ["demo", "feature", "task", "sites"],
        ["n_pos", "n_neg", "min_len", "n_features"],
    )

    if not paper_w.empty:
        paper_w.to_csv(out_dir / "summary_within_paper.csv", index=False)
    if not paper_c.empty:
        paper_c.to_csv(out_dir / "summary_cross_paper.csv", index=False)
    if not paper_m.empty:
        paper_m.to_csv(out_dir / "summary_merge_paper.csv", index=False)

    xlsx_path = out_dir / "summary_all.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        if not summary_w.empty:
            summary_w.to_excel(xw, sheet_name="within_numeric", index=False)
            paper_w.to_excel(xw, sheet_name="within_mean_std", index=False)
        if not summary_c.empty:
            summary_c.to_excel(xw, sheet_name="cross_numeric_balACC", index=False)
            paper_c.to_excel(xw, sheet_name="cross_mean_std_balACC", index=False)
        if not summary_m.empty:
            summary_m.to_excel(xw, sheet_name="merge_numeric", index=False)
            paper_m.to_excel(xw, sheet_name="merge_mean_std", index=False)
    print(f"[saved] {xlsx_path}")
    print("[note] cross_site 的 ACC 列为 balanced accuracy; "
          "within / merge 的 ACC 为 accuracy (测试集已平衡).")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.results_dir) if args.results_dir else None)
