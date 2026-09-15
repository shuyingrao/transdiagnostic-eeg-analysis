# -*- coding: utf-8 -*-
"""数据加载: 读取多套特征 + 人口学, 按 subID 对齐, sex编码.

注意:
  - HX(华西)的特征文件 subID 带前缀 'v' (例如 vND004),
    demographic 文件 subID 是 'ND004'(不带 'v'). 加载时统一 strip 'v'.
  - HQ subID 在所有文件中一致.
  - 不同特征文件的样本量不同 (例如 alpha_peak 缺少没找到 alpha 峰的被试),
    在做特征组合 / 加 demographic 时按 subID 交集对齐.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Sequence
from functools import lru_cache

from config import FEATURES_DIR, FEATURE_FILES, DEMO_FILE


def _strip_v(sid: str) -> str:
    if isinstance(sid, str) and sid.startswith("v"):
        return sid[1:]
    return sid


@lru_cache(maxsize=None)
def _read_sheet(file_key: str, site: str, group: str) -> pd.DataFrame:
    if file_key == "DEMO":
        path = FEATURES_DIR / DEMO_FILE
    else:
        path = FEATURES_DIR / FEATURE_FILES[file_key]
    df = pd.read_excel(path, sheet_name=f"{site}_{group}")
    df = df.copy()
    df["subID"] = df["subID"].astype(str).map(_strip_v)
    return df


def load_demographic(site: str, group: str) -> pd.DataFrame:
    df = _read_sheet("DEMO", site, group)
    df = df[["subID", "age", "sex"]].copy()
    sex_map = {"女": 0, "男": 1, "F": 0, "M": 1, "f": 0, "m": 1, 0: 0, 1: 1}
    df["sex"] = df["sex"].map(lambda x: sex_map.get(x, np.nan))
    if df["sex"].isna().any():
        unmapped = df.loc[df["sex"].isna(), "sex"].unique()
        raise ValueError(f"sex 含未识别值: {unmapped}, site={site}, group={group}")
    df["sex"] = df["sex"].astype(int)
    return df


def load_feature(file_key: str, site: str, group: str) -> pd.DataFrame:
    df = _read_sheet(file_key, site, group)
    feat_cols = [c for c in df.columns if c != "subID"]
    out = df[["subID"] + feat_cols].copy()
    out.columns = ["subID"] + [f"{file_key}__{c}" for c in feat_cols]
    return out


def load_combined(
    feature_keys: Iterable[str],
    site: str,
    group: str,
    add_demographic: bool = False,
) -> pd.DataFrame:
    feature_keys = list(feature_keys)
    if not feature_keys:
        raise ValueError("feature_keys 不能为空")
    dfs = [load_feature(k, site, group) for k in feature_keys]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="subID", how="inner")
    if add_demographic:
        demo = load_demographic(site, group)
        merged = merged.merge(demo, on="subID", how="inner")
    merged = merged.dropna(axis=0, how="any").reset_index(drop=True)
    return merged


def get_feature_matrix(
    feature_keys: Iterable[str],
    site: str,
    group: str,
    add_demographic: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    df = load_combined(feature_keys, site, group, add_demographic=add_demographic)
    feat_cols = [c for c in df.columns if c != "subID"]
    X = df[feat_cols].to_numpy(dtype=float)
    return X, feat_cols, df["subID"].tolist()


def get_feature_matrix_multi_site(
    feature_keys: Iterable[str],
    sites: Sequence[str],
    group: str,
    add_demographic: bool = False,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """从多个 site 的同一个 group 中拼接样本.

    返回 (X, feature_names, subIDs, site_tags). 不同 site 的 subID 加 site 前缀避免重名.
    """
    Xs, sids_all, sites_all = [], [], []
    feat_cols_ref = None
    for site in sites:
        Xi, feat_cols, sids = get_feature_matrix(
            feature_keys, site, group, add_demographic=add_demographic
        )
        if feat_cols_ref is None:
            feat_cols_ref = feat_cols
        elif feat_cols != feat_cols_ref:
            raise RuntimeError(
                f"site={site} 的特征列与首个 site 不一致 (feature_keys={list(feature_keys)})"
            )
        Xs.append(Xi)
        sids_all.extend([f"{site}::{s}" for s in sids])
        sites_all.extend([site] * len(sids))
    X = np.vstack(Xs)
    return X, feat_cols_ref, sids_all, sites_all


def get_two_class_data(
    feature_keys: Iterable[str],
    site: str,
    group_pos: str,
    group_neg: str,
    add_demographic: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """约定: group_pos -> y=0, group_neg -> y=1."""
    X1, feat1, sid1 = get_feature_matrix(feature_keys, site, group_pos, add_demographic)
    X2, feat2, sid2 = get_feature_matrix(feature_keys, site, group_neg, add_demographic)
    if feat1 != feat2:
        raise RuntimeError("两类的特征列不一致.")
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1), dtype=int),
                        np.ones(len(X2), dtype=int)])
    sids = list(sid1) + list(sid2)
    return X, y, feat1, sids


def get_two_class_data_multi_site(
    feature_keys: Iterable[str],
    sites: Sequence[str],
    group_pos: str,
    group_neg: str,
    add_demographic: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """合并多个 site 的样本组装二分类数据.

    返回 (X, y, feature_names, subIDs(带 site::前缀), site_tags).
    约定: group_pos -> y=0, group_neg -> y=1.
    """
    X1, feat1, sid1, ste1 = get_feature_matrix_multi_site(
        feature_keys, sites, group_pos, add_demographic=add_demographic
    )
    X2, feat2, sid2, ste2 = get_feature_matrix_multi_site(
        feature_keys, sites, group_neg, add_demographic=add_demographic
    )
    if feat1 != feat2:
        raise RuntimeError("两类的特征列不一致.")
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1), dtype=int),
                        np.ones(len(X2), dtype=int)])
    sids = list(sid1) + list(sid2)
    site_tags = list(ste1) + list(ste2)
    return X, y, feat1, sids, site_tags


if __name__ == "__main__":
    for site in ["HX", "HQ"]:
        for grp in ["HC", "BD", "MDD", "SCZ"]:
            X, _, _ = get_feature_matrix(["PSD"], site, grp)
            print(f"{site}_{grp}  PSD  X={X.shape}")
    print("--- merge HX+HQ ---")
    for grp in ["HC", "BD", "MDD", "SCZ"]:
        X, _, sids, sts = get_feature_matrix_multi_site(
            ["PSD"], ["HX", "HQ"], grp, add_demographic=False
        )
        print(f"merge_{grp}  PSD  X={X.shape}  HX={sts.count('HX')}  HQ={sts.count('HQ')}")
