# XGBoost 多中心 EEG 分类训练

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 路径、任务、特征集、超参数搜索空间 |
| `data_loader.py` | 读取 features/*.xlsx + demographic, subID 对齐, sex 编码 |
| `train_within_site.py` | 中心内嵌套 CV (downsample → outer 5-fold → inner 3-fold RandomizedSearch) |
| `train_cross_site.py` | 跨中心验证 (训练中心 downsample + 内层 3-fold RandomizedSearch → 测试中心全量推理) |
| `run_all.py` | 主入口, 覆盖所有特征/任务/中心/有无 demographic 组合 |

## 数据约定

- `features/*.xlsx` 工作表名 = `<site>_<group>` (HX/HQ × HC/BD/MDD/SCZ)
- `demographic.xlsx` 同样按 `<site>_<group>` 组织, 列: `subID, age, sex`
- HX 特征文件 subID 带 `v` 前缀 (`vND004`), demographic 不带 (`ND004`); 加载器会自动 strip
- 不同特征文件样本量不一致 (alpha_peak < aperiodic < PSD); 合并时按 subID 内连接
- sex: `女 -> 0`, `男 -> 1`

## 输出结构

```
results/
  within_site/
    <demo_tag>/<feature_label>/<task>/<site>/
      feature_weights.csv      # 行=repeat, 列=repeat_i + 各特征
      predictions.csv          # 列: repeat_i, fold_i, sample_idx, y_true, y_pred, y_proba
      best_params.csv          # 每折最优超参
      meta.json                # 样本量/特征名/任务信息
  cross_site/
    <demo_tag>/<feature_label>/<task>/<train>_to_<test>/
      feature_weights.csv      # 行=repeat
      predictions.csv          # 列: repeat_i, sample_idx, y_true, y_pred, y_proba
      best_params.csv
      meta.json
```

`<demo_tag>` ∈ {`no_demo`, `with_demo`}
`<feature_label>` ∈ {`PSD`, `CONN`, `PSD+CONN`, `APER`, `AP`, `APER+AP`, `BP`}
`<task>` ∈ {`BD_vs_HC`, `MDD_vs_HC`, `SCZ_vs_HC`, `BD_vs_MDD`, `BD_vs_SCZ`, `MDD_vs_SCZ`}

## 使用示例

```bash
# 1) 全量跑 (REPEAT=100, outer=5, inner=3)
python run_all.py

# 2) 试跑(改小 repeat) 验证流程
python run_all.py --repeat 3

# 3) 只跑中心内
python run_all.py --mode within

# 4) 只跑跨中心 BD_vs_HC, PSD+CONN, HX->HQ
python run_all.py --mode cross --feature_set PSD+CONN --task BD_vs_HC --direction HX_to_HQ

# 5) 续跑 (跳过已完成结果目录)
python run_all.py --skip_existing
```

## 训练逻辑

### 中心内嵌套 CV (within-site)
```
for repeat in 1..REPEAT_N:
    sel_pos = sample(idx_pos, n=min_len, replace=False)
    sel_neg = sample(idx_neg, n=min_len, replace=False)
    Xb, yb = balanced subset
    outer_5fold:
        train_fold:
            RandomizedSearchCV(XGB pipeline, inner_3fold, n_iter=20, scoring=ROC-AUC)
        test_fold:
            best_estimator.predict / predict_proba
        feature_importances_ from best_estimator
    weight[repeat] = mean over 5 folds
```

### 跨中心验证 (cross-site)
```
for repeat in 1..REPEAT_N:
    Xtr_balanced = balanced downsample(train_site)
    RandomizedSearchCV(XGB pipeline, inner_3fold, n_iter=20, scoring=ROC-AUC).fit(Xtr_balanced)
    best_estimator.predict / predict_proba on full test_site (imbalanced)
    weight[repeat] = best_estimator.feature_importances_
```

`StandardScaler` 通过 `Pipeline` 嵌入, 内层 CV 每个 fold 独立 fit / transform, 避免泄漏.

## 性能预估

| 维度 | 数量 |
| --- | --- |
| 特征集 | 7 |
| 任务 | 6 |
| 中心 (within) | 2 |
| 方向 (cross) | 2 |
| 含/不含 demo | 2 |

总组合: within = 7×6×2×2 = 168, cross = 7×6×2×2 = 168, 共 **336** 组.
每组 100 repeats × 5 outer × (20 candidates × 3 inner = 60 fits) = 30000 fit/组.
样本量大的任务(BD vs MDD/SCZ, MDD vs SCZ on HX)单组耗时数小时, 建议:

1. 先用 `--repeat 3` 走通流程;
2. 按特征集分批跑;
3. 算力允许时全量跑, 单台机器配合 `tqdm` 进度即可监控.

