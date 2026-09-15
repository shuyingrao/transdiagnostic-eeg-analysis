# transdiagnostic-eeg-analysis

This repository contains the implementation code for the paper titled "Large-scale EEG Reveals Shared and Distinct Neural Oscillatory Features Across Major Psychiatric Disorders". The project provides analysis pipelines for identifying shared and disorder-specific EEG oscillatory alterations across major psychiatric disorders and evaluating the predictive utility of EEG features using machine learning.

The study investigates resting-state EEG patterns in:

- Schizophrenia (SCZ)
- Bipolar disorder (BD)
- Major depressive disorder (MDD)
- Healthy controls (HC)

using large-scale, multi-site EEG datasets.

The complete workflow includes:

1. EEG feature generation
2. Statistical analysis of spectral and connectivity alterations
3. Machine learning-based disorder detection and diagnostic discrimination

---

# Citation

If you use this code, please cite:

```bibtex
@article{
  title={Large-scale EEG Reveals Shared and Distinct Neural Oscillatory Features Across Major Psychiatric Disorders},
  author={Shuying Rao, Junyi Xie,Mengfan Niu, Yue Pan, Yue Li, Hua Yu, Jingyi Xie, Yaoyun Zhang, Sha Zhao, Gang Pan, Wanjun Guo, Tao Li, Haiteng Jiang},
  journal={},
  volume={},
  number={},
  pages={},
  year={2026}
}
```

---

# Statistical Analysis

Implemented in MATLAB.

The statistical pipeline evaluates group-level EEG alterations using:

- Power spectral density (PSD)
- Functional connectivity (FC)

Main analyses:

- Cluster-based permutation tests for PSD
- Network-based statistics (NBS) for FC
- Cross-site validation

Covariates:

- Age
- Sex

The analysis follows a discovery-validation framework:

```
 Chengdu cohort: Statistical discovery
               |
               v
Hangzhou cohort: Cross-site validation
```

---

## EEG Feature Generation

Scripts:

```
└── statistic_analysis/
    ├── Calculate_PSD.m
    ├── Calculate_PSD_Interpolation.m
    ├── Calculate_connectivity.m
```

---

## PSD Statistical Analysis

Scripts:
```
└── statistic_analysis/
    ├── psd_anova_cluster_with_covariates.m
    ├── psd_hx_cluster_hq_ttest_psd_with_covariates.m
    ├── psd_cross_site_with_covariates.m
```

---

## Connectivity Statistical Analysis

Scripts:

```
└── statistic_analysis/
    ├── conn_anova_nbs_with_covariates.m
    ├── conn_hx_ttest_hq_ttest_conn_with_covariates.m
    ├── conn_cross_site_with_covariates.m
```

---

# Predictive Modeling

Implemented in Python.

The machine learning pipeline uses:

- XGBoost classifiers
- Repeated validation
- Within-site evaluation
- Cross-site generalization

Classification tasks include:

### Disorder detection

```
BD vs HC
MDD vs HC
SCZ vs HC
```

### Diagnostic discrimination

```
BD vs MDD
BD vs SCZ
MDD vs SCZ
```

---

## Main configuration:

```
predictive_modeling/config.py
```

Defined parameters include:

- Dataset locations
- Feature sets
- Classification tasks
- XGBoost hyperparameter search space
- Repetition number

---

## Feature Sets

The pipeline supports:

| Feature set | Description |
|---|---|
| PSD | Spectral power |
| CONN | Functional connectivity |
| PSD+CONN | Combined EEG features |
| APER | Aperiodic spectral parameters |
| AP | Alpha peak features |
| APER+AP | Combined spectral parameter features |
| BP | Band power features |

---

## Run only within-site validation

```bash
cd predictive_modeling

python run_all.py --mode within
```

---

## Run cross-site validation

Example:

```bash
cd predictive_modeling

python run_all.py \
--mode cross \
--feature_set PSD+CONN \
--task BD_vs_HC \
--direction HX_to_HQ
```

---

## Continue interrupted experiments

```bash
cd predictive_modeling

python run_all.py --skip_existing
```

---

