# -*- coding: utf-8 -*-
"""全局配置: 路径/任务/特征集/超参数搜索空间."""
from pathlib import Path
import scipy.stats as st

ROOT = Path(__file__).resolve().parent
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SITES = ["HX", "HQ"]
GROUPS = ["HC", "BD", "MDD", "SCZ"]
TASKS = [
    ("BD", "HC"), ("MDD", "HC"), ("SCZ", "HC"),
    ("BD", "MDD"), ("BD", "SCZ"), ("MDD", "SCZ"),
]

FEATURE_FILES = {
    "PSD":  "PSD_normpowspctrm.xlsx",
    "CONN": "powcorr_ortho_connectivity.xlsx",
    "APER": "aperiodic.xlsx",
    "AP":   "alpha_peak.xlsx",
    "BP":   "bandpower_normpow.xlsx",
}
DEMO_FILE = "demographic.xlsx"

FEATURE_SETS_GROUP1 = {
    "PSD":      ["PSD"],
    "CONN":     ["CONN"],
    "PSD+CONN": ["PSD", "CONN"],
}
FEATURE_SETS_GROUP2 = {
    "APER":    ["APER"],
    "AP":      ["AP"],
    "APER+AP": ["APER", "AP"],
    "BP":      ["BP"],
}
ALL_FEATURE_SETS = {**FEATURE_SETS_GROUP1, **FEATURE_SETS_GROUP2}

REPEAT_N = 100
TEST_SIZE = 0.2
INNER_K = 3
RANDOM_SEED = 42
N_JOBS = -1
N_ITER_SEARCH = 20

PARAM_DIST = {
    "n_estimators":      [200, 300, 500, 800],
    "max_depth":         [3, 4, 5, 6],
    "learning_rate":     st.loguniform(0.01, 0.2),
    "subsample":         st.uniform(0.6, 0.4),
    "colsample_bytree":  st.uniform(0.5, 0.5),
    "min_child_weight":  [1, 3, 5, 7],
    "gamma":             st.uniform(0.0, 0.5),
    "reg_alpha":         st.loguniform(1e-3, 1.0),
    "reg_lambda":        st.loguniform(1e-2, 5.0),
}
