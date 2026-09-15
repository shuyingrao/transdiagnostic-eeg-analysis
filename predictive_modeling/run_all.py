# -*- coding: utf-8 -*-
"""主入口: 一次跑完所有 (特征集, 任务, 中心组合, 加/不加 demographic).

模式:
  - within : 中心内 (单次切分 + 训练集 CV 超参搜索)
  - cross  : 跨中心 (训练中心下采样平衡 + 内层 CV 超参搜索, 测试中心全量)
  - merge  : HX + HQ 合并后, 与 within 同样的策略 (单次切分)
  - both   : within + cross (不含 merge, 兼容历史)
  - all    : within + cross + merge

使用示例:
    python run_all.py                          # 默认 mode=all
    python run_all.py --mode merge --feature_set PSD --task BD_vs_HC
    python run_all.py --mode within --repeat 10
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from config import ALL_FEATURE_SETS, TASKS, SITES, REPEAT_N, TEST_SIZE, RESULTS_DIR
from train_within_site import run_within_site
from train_cross_site import run_cross_site
from train_merge_site import run_merge_site


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="all",
                    choices=["within", "cross", "merge", "both", "all"])
    ap.add_argument("--feature_set", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--site", type=str, default=None,
                    help="within 模式下的单 site (HX/HQ); 默认全跑")
    ap.add_argument("--direction", type=str, default="both",
                    choices=["HX_to_HQ", "HQ_to_HX", "both"])
    ap.add_argument("--merge_sites", type=str, default=",".join(SITES),
                    help="merge 模式合并的 site 列表 (默认 HX,HQ)")
    ap.add_argument("--demo", type=str, default="both",
                    choices=["with", "no", "both"])
    ap.add_argument("--repeat", type=int, default=REPEAT_N)
    ap.add_argument("--test_size", type=float, default=TEST_SIZE)
    ap.add_argument("--skip_existing", action="store_true",
                    help="若结果目录的 feature_weights.csv 已存在则跳过")
    return ap.parse_args()


def _need_run(out_dir: Path, skip_existing: bool) -> bool:
    if not skip_existing:
        return True
    return not (out_dir / "feature_weights.csv").exists()


def _run_within(args, feat_sets, tasks, sites, demo_opts):
    n_done = n_skip = n_err = 0
    for fl, fk in feat_sets.items():
        for tk in tasks:
            for st in sites:
                for d in demo_opts:
                    demo_tag = "with_demo" if d else "no_demo"
                    out_dir = (RESULTS_DIR / "within_site" / demo_tag /
                               fl / f"{tk[0]}_vs_{tk[1]}" / st)
                    if not _need_run(out_dir, args.skip_existing):
                        print(f"[skip] {out_dir}"); n_skip += 1; continue
                    try:
                        run_within_site(
                            feature_label=fl, feature_keys=fk,
                            task=tk, site=st, add_demographic=d,
                            repeat_n=args.repeat, test_size=args.test_size,
                        )
                        n_done += 1
                    except Exception as e:
                        print(f"[ERROR within] {st} {fl} {tk[0]}_vs_{tk[1]} demo={d}: {e}")
                        n_err += 1
    return n_done, n_skip, n_err


def _run_cross(args, feat_sets, tasks, directions, demo_opts):
    n_done = n_skip = n_err = 0
    for fl, fk in feat_sets.items():
        for tk in tasks:
            for tr_site, te_site in directions:
                for d in demo_opts:
                    demo_tag = "with_demo" if d else "no_demo"
                    out_dir = (RESULTS_DIR / "cross_site" / demo_tag /
                               fl / f"{tk[0]}_vs_{tk[1]}" /
                               f"{tr_site}_to_{te_site}")
                    if not _need_run(out_dir, args.skip_existing):
                        print(f"[skip] {out_dir}"); n_skip += 1; continue
                    try:
                        run_cross_site(
                            feature_label=fl, feature_keys=fk,
                            task=tk, train_site=tr_site, test_site=te_site,
                            add_demographic=d, repeat_n=args.repeat,
                        )
                        n_done += 1
                    except Exception as e:
                        print(f"[ERROR cross] {tr_site}->{te_site} {fl} "
                              f"{tk[0]}_vs_{tk[1]} demo={d}: {e}")
                        n_err += 1
    return n_done, n_skip, n_err


def _run_merge(args, feat_sets, tasks, merge_sites, demo_opts):
    n_done = n_skip = n_err = 0
    sites_tag = "+".join(merge_sites)
    for fl, fk in feat_sets.items():
        for tk in tasks:
            for d in demo_opts:
                demo_tag = "with_demo" if d else "no_demo"
                out_dir = (RESULTS_DIR / "merge_site" / demo_tag /
                           fl / f"{tk[0]}_vs_{tk[1]}" / sites_tag)
                if not _need_run(out_dir, args.skip_existing):
                    print(f"[skip] {out_dir}"); n_skip += 1; continue
                try:
                    run_merge_site(
                        feature_label=fl, feature_keys=fk,
                        task=tk, sites=merge_sites, add_demographic=d,
                        repeat_n=args.repeat, test_size=args.test_size,
                    )
                    n_done += 1
                except Exception as e:
                    print(f"[ERROR merge] {sites_tag} {fl} "
                          f"{tk[0]}_vs_{tk[1]} demo={d}: {e}")
                    n_err += 1
    return n_done, n_skip, n_err


def main():
    args = parse_args()
    feat_sets = ALL_FEATURE_SETS if args.feature_set is None \
        else {args.feature_set: ALL_FEATURE_SETS[args.feature_set]}
    tasks = TASKS if args.task is None else [tuple(args.task.split("_vs_"))]
    sites = SITES if args.site is None else [args.site]
    if args.direction == "both":
        directions = [("HX", "HQ"), ("HQ", "HX")]
    else:
        a, b = args.direction.split("_to_")
        directions = [(a, b)]
    merge_sites = [s.strip() for s in args.merge_sites.split(",") if s.strip()]
    demo_opts = ([True, False] if args.demo == "both"
                 else ([True] if args.demo == "with" else [False]))

    t0 = time.time()
    tot_done = tot_skip = tot_err = 0

    if args.mode in ("within", "both", "all"):
        d, s, e = _run_within(args, feat_sets, tasks, sites, demo_opts)
        tot_done += d; tot_skip += s; tot_err += e
    if args.mode in ("cross", "both", "all"):
        d, s, e = _run_cross(args, feat_sets, tasks, directions, demo_opts)
        tot_done += d; tot_skip += s; tot_err += e
    if args.mode in ("merge", "all"):
        d, s, e = _run_merge(args, feat_sets, tasks, merge_sites, demo_opts)
        tot_done += d; tot_skip += s; tot_err += e

    elapsed = time.time() - t0
    print(f"\n=== 完成 ===  done={tot_done}  skip={tot_skip}  error={tot_err}  "
          f"耗时={elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
