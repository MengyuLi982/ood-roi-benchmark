"""
safety_weighted_dino.py
-----------------------
Extra experiment: an OoD method as a first-step DETECTOR, then safety-weighting.

Default detector is DINOv2-kNN (--method DINOv2-kNN), but any cached method works
(--method RbA, Energy, PixOOD, MSP). Pipeline, with NO ground-truth objects:

  1. Detector  : threshold the cached OoD score map (oriented so higher = more OoD)
                 at a per-dataset threshold (SMIYC: single global tau at >=95% pixel
                 TPR; L&F: per-image top DET_TOP_PCT% percentile, because 95%-TPR
                 floods L&F's tiny obstacles to a near-floor tau). GT is used ONLY to
                 calibrate the SMIYC tau and to flag TP/FP -- detections come purely
                 from the score map.
  2. Detections: connected components of the thresholded map (>= MIN_DET_SIZE px).
  3. Weighting : distribute W_safety = M_road*(0.7*L + 0.3*D) over each detection
                 -> per-detection risk R_i = p_i * w_i.
  4. Ranking   : rank detections by R_i (safety-aware) vs p_i (raw OoD).

Output folder depends on the method: DINOv2-kNN -> results/safety_weighted_dino/,
otherwise results/safety_weighted_<method>/  (e.g. RbA -> safety_weighted_rba/).

Usage (repo root, `ood` venv):
    ./ood/bin/python src/evaluation/safety_weighted_dino.py --method DINOv2-kNN --dataset all
    ./ood/bin/python src/evaluation/safety_weighted_dino.py --method RbA --dataset all
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR  # noqa: E402
from evaluation.safety_weighted import (  # noqa: E402
    METHODS, to_ood_score, norm01, fill_road, build_gate, ego_lane_map, distance_map,
    score_map_dir, W_LANE, W_DIST,
)

TPR_TARGET    = 0.95
MIN_DET_SIZE  = 50            # drop detections smaller than this (px)
TP_OVERLAP    = 0.10         # detection counts as TP if >=10% of its pixels are GT OoD
HIST_BINS     = 4000

# Detection-threshold strategy per dataset (see module docstring).
THRESH_TPR  = {"RoadAnomaly21", "RoadObstacle21"}   # global 95%-TPR
DET_TOP_PCT = 0.5                                    # L&F: top 0.5% of pixels/image


def out_dir_for(method: str, tag: str = "") -> Path:
    name = "safety_weighted_dino" if method == "DINOv2-kNN" \
        else f"safety_weighted_{method.lower().replace('-', '').replace('v2', '')}"
    if tag:
        name = f"{name}_{tag}"
    return RESULTS_DIR / name


def oriented(d, method) -> np.ndarray:
    """Cached map oriented so HIGHER = more OoD (handles MSP flip)."""
    return to_ood_score(method, d[METHODS[method]].astype(np.float32))


def global_threshold(files, method) -> float:
    """tau = highest oriented-score value still achieving >=TPR_TARGET pixel TPR."""
    sample = np.concatenate([oriented(np.load(f), method).ravel()[::97]
                             for f in files[:min(50, len(files))]])
    lo, hi = float(sample.min()), float(sample.max() + 1e-6)
    edges = np.linspace(lo, hi, HIST_BINS + 1)
    hist_pos = np.zeros(HIST_BINS, np.int64)
    n_pos = 0
    for f in files:
        d = np.load(f)
        s = oriented(d, method).ravel()
        y = d["ood_label"].astype(np.int32).ravel()
        pos = s[y == 1]
        if pos.size:
            hist_pos += np.histogram(np.clip(pos, lo, hi), bins=edges)[0]
            n_pos += pos.size
    if n_pos == 0:
        return float(lo)
    tpr = np.cumsum(hist_pos[::-1])[::-1] / n_pos
    valid = np.where(tpr >= TPR_TARGET)[0]
    t_idx = int(valid.max()) if valid.size else 0
    return float(edges[t_idx])


def resolve_threshold(dataset, files, method, det_top_pct=DET_TOP_PCT,
                      perimage_all=False):
    """('global', tau_float) for TPR datasets, else ('perimage', top_pct).

    perimage_all forces the per-image top-`det_top_pct`% strategy on every dataset
    (used by the per-image top-20% experiment)."""
    if not perimage_all and dataset in THRESH_TPR:
        return ("global", global_threshold(files, method))
    return ("perimage", det_top_pct)


def tau_for_image(mode, s):
    """s = oriented score array (or its values)."""
    kind, val = mode
    return val if kind == "global" else float(np.percentile(s, 100.0 - val))


def eval_dataset(dataset, method, use_depth, limit, det_rows, summary_rows,
                 det_top_pct=DET_TOP_PCT, perimage_all=False, gate_mode="span"):
    files = sorted(score_map_dir(dataset).glob("*.npz"))
    if limit:
        files = files[:limit]
    if not files:
        print(f"[{dataset}] no score maps -- skipped.")
        return

    mode = resolve_threshold(dataset, files, method, det_top_pct, perimage_all)
    if mode[0] == "global":
        print(f"[{dataset}] {method}: global threshold tau={mode[1]:.4f} (>= {TPR_TARGET:.0%} TPR)")
    else:
        print(f"[{dataset}] {method}: per-image threshold = top {mode[1]}% (p{100 - mode[1]:.1f})")

    n_det = n_tp = n_fp = 0
    R_on, R_off, R_tp, R_fp = [], [], [], []
    n_multi = n_flip = 0
    taus = []

    for f in files:
        d = np.load(f)
        stem = f.stem
        s = oriented(d, method)
        pred = d["pred_class"].astype(np.int32)
        gt = d["ood_label"].astype(np.int32)
        H, W = gt.shape

        filled = fill_road((pred == 0) | (pred == 1))
        Mroad = build_gate(filled, gate_mode)     # span (orig) or poly right-ext band
        L = ego_lane_map(filled)                  # lane center from ORIGINAL road
        if use_depth:
            D = distance_map(dataset, stem, (H, W))
            Wsafe = Mroad * (W_LANE * L + W_DIST * (D if D is not None else 0.0))
        else:
            Wsafe = Mroad * L
        Wsafe = Wsafe.astype(np.float32)
        p = norm01(s)

        tau = tau_for_image(mode, s)
        taus.append(tau)
        labeled, n = ndimage.label(s >= tau)
        comps = []
        for did in range(1, n + 1):
            cm = labeled == did
            size = int(cm.sum())
            if size < MIN_DET_SIZE:
                continue
            ys, xs = np.where(cm)
            overlap = float((gt[cm] == 1).mean())
            is_tp = overlap >= TP_OVERLAP
            p_i = float(p[cm].mean())
            w_i = float(Wsafe[cm].mean())
            onroad = float(Mroad[cm].mean())     # on-road test against the active gate
            comps.append(dict(det_id=did, size=size, cx=int(xs.mean()), cy=int(ys.mean()),
                              on_road_pct=round(100 * onroad, 1),
                              gt_overlap=round(overlap, 3), is_TP=int(is_tp),
                              p_ood=round(p_i, 4), w_safety=round(w_i, 4),
                              R=round(p_i * w_i, 4)))
        if not comps:
            continue
        by_p = sorted(range(len(comps)), key=lambda i: -comps[i]["p_ood"])
        by_R = sorted(range(len(comps)), key=lambda i: -comps[i]["R"])
        rank_p = {i: r + 1 for r, i in enumerate(by_p)}
        rank_R = {i: r + 1 for r, i in enumerate(by_R)}
        for i, c in enumerate(comps):
            c.update(dataset=dataset, stem=stem,
                     rank_by_p=rank_p[i], rank_by_R=rank_R[i])
            det_rows.append(c)
            n_det += 1
            (R_tp if c["is_TP"] else R_fp).append(c["R"])
            (R_on if c["on_road_pct"] >= 50 else R_off).append(c["R"])
            n_tp += c["is_TP"]
            n_fp += (1 - c["is_TP"])
        if len(comps) >= 2:
            n_multi += 1
            n_flip += int(by_p[0] != by_R[0])

    mean = lambda a: round(float(np.mean(a)), 4) if a else float("nan")
    tau_label = (round(mode[1], 4) if mode[0] == "global"
                 else f"p{100 - mode[1]:.1f}(~{np.mean(taus):.3f})")
    summary_rows.append(dict(
        dataset=dataset, method=method, tau=tau_label, n_images=len(files),
        n_detections=n_det, n_TP=n_tp, n_FP=n_fp,
        meanR_on_road=mean(R_on), meanR_off_road=mean(R_off),
        meanR_TP=mean(R_tp), meanR_FP=mean(R_fp),
        n_multi_det=n_multi, n_top1_changed=n_flip,
        pct_top1_changed=round(100 * n_flip / n_multi, 1) if n_multi else 0.0))


def write_csv(path, rows, fields=None):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[Saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="DINOv2-kNN", choices=list(METHODS))
    ap.add_argument("--dataset", default="all",
                    choices=["all", "laf", "RoadAnomaly21", "RoadObstacle21"])
    ap.add_argument("--no-depth", dest="use_depth", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det_top_pct", type=float, default=DET_TOP_PCT,
                    help="per-image top-percent threshold (default 0.5)")
    ap.add_argument("--perimage_all", action="store_true",
                    help="force per-image top-pct threshold on ALL datasets (incl. SMIYC)")
    ap.add_argument("--out_tag", default="",
                    help="suffix for the output folder, e.g. top20")
    ap.add_argument("--road_gate", choices=["span", "poly"], default="span",
                    help="M_road gate: 'span' (original) or 'poly' (right-extended "
                         "polynomial band); 'poly' adds a 'polyroad' folder suffix")
    args = ap.parse_args()

    datasets = ["laf", "RoadAnomaly21", "RoadObstacle21"] if args.dataset == "all" \
        else [args.dataset]
    # 'poly' -> separate folder (combined with any --out_tag) so originals are untouched
    tag = "_".join(t for t in [args.out_tag,
                               "polyroad" if args.road_gate == "poly" else ""] if t)
    out = out_dir_for(args.method, tag)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Detector] method={args.method}  road_gate={args.road_gate}  ->  {out}")
    if args.perimage_all:
        print(f"[Detector] per-image top {args.det_top_pct}% threshold on ALL datasets")

    det_rows, summary_rows = [], []
    for ds in datasets:
        eval_dataset(ds, args.method, args.use_depth, args.limit, det_rows,
                     summary_rows, args.det_top_pct, args.perimage_all, args.road_gate)

    det_fields = ["dataset", "method", "stem", "det_id", "size", "cx", "cy",
                  "on_road_pct", "gt_overlap", "is_TP", "p_ood", "w_safety", "R",
                  "rank_by_p", "rank_by_R"]
    write_csv(out / "detection_ranking.csv", det_rows, det_fields)
    write_csv(out / "detection_summary.csv", summary_rows)

    print(f"\n=== Detector summary [{args.method}] (mean R: lower = less safety-relevant) ===")
    print(f"{'dataset':<15}{'tau':>16}{'#det':>7}{'#TP':>6}{'#FP':>6}"
          f"{'R_on':>8}{'R_off':>8}{'R_TP':>8}{'R_FP':>8}{'%flip':>7}")
    for r in summary_rows:
        print(f"{r['dataset']:<15}{str(r['tau']):>16}{r['n_detections']:>7}{r['n_TP']:>6}"
              f"{r['n_FP']:>6}{r['meanR_on_road']:>8}{r['meanR_off_road']:>8}"
              f"{r['meanR_TP']:>8}{r['meanR_FP']:>8}{r['pct_top1_changed']:>7}")


if __name__ == "__main__":
    main()
