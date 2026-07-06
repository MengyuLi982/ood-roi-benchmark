"""
analyze_rba_recall.py
---------------------
Object-level recall of the RbA detector at two per-image thresholds:
top-0.5% (p99.5, the collapse case) vs top-20% (p80, the recovery case).

For each image we enumerate GT OoD objects (connected components) and ask whether
each is *recalled* = intersected by a kept detection (a >=MIN_DET_SIZE connected
component of the thresholded score map) -- i.e. the same detection logic the
pipeline uses, just measured per GT object instead of per detection.

Reports per dataset:
  - #GT objects, object-level recall at each threshold
  - pixel-level recall (fraction of GT pixels that fire) for reference
  - recall split by GT object size (small/medium/large terciles)

Output: results/rba_analysis/rba_recall_by_threshold.csv + console table.
"""

import sys
import csv
import numpy as np
from pathlib import Path
from scipy import ndimage

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR  # noqa: E402
from evaluation.safety_weighted import METHODS, to_ood_score, score_map_dir  # noqa: E402
from evaluation.safety_weighted_dino import MIN_DET_SIZE  # noqa: E402

METHOD = "RbA"
THRESHOLDS = {"top0.5%": 0.5, "top20%": 20.0}   # per-image top-percent
DATASETS = ["laf", "RoadAnomaly21", "RoadObstacle21"]
OUT = RESULTS_DIR / "rba_analysis"


def recall_for_image(s, gt, top_pct):
    """Return (per-object recalled bool list, object sizes, pixel-recall)."""
    tau = np.percentile(s, 100.0 - top_pct)
    fired = s >= tau
    # kept detections = connected components of fired map with >= MIN_DET_SIZE px
    det_lab, n_det = ndimage.label(fired)
    if n_det:
        det_sizes = ndimage.sum(np.ones_like(det_lab), det_lab, range(1, n_det + 1))
        keep = {i + 1 for i, sz in enumerate(det_sizes) if sz >= MIN_DET_SIZE}
        kept = np.isin(det_lab, list(keep)) if keep else np.zeros_like(fired)
    else:
        kept = np.zeros_like(fired)

    gt_lab, n_obj = ndimage.label(gt)
    recalled, sizes = [], []
    for oid in range(1, n_obj + 1):
        obj = gt_lab == oid
        sizes.append(int(obj.sum()))
        recalled.append(bool(kept[obj].any()))
    pix_recall = float(fired[gt].mean()) if gt.any() else float("nan")
    return recalled, sizes, pix_recall


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Detector = {METHOD};  object recalled = hit by a >= {MIN_DET_SIZE}px "
          f"kept detection\n")
    header = (f"{'dataset':<15}{'#GTobj':>8}" +
              "".join(f"{('objRcl '+k):>14}" for k in THRESHOLDS) +
              "".join(f"{('pixRcl '+k):>14}" for k in THRESHOLDS))
    print(header)

    for ds in DATASETS:
        files = sorted(score_map_dir(ds).glob("*.npz"))
        # accumulate per-threshold
        obj_hit = {k: 0 for k in THRESHOLDS}
        pix_acc = {k: [] for k in THRESHOLDS}
        all_sizes, hit_by_thr = [], {k: [] for k in THRESHOLDS}
        n_obj_total = 0
        for f in files:
            d = np.load(f)
            gt = d["ood_label"].astype(np.int32) == 1
            if not gt.any():
                continue
            s = to_ood_score(METHOD, d[METHODS[METHOD]].astype(np.float32))
            per_thr_recall = {}
            for k, pct in THRESHOLDS.items():
                recalled, sizes, pr = recall_for_image(s, gt, pct)
                per_thr_recall[k] = recalled
                obj_hit[k] += sum(recalled)
                pix_acc[k].append(pr)
                hit_by_thr[k].extend(recalled)
            n_obj_total += len(sizes)
            all_sizes.extend(sizes)

        objr = {k: obj_hit[k] / n_obj_total if n_obj_total else 0.0 for k in THRESHOLDS}
        pixr = {k: float(np.nanmean(pix_acc[k])) if pix_acc[k] else 0.0 for k in THRESHOLDS}
        print(f"{ds:<15}{n_obj_total:>8}" +
              "".join(f"{objr[k]:>13.1%} " for k in THRESHOLDS) +
              "".join(f"{pixr[k]:>13.1%} " for k in THRESHOLDS))

        # size-tercile breakdown
        sizes_arr = np.array(all_sizes)
        terciles = np.quantile(sizes_arr, [1/3, 2/3]) if len(sizes_arr) else [0, 0]
        size_bins = [("small", sizes_arr <= terciles[0]),
                     ("medium", (sizes_arr > terciles[0]) & (sizes_arr <= terciles[1])),
                     ("large", sizes_arr > terciles[1])]
        row = dict(dataset=ds, n_gt_objects=n_obj_total)
        for k in THRESHOLDS:
            row[f"obj_recall_{k}"] = round(objr[k], 4)
            row[f"pix_recall_{k}"] = round(pixr[k], 4)
            hits = np.array(hit_by_thr[k])
            for name, m in size_bins:
                row[f"obj_recall_{k}_{name}"] = (
                    round(float(hits[m].mean()), 4) if m.any() else float("nan"))
        rows.append(row)

    # size-tercile console table (L&F focus)
    print("\nObject recall by GT object-size tercile:")
    print(f"{'dataset':<15}{'size':>8}" +
          "".join(f"{k:>12}" for k in THRESHOLDS))
    for r in rows:
        for name in ("small", "medium", "large"):
            print(f"{r['dataset']:<15}{name:>8}" +
                  "".join(f"{r[f'obj_recall_{k}_{name}']:>11.1%} " for k in THRESHOLDS))

    with open(OUT / "rba_recall_by_threshold.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[Saved] {OUT/'rba_recall_by_threshold.csv'}")


if __name__ == "__main__":
    main()
