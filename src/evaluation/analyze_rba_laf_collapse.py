"""
analyze_rba_laf_collapse.py
---------------------------
Focused diagnosis of WHY the RbA detector collapses on Lost & Found in the
safety-weighting detector pipeline (1 TP / ~30k FP, R_FP > R_TP), while kNN /
Energy / PixOOD do not.

The detector step thresholds each image at its top DET_TOP_PCT% of OoD scores and
keeps connected components. A detector "recalls" a GT object only if that object's
pixels rank in the top DET_TOP_PCT% of the image. So the question is:

   For each method, where do the GT-obstacle pixels rank inside their own image's
   score distribution?  (high percentile -> caught by top-pct threshold; low -> missed)

Per L&F image we compute, for each method, the median per-image percentile-rank of
the GT-obstacle pixels (0..100). A detector with median rank >= (100-DET_TOP_PCT)
catches the object; far below means the object is buried under in-distribution
pixels that the method scores HIGHER than the real obstacle.

We also report, for the pixels that DO exceed the top-pct threshold, what fraction
are on the road but NOT GT (the on-road false positives that get amplified by
W_safety -> high R_FP).

Output: results/rba_analysis/laf_collapse.csv + console summary.
"""

import sys
import csv
import numpy as np
from pathlib import Path
from scipy import ndimage

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR  # noqa: E402
from evaluation.safety_weighted import (  # noqa: E402
    METHODS, to_ood_score, score_map_dir, fill_road,
)

DET_TOP_PCT = 0.5                      # same as the detector pipeline (L&F)
ANALYZE = ["RbA", "DINOv2-kNN", "Energy", "PixOOD"]
OUT = RESULTS_DIR / "rba_analysis"


def pct_rank_of(values, full):
    """Percentile rank (0..100) of each `values` entry within distribution `full`."""
    order = np.sort(full)
    idx = np.searchsorted(order, values, side="right")
    return 100.0 * idx / order.size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(score_map_dir("laf").glob("*.npz"))
    print(f"[L&F] {len(files)} images; threshold = top {DET_TOP_PCT}% per image\n")

    rows = []
    agg = {m: dict(gt_rank=[], recall=[], fp_onroad=[], n_obj=0, obj_caught=0)
           for m in ANALYZE}

    for f in files:
        d = np.load(f)
        gt = d["ood_label"].astype(np.int32) == 1
        if not gt.any():
            continue
        pred = d["pred_class"].astype(np.int32)
        road = fill_road((pred == 0) | (pred == 1))
        labeled, n_obj = ndimage.label(gt)

        for m in ANALYZE:
            s = to_ood_score(m, d[METHODS[m]].astype(np.float32))
            flat = s.ravel()
            tau = np.percentile(s, 100.0 - DET_TOP_PCT)   # top-pct threshold
            fired = s >= tau

            # 1) where do GT pixels rank in this image's score distribution?
            gt_ranks = pct_rank_of(s[gt], flat)
            med_rank = float(np.median(gt_ranks))

            # 2) object-level recall: an object is "caught" if any of its pixels fire
            caught = 0
            for oid in range(1, n_obj + 1):
                if fired[labeled == oid].any():
                    caught += 1
            agg[m]["n_obj"] += n_obj
            agg[m]["obj_caught"] += caught

            # 3) of the fired pixels, fraction that are on-road but NOT GT (the FPs
            #    that W_safety amplifies)
            fired_n = int(fired.sum())
            fp_onroad = float((fired & road & ~gt).sum()) / fired_n if fired_n else 0.0

            agg[m]["gt_rank"].append(med_rank)
            agg[m]["recall"].append(caught / n_obj if n_obj else 0.0)
            agg[m]["fp_onroad"].append(fp_onroad)
            rows.append(dict(stem=f.stem, method=m, n_obj=n_obj,
                             gt_median_pctrank=round(med_rank, 2),
                             obj_recall=round(caught / n_obj, 3) if n_obj else 0.0,
                             fired_px=fired_n,
                             fp_onroad_frac=round(fp_onroad, 3)))

    with open(OUT / "laf_collapse.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Saved] {OUT/'laf_collapse.csv'}\n")

    thr = 100.0 - DET_TOP_PCT
    print(f"{'method':<13}{'GT med pctrank':>15}{'obj recall':>12}"
          f"{'pix recall':>12}{'fired FP on-road':>18}")
    print(f"{'(need >= '+str(thr)+')':<13}{'':>15}{'(object lvl)':>12}{'':>12}{'(frac of fired)':>18}")
    for m in ANALYZE:
        a = agg[m]
        gt_rank = np.mean(a["gt_rank"])
        recall_img = np.mean(a["recall"])
        recall_obj = a["obj_caught"] / a["n_obj"] if a["n_obj"] else 0.0
        fp = np.mean(a["fp_onroad"])
        print(f"{m:<13}{gt_rank:>15.2f}{recall_obj:>12.3f}{recall_img:>12.3f}{fp:>18.3f}")
    print(f"\nInterpretation: a GT object is detectable by the top-{DET_TOP_PCT}% "
          f"rule only if its pixels rank >= p{thr}. The lower the GT median "
          f"percentile-rank, the more in-distribution pixels the method scores "
          f"ABOVE the real obstacle -> object missed, and the fired pixels are "
          f"on-road FPs that W_safety amplifies into high R_FP.")


if __name__ == "__main__":
    main()
