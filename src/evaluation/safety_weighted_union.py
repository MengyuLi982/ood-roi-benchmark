"""
safety_weighted_union.py
------------------------
Variant experiment: the **UNION** road gate.

The polynomial poly gate (safety_weighted.py `road_gate`) is a PURE band between the
fitted left/right edges. We found it can sit *inside* the jagged hole-filled span mask
in places, so switching span->poly removes some real road coverage even as the right
extension adds margin -- the two partly cancel (see reports/polyroad_object_position_report.md).

The UNION gate removes that self-cancellation:

    M_road(union) = road_gate(filled)  OR  filled
                  = (right-extended poly band)  ∪  (hole-filled span road)

It is a strict superset of BOTH the span and the poly gate: it never loses real road,
it only ADDS the right margin. So vs the original span gate it is "span + right strip" --
the strictly-more-conservative version of the right extension.

This file does NOT modify any existing script. It reuses the detector pipeline in
safety_weighted_dino.py unchanged, locally overriding the `build_gate` symbol that
pipeline looks up so a new gate mode 'union' is available. Results go to separate
`results/safety_weighted_<method>_union/` folders (the "union symbol"), parallel to the
poly results in `results/safety_weighted_<method>_polyroad/`.

Usage (repo root, `ood` venv):
    ./ood/bin/python src/evaluation/safety_weighted_union.py --method all --dataset all
    ./ood/bin/python src/evaluation/safety_weighted_union.py --method DINOv2-kNN --dataset laf

After running, it prints and saves a span-vs-poly-vs-union comparison
(results/gate_comparison_union/gate_comparison.csv).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR  # noqa: E402
from evaluation import safety_weighted_dino as swd  # noqa: E402
from evaluation.safety_weighted import (  # noqa: E402
    fill_road, road_gate, build_gate as _orig_build_gate,
)

MAIN_METHODS = ["DINOv2-kNN", "Energy", "PixOOD", "RbA"]


def union_gate(filled: np.ndarray) -> np.ndarray:
    """UNION of the right-extended poly band and the hole-filled span road."""
    return road_gate(filled) | filled


def build_gate_union(filled: np.ndarray, mode: str = "span") -> np.ndarray:
    """Adds a 'union' mode; delegates 'span'/'poly' to the original build_gate."""
    if mode == "union":
        return union_gate(filled)
    return _orig_build_gate(filled, mode)


# The detector pipeline (swd.eval_dataset) looks up `build_gate` in its own module
# globals at call time, so this override is all that is needed to drive it with the
# union gate -- no edits to safety_weighted_dino.py.
swd.build_gate = build_gate_union

DET_FIELDS = ["dataset", "method", "stem", "det_id", "size", "cx", "cy",
              "on_road_pct", "gt_overlap", "is_TP", "p_ood", "w_safety", "R",
              "rank_by_p", "rank_by_R"]


def run_method(method, datasets, use_depth, limit, det_top_pct, perimage_all):
    out = swd.out_dir_for(method, "union")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Union] method={method}  ->  {out}")
    det_rows, summary_rows = [], []
    for ds in datasets:
        swd.eval_dataset(ds, method, use_depth, limit, det_rows, summary_rows,
                         det_top_pct, perimage_all, gate_mode="union")
    swd.write_csv(out / "detection_ranking.csv", det_rows, DET_FIELDS)
    swd.write_csv(out / "detection_summary.csv", summary_rows)
    return summary_rows


def load_summary(folder):
    """method/dataset -> row dict from an existing detection_summary.csv, or {}."""
    p = RESULTS_DIR / folder / "detection_summary.csv"
    if not p.exists():
        return {}
    return {(r["dataset"]): r for r in csv.DictReader(open(p))}


def gate_folder(method, tag):
    name = swd.out_dir_for(method, tag).name
    return name


def write_comparison(method_summaries):
    """span vs poly vs union per (method,dataset), saved with the union marker."""
    out = RESULTS_DIR / "gate_comparison_union"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, union_rows in method_summaries.items():
        span = load_summary(gate_folder(method, ""))
        poly = load_summary(gate_folder(method, "polyroad"))
        union = {r["dataset"]: r for r in union_rows}
        for ds in ["laf", "RoadAnomaly21", "RoadObstacle21"]:
            u = union.get(ds)
            if u is None:
                continue
            s, p = span.get(ds, {}), poly.get(ds, {})
            def g(d, k):
                v = d.get(k, "")
                return v if v != "" else "na"
            rows.append(dict(
                method=method, dataset=ds, n_det=u["n_detections"],
                n_TP=u["n_TP"], n_FP=u["n_FP"],
                R_TP_span=g(s, "meanR_TP"), R_TP_poly=g(p, "meanR_TP"), R_TP_union=u["meanR_TP"],
                R_FP_span=g(s, "meanR_FP"), R_FP_poly=g(p, "meanR_FP"), R_FP_union=u["meanR_FP"],
                R_on_union=u["meanR_on_road"], R_off_union=u["meanR_off_road"],
                pct_flip_union=u["pct_top1_changed"]))
    p = out / "gate_comparison.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[Saved] {p}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all",
                    help="one of DINOv2-kNN/Energy/PixOOD/RbA/MSP, or 'all'")
    ap.add_argument("--dataset", default="all",
                    choices=["all", "laf", "RoadAnomaly21", "RoadObstacle21"])
    ap.add_argument("--no-depth", dest="use_depth", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det_top_pct", type=float, default=swd.DET_TOP_PCT)
    ap.add_argument("--perimage_all", action="store_true")
    args = ap.parse_args()

    methods = MAIN_METHODS if args.method == "all" else [args.method]
    datasets = ["laf", "RoadAnomaly21", "RoadObstacle21"] if args.dataset == "all" \
        else [args.dataset]

    method_summaries = {}
    for m in methods:
        method_summaries[m] = run_method(m, datasets, args.use_depth, args.limit,
                                         args.det_top_pct, args.perimage_all)

    rows = write_comparison(method_summaries)

    print("\n=== Gate comparison  R_TP (span -> poly -> UNION)  |  R_FP (span -> poly -> UNION) ===")
    print(f"{'method':<12}{'dataset':<15}{'#TP':>5}{'#FP':>6}"
          f"{'  R_TP s/p/u':>22}{'  R_FP s/p/u':>22}")
    for r in rows:
        rtp = f"{r['R_TP_span']}/{r['R_TP_poly']}/{r['R_TP_union']}"
        rfp = f"{r['R_FP_span']}/{r['R_FP_poly']}/{r['R_FP_union']}"
        print(f"{r['method']:<12}{r['dataset']:<15}{r['n_TP']:>5}{r['n_FP']:>6}"
              f"{rtp:>22}{rfp:>22}")


if __name__ == "__main__":
    main()
