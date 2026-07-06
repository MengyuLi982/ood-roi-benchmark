"""
analyze_object_lateral_position.py
----------------------------------
Diagnostic for the right-extended polynomial road gate (see
reports/safety_weighted_polyroad_report.md).

Question: the poly gate widens M_road on the RIGHT by BETA_RIGHT*width(y) to be
conservative about objects near the ego vehicle's right edge, yet on Lost & Found
(L&F) it barely changed the results. Hypothesis: **L&F has few anomaly objects near
the road lane edge**, so the added right strip rarely contains a real object.

This script tests that, per dataset, by measuring WHERE each GT anomaly object sits
relative to the fitted road boundaries, and how many objects the right extension can
newly touch. It is read-only and method-agnostic -- it uses only `pred_class` and
`ood_label` from each cached score map (no OoD score maps, no depth).

The decisive number is `pct_in_ext_strip`: % of GT objects with >=1 pixel in the
added right margin (strip = poly-with-extension band minus poly-without-extension
band). Hypothesis predicts L&F ~ near-zero.

Usage (repo root, `ood` venv):
    ./ood/bin/python src/evaluation/analyze_object_lateral_position.py
    ./ood/bin/python src/evaluation/analyze_object_lateral_position.py --limit 50

Outputs to results/polyroad_analysis/:
    object_lateral_position_objects.csv   (one row per GT object)
    object_lateral_position_summary.csv   (one row per dataset)
    object_lateral_position.png           (2x2 summary figure)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR  # noqa: E402
from evaluation.safety_weighted import (  # noqa: E402
    score_map_dir, fill_road, road_boundaries, road_gate, ego_lane_map, BETA_RIGHT,
)

OUT = RESULTS_DIR / "polyroad_analysis"
DATASETS = ["laf", "RoadAnomaly21", "RoadObstacle21"]
COLORS = {"laf": "#c0392b", "RoadAnomaly21": "#28b463", "RoadObstacle21": "#2471a3"}
NEAR_RIGHT_U = 0.6     # |u| at/above which a centroid is "near the right edge"
DW_EPS = 0.01          # |dW| above which the extension meaningfully reweights an object


def _row_geom(b, y):
    """Boundary x-values at image row y (clamped into the fitted road row range)."""
    rows, xL, xR, xR_ext = b
    idx = int(np.clip(round(y) - rows[0], 0, len(rows) - 1))
    xl, xr, xre = float(xL[idx]), float(xR[idx]), float(xR_ext[idx])
    width = xr - xl
    return xl, xr, xre, 0.5 * (xl + xr), width, max(width, 1.0)


def analyze_image(pred, ood):
    """Return (list-of-object-dicts, strip_frac_of_road) for one image."""
    road = (pred == 0) | (pred == 1)
    filled = fill_road(road)
    poly_ext = road_gate(filled, BETA_RIGHT)     # band incl. right extension
    poly_band = road_gate(filled, 0.0)           # same band, no extension
    strip = poly_ext & ~poly_band                # the added right margin (exact)
    L = ego_lane_map(filled)
    Wspan = filled.astype(np.float32) * L
    Wpoly = poly_ext.astype(np.float32) * L
    b = road_boundaries(filled)                  # None if < 5 road rows

    road_px = int(filled.sum())
    strip_frac = (int(strip.sum()) / road_px) if road_px else 0.0

    labeled, n_obj = ndimage.label(ood == 1)
    rows_out = []
    for oid in range(1, n_obj + 1):
        cm = labeled == oid
        size = int(cm.sum())
        if size == 0:
            continue
        ys, xs = np.where(cm)
        cy, cx = float(ys.mean()), float(xs.mean())
        row_bottom = int(ys.max())

        in_strip = bool(strip[cm].any())
        n_strip = int(strip[cm].sum())
        frac_span = float(filled[cm].mean())
        frac_band = float(poly_band[cm].mean())
        frac_strip = float(strip[cm].mean())
        w_span = float(Wspan[cm].mean())
        w_poly = float(Wpoly[cm].mean())
        dW = w_poly - w_span

        if b is None:
            vert = "no_fit"
            geom = dict(road_in_range=0, xL=np.nan, xR=np.nan, xR_ext=np.nan, xc=np.nan,
                        width=np.nan, u_centroid=np.nan, u_bottom=np.nan,
                        dist_right_norm=np.nan, right_margin_px=np.nan, lat_bucket="na",
                        near_right_edge=0)
        else:
            rows = b[0]
            vert = ("above" if round(cy) < rows[0]
                    else "below" if round(cy) > rows[-1] else "in_range")
            xl, xr, xre, xc, width, weff = _row_geom(b, cy)
            _, xrb, _, xcb, _, weffb = _row_geom(b, row_bottom)
            u_c = (cx - xc) / (0.5 * weff)
            u_b = (cx - xcb) / (0.5 * weffb)
            if cx < xl:
                lat = "left_off"
            elif cx <= xr:
                lat = "on_road"
            elif cx <= xre:
                lat = "right_shoulder"
            else:
                lat = "right_off"
            near = int((u_b >= NEAR_RIGHT_U) or (lat == "right_shoulder"))
            geom = dict(road_in_range=int(vert == "in_range"),
                        xL=round(xl, 1), xR=round(xr, 1), xR_ext=round(xre, 1),
                        xc=round(xc, 1), width=round(width, 1),
                        u_centroid=round(u_c, 3), u_bottom=round(u_b, 3),
                        dist_right_norm=round((cx - xr) / weff, 3),
                        right_margin_px=round(BETA_RIGHT * width, 1),
                        lat_bucket=lat, near_right_edge=near)

        rows_out.append(dict(
            obj_id=oid, size=size, cy=round(cy, 1), cx=round(cx, 1),
            row_bottom=row_bottom, vert_bucket=vert, **geom,
            in_ext_strip=int(in_strip), n_px_in_strip=n_strip,
            frac_in_span=round(frac_span, 3), frac_in_polyband=round(frac_band, 3),
            frac_in_ext_strip=round(frac_strip, 3), frac_offroad=round(1 - frac_span, 3),
            w_span=round(w_span, 4), w_poly=round(w_poly, 4),
            dW=round(dW, 4), dW_sig=int(abs(dW) > DW_EPS)))
    return rows_out, strip_frac


def pct(mask_count, denom):
    return round(100.0 * mask_count / denom, 1) if denom else float("nan")


def summarize(ds, rows, strip_fracs):
    n = len(rows)
    inr = [r for r in rows if r["vert_bucket"] == "in_range"]
    nin = len(inr)
    sizes = np.array([r["size"] for r in rows], float)
    # size terciles (small/medium/large) within dataset
    if n >= 3:
        q1, q2 = np.quantile(sizes, [1 / 3, 2 / 3])
    else:
        q1 = q2 = np.inf
    def strip_pct_for(lo, hi):
        sub = [r for r in rows if lo <= r["size"] < hi]
        return pct(sum(r["in_ext_strip"] for r in sub), len(sub))
    u_b = np.array([r["u_bottom"] for r in inr if r["u_bottom"] == r["u_bottom"]], float)
    rm = np.array([r["right_margin_px"] for r in inr
                   if r["right_margin_px"] == r["right_margin_px"]], float)
    fstrip = np.array([r["frac_in_ext_strip"] for r in rows], float)
    return dict(
        dataset=ds, n_gt_objects=n,
        n_no_fit=sum(r["vert_bucket"] == "no_fit" for r in rows),
        n_above=sum(r["vert_bucket"] == "above" for r in rows),
        n_below=sum(r["vert_bucket"] == "below" for r in rows),
        n_in_range=nin,
        pct_on_road=pct(sum(r["lat_bucket"] == "on_road" for r in inr), nin),
        pct_right_shoulder=pct(sum(r["lat_bucket"] == "right_shoulder" for r in inr), nin),
        pct_right_off=pct(sum(r["lat_bucket"] == "right_off" for r in inr), nin),
        pct_left_off=pct(sum(r["lat_bucket"] == "left_off" for r in inr), nin),
        pct_near_right_edge=pct(sum(r["near_right_edge"] for r in inr), nin),
        pct_in_ext_strip=pct(sum(r["in_ext_strip"] for r in rows), n),
        pct_in_ext_strip_small=strip_pct_for(-1, q1),
        pct_in_ext_strip_medium=strip_pct_for(q1, q2),
        pct_in_ext_strip_large=strip_pct_for(q2, np.inf),
        median_frac_in_ext_strip=round(float(np.median(fstrip)), 4) if n else float("nan"),
        median_u_bottom=round(float(np.median(u_b)), 3) if u_b.size else float("nan"),
        median_right_margin_px=round(float(np.median(rm)), 1) if rm.size else float("nan"),
        pct_dW_sig=pct(sum(r["dW_sig"] for r in rows), n),
        mean_abs_dW=round(float(np.mean([abs(r["dW"]) for r in rows])), 4) if n else float("nan"),
        mean_strip_frac_of_road=round(float(np.mean(strip_fracs)), 4) if strip_fracs else float("nan"))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[Saved] {path}")


def make_figure(obj_rows, summary_rows, path):
    by_ds = {ds: [r for r in obj_rows if r["dataset"] == ds] for ds in DATASETS}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # TL: violin of u_bottom (in_range only), clipped to [-3,3]
    data, labels, cols = [], [], []
    for ds in DATASETS:
        u = [min(3, max(-3, r["u_bottom"])) for r in by_ds[ds]
             if r["vert_bucket"] == "in_range" and r["u_bottom"] == r["u_bottom"]]
        if u:
            data.append(u); labels.append(ds); cols.append(COLORS[ds])
    if data:
        parts = ax[0, 0].violinplot(data, showmedians=True, vert=False)
        for pc, c in zip(parts["bodies"], cols):
            pc.set_facecolor(c); pc.set_alpha(0.6)
        ax[0, 0].set_yticks(range(1, len(labels) + 1)); ax[0, 0].set_yticklabels(labels)
    for xv, ls, lab in [(0, "-", "center"), (1, "--", "right edge u=1"), (-1, "--", "left edge")]:
        ax[0, 0].axvline(xv, color="gray", ls=ls, lw=1)
    ax[0, 0].text(1.02, 0.5, "u=1\nright edge", transform=ax[0, 0].get_xaxis_transform(),
                  fontsize=8, color="gray")
    ax[0, 0].set_title("Lateral position u_bottom (0=center, ±1=road edges)")
    ax[0, 0].set_xlabel("u_bottom (clipped to [-3,3])")

    # TR: grouped bars pct_right_side / pct_near_right_edge / pct_in_ext_strip
    sm = {r["dataset"]: r for r in summary_rows}
    metrics = ["pct_near_right_edge", "pct_in_ext_strip"]
    metrics_extra = ["pct_right_shoulder", "pct_right_off"]
    keys = metrics_extra + metrics
    x = np.arange(len(keys)); w = 0.25
    for i, ds in enumerate(DATASETS):
        if ds not in sm:
            continue
        vals = [sm[ds][k] if sm[ds][k] == sm[ds][k] else 0 for k in keys]
        ax[0, 1].bar(x + (i - 1) * w, vals, w, label=ds, color=COLORS[ds])
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(["right_\nshoulder", "right_\noff", "near_\nright_edge",
                              "in_ext_\nstrip"], fontsize=9)
    ax[0, 1].set_ylabel("% of GT objects")
    ax[0, 1].set_title("Where GT objects sit (the answer at a glance)")
    ax[0, 1].legend(fontsize=8)

    # BL: object size (log-x) vs right_margin_px (log-y)
    for ds in DATASETS:
        s = [r["size"] for r in by_ds[ds] if r["right_margin_px"] == r["right_margin_px"]
             and r["right_margin_px"] > 0]
        m = [r["right_margin_px"] for r in by_ds[ds]
             if r["right_margin_px"] == r["right_margin_px"] and r["right_margin_px"] > 0]
        if s:
            ax[1, 0].scatter(s, m, s=10, alpha=0.4, color=COLORS[ds], label=ds, edgecolors="none")
    ax[1, 0].axhline(1.0, color="red", ls=":", lw=1, label="1 px margin")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("GT object size (px, log)")
    ax[1, 0].set_ylabel("right_margin_px = BETA*width (log)")
    ax[1, 0].set_title("Right margin available at each object's row")
    ax[1, 0].legend(fontsize=8)

    # BR: ECDF of frac_in_ext_strip
    for ds in DATASETS:
        v = np.sort([r["frac_in_ext_strip"] for r in by_ds[ds]])
        if v.size:
            ax[1, 1].plot(v, np.arange(1, v.size + 1) / v.size, color=COLORS[ds],
                          lw=2, label=f"{ds} (n={v.size})")
    ax[1, 1].set_xlabel("frac of object pixels in extension strip")
    ax[1, 1].set_ylabel("ECDF")
    ax[1, 1].set_title("How much of each object the strip covers")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("GT anomaly object position vs the right-extended road gate", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap images per dataset (debug)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    obj_rows, summary_rows = [], []
    for ds in DATASETS:
        files = sorted(score_map_dir(ds).glob("*.npz"))
        if args.limit:
            files = files[:args.limit]
        if not files:
            print(f"[{ds}] no score maps -- skipped.")
            continue
        ds_rows, strip_fracs = [], []
        for n, f in enumerate(files):
            d = np.load(f)
            pred = d["pred_class"].astype(np.int32)
            ood = d["ood_label"].astype(np.int32)
            if not (ood == 1).any():
                continue
            rs, sf = analyze_image(pred, ood)
            strip_fracs.append(sf)
            for r in rs:
                r_full = dict(dataset=ds, stem=f.stem, **r)
                ds_rows.append(r_full)
            if (n + 1) % 200 == 0:
                print(f"[{ds}] ... {n + 1}/{len(files)}")
        obj_rows.extend(ds_rows)
        summary_rows.append(summarize(ds, ds_rows, strip_fracs))
        print(f"[{ds}] {len(ds_rows)} GT objects from {len(files)} images")

    write_csv(OUT / "object_lateral_position_objects.csv", obj_rows)
    write_csv(OUT / "object_lateral_position_summary.csv", summary_rows)
    make_figure(obj_rows, summary_rows, OUT / "object_lateral_position.png")

    print("\n=== Headline: where do GT anomaly objects sit? ===")
    print(f"{'dataset':<16}{'#obj':>6}{'%near_right':>12}{'%in_strip':>11}"
          f"{'med_u_bot':>11}{'med_margin_px':>15}{'%dW_sig':>9}")
    for r in summary_rows:
        print(f"{r['dataset']:<16}{r['n_gt_objects']:>6}{r['pct_near_right_edge']:>12}"
              f"{r['pct_in_ext_strip']:>11}{r['median_u_bottom']:>11}"
              f"{r['median_right_margin_px']:>15}{r['pct_dW_sig']:>9}")


if __name__ == "__main__":
    main()
