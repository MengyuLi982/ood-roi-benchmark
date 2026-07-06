"""
visualize_rba_collapse.py
-------------------------
Two figures explaining why RbA collapses on Lost & Found in the detector pipeline.

  Fig 1  laf_rank_distributions.png
         Per-method distribution (box + strip) of the GT-obstacle pixels' median
         per-image percentile-rank, with the p99.5 top-0.5% detector threshold
         drawn in. RbA sits far below the line -> obstacles never reach the tail.

  Fig 2  laf_rba_vs_knn_example.png
         One real L&F image (RbA misses, kNN catches): RGB + GT outline, then for
         RbA and kNN the oriented score map (GT contoured) and the top-0.5% fired
         pixels. Shows RbA firing on on-road clutter while the obstacle stays dark.

Usage (repo root, `ood` venv):
    ./ood/bin/python src/visualization/visualize_rba_collapse.py
    ./ood/bin/python src/visualization/visualize_rba_collapse.py --stem <laf_stem>
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR  # noqa: E402
from evaluation.safety_weighted import (  # noqa: E402
    METHODS, to_ood_score, norm01, score_map_dir,
)
from scoring.compute_depth_maps import image_root, find_rgb  # noqa: E402

OUT = RESULTS_DIR / "rba_analysis"
DET_TOP_PCT = 0.5
ORDER = ["RbA", "Energy", "DINOv2-kNN", "PixOOD"]
COLORS = {"RbA": "#c0392b", "Energy": "#e67e22",
          "DINOv2-kNN": "#2471a3", "PixOOD": "#27ae60"}
DEFAULT_STEM = "02_Hanns_Klemm_Str_44_000015_000200_leftImg8bit"


def fig_rank_distributions():
    csv_path = OUT / "laf_collapse.csv"
    by_method = {m: [] for m in ORDER}
    for r in csv.DictReader(open(csv_path)):
        if r["method"] in by_method:
            by_method[r["method"]].append(float(r["gt_median_pctrank"]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [by_method[m] for m in ORDER]
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.45)
    rng = np.random.RandomState(0)
    for i, m in enumerate(ORDER):
        y = (i + 1) + (rng.rand(len(by_method[m])) - 0.5) * 0.28
        ax.scatter(by_method[m], y, s=6, color=COLORS[m], alpha=0.30,
                   edgecolors="none", zorder=3)
    thr = 100.0 - DET_TOP_PCT
    ax.axvline(thr, color="red", ls="--", lw=2,
               label=f"detector threshold (top {DET_TOP_PCT}% = p{thr})")
    ax.set_yticks(range(1, len(ORDER) + 1))
    ax.set_yticklabels(ORDER)
    ax.set_xlabel("Median per-image percentile-rank of GT-obstacle pixels")
    ax.set_xlim(0, 101)
    ax.set_title("Lost & Found: where each method ranks the real obstacle\n"
                 "(right of the red line = obstacle reaches the top-0.5% tail "
                 "-> detectable)")
    for i, m in enumerate(ORDER):
        med = np.median(by_method[m])
        ax.text(med, i + 1.34, f"median p{med:.0f}", ha="center",
                fontsize=9, color=COLORS[m], weight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = OUT / "laf_rank_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}")


def fig_example(stem):
    f = score_map_dir("laf") / f"{stem}.npz"
    if not f.exists():
        print(f"[Error] no score map for {stem}"); return
    d = np.load(f)
    gt = d["ood_label"].astype(np.int32) == 1
    H, W = gt.shape
    rgb_path = find_rgb(image_root("laf"), stem)
    rgb = (np.array(Image.open(rgb_path).convert("RGB").resize((W, H)))
           if rgb_path else np.zeros((H, W, 3), np.uint8))

    methods = ["RbA", "DINOv2-kNN"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # row spanning: top-left = RGB+GT, rest per method
    axes[0, 0].imshow(rgb)
    axes[0, 0].contour(gt, colors="cyan", linewidths=1.5)
    axes[0, 0].set_title("RGB + GT obstacle (cyan)", fontsize=12)
    axes[1, 0].axis("off")
    # small legend text in the empty cell
    axes[1, 0].text(0.5, 0.5,
                    f"Lost & Found\n{stem.replace('_leftImg8bit','')}\n\n"
                    f"GT obstacle = {int(gt.sum())} px\n"
                    f"detector keeps top {DET_TOP_PCT}% of\n"
                    f"each map (red below).",
                    ha="center", va="center", fontsize=12)

    for col, m in enumerate(methods, start=1):
        s = to_ood_score(m, d[METHODS[m]].astype(np.float32))
        p = norm01(s)
        tau = np.percentile(s, 100.0 - DET_TOP_PCT)
        fired = s >= tau
        ranks = 100.0 * np.searchsorted(np.sort(s.ravel()), s[gt], side="right") / s.size
        med = float(np.median(ranks))
        recall = float(fired[gt].any())

        ax_s = axes[0, col]
        ax_s.imshow(p, cmap="inferno", vmin=0, vmax=1)
        ax_s.contour(gt, colors="cyan", linewidths=1.5)
        ax_s.set_title(f"{m}: OoD score (GT cyan)\nGT median rank = p{med:.0f}",
                       fontsize=12)
        ax_s.axis("off")

        ax_f = axes[1, col]
        ax_f.imshow(rgb)
        overlay = np.zeros((H, W, 4))
        overlay[fired] = (1, 0, 0, 0.55)
        ax_f.imshow(overlay)
        ax_f.contour(gt, colors="cyan", linewidths=1.5)
        hit = "HIT" if recall else "MISS"
        ax_f.set_title(f"{m}: fired top-{DET_TOP_PCT}% (red) -> obstacle {hit}",
                       fontsize=12)
        ax_f.axis("off")

    axes[0, 0].axis("off")
    fig.suptitle("Why RbA misses the obstacle while kNN catches it "
                 "(top-0.5% detector, Lost & Found)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / "laf_rba_vs_knn_example.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_rank_distributions()
    fig_example(args.stem)


if __name__ == "__main__":
    main()
