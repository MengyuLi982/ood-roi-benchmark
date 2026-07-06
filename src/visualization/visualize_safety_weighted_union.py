"""
visualize_safety_weighted_union.py
----------------------------------
Figures for the UNION road-gate experiment (src/evaluation/safety_weighted_union.py).

Same 5-panel detector layout as visualize_safety_weighted_dino.py
    RGB | Raw P_ood | Detections (#=safety rank) | W_safety (union) | R_safety
but the safety weight uses the UNION gate  M_road = road_gate(filled) | filled
(hole-filled span road OR right-extended poly band). Figures are written to each
method's union folder: results/safety_weighted_<method>_union/figures/.

The original road edge (cyan dashed) and the right-extended edge (orange solid) are
overlaid so the added right strip is visible.

This file does NOT modify any existing script; it reuses the helpers from
visualize_safety_weighted_dino.py / safety_weighted_dino.py.

Usage (repo root, `ood` venv):
    # several auto-picked images per method, all datasets:
    ./ood/bin/python src/visualization/visualize_safety_weighted_union.py --method all --dataset all --auto 3
    # a specific image for one method:
    ./ood/bin/python src/visualization/visualize_safety_weighted_union.py --method DINOv2-kNN \
        --dataset RoadAnomaly21 --stems validation0009
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from evaluation.safety_weighted import (  # noqa: E402
    norm01, fill_road, road_gate, ego_lane_map, distance_map, score_map_dir,
    W_LANE, W_DIST,
)
from evaluation.safety_weighted_dino import (  # noqa: E402
    METHODS, oriented, resolve_threshold, tau_for_image, out_dir_for, DET_TOP_PCT,
)
from visualization.visualize_safety_weighted_dino import detections, draw_lane_lines  # noqa: E402
from scoring.compute_depth_maps import image_root, find_rgb  # noqa: E402

MAIN_METHODS = ["DINOv2-kNN", "Energy", "PixOOD", "RbA"]


def union_gate(filled):
    """UNION gate = hole-filled span road OR right-extended poly band."""
    return road_gate(filled) | filled


def pick_files(all_files, method, mode, stems, auto):
    if stems:
        wanted = set(stems)
        return [f for f in all_files if f.stem in wanted]
    scored = []
    for f in all_files:
        s = oriented(np.load(f), method)
        nd = len(detections(s, tau_for_image(mode, s)))
        if nd >= 2:
            scored.append((nd, f))
    scored.sort(key=lambda t: -t[0])
    return [f for _, f in scored[:auto]] or all_files[:auto]


def make_figs(method, dataset, stems, auto, top_k, use_depth, det_top_pct, perimage_all):
    all_files = sorted(score_map_dir(dataset).glob("*.npz"))
    if not all_files:
        print(f"[{dataset}] no score maps -- skipped.")
        return
    mode = resolve_threshold(dataset, all_files, method, det_top_pct, perimage_all)
    files = pick_files(all_files, method, mode, stems, auto)
    out_dir = out_dir_for(method, "union") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = image_root(dataset)
    titles = ["RGB", f"Raw P_ood ({method})", "Detections (#=safety rank)",
              "W_safety (union)", "R_safety"]
    print(f"[Union viz] {method} / {dataset}: {len(files)} images -> {out_dir}")

    for f in files:
        d = np.load(f)
        s = oriented(d, method)
        pred = d["pred_class"].astype(np.int32)
        H, W = s.shape
        filled = fill_road((pred == 0) | (pred == 1))
        Mroad = union_gate(filled)
        L = ego_lane_map(filled)
        Dm = distance_map(dataset, f.stem, (H, W)) if use_depth else None
        Wsafe = (Mroad * (W_LANE * L + W_DIST * (Dm if Dm is not None else 0.0))).astype(np.float32)
        p = norm01(s)
        R = p * Wsafe

        tau = tau_for_image(mode, s)
        ranked = sorted(detections(s, tau), key=lambda cm: -float(R[cm].mean()))

        rgb_path = find_rgb(img_root, f.stem) if img_root.exists() else None
        rgb = (np.array(Image.open(rgb_path).convert("RGB").resize((W, H)))
               if rgb_path else np.zeros((H, W, 3), np.uint8))

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(rgb)
        axes[1].imshow(p, cmap="inferno", vmin=0, vmax=1)
        axes[2].imshow(rgb)
        overlay = np.zeros((H, W, 4))
        for cm in ranked:
            overlay[cm] = (0.4, 0.4, 0.4, 0.30)
        for k, cm in enumerate(ranked[:top_k]):
            overlay[cm] = (*plt.cm.tab10(k % 10)[:3], 0.55)
        axes[2].imshow(overlay)
        for k, cm in enumerate(ranked[:top_k]):
            ys, xs = np.where(cm)
            color = plt.cm.tab10(k % 10)
            axes[2].add_patch(patches.Rectangle(
                (xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                fill=False, edgecolor=color, linewidth=1.8))
            axes[2].text(xs.min(), max(ys.min() - 6, 6),
                         f"#{k+1} R={float(R[cm].mean()):.2f}",
                         color="white", fontsize=9, weight="bold",
                         bbox=dict(facecolor=color, alpha=0.9, pad=0.5, edgecolor="none"))
        axes[3].imshow(Wsafe, cmap="viridis", vmin=0, vmax=1)
        axes[4].imshow(R, cmap="inferno", vmin=0, vmax=float(max(R.max(), 1e-6)))
        draw_lane_lines(axes[2], filled)      # original (cyan) vs extended (orange) edge
        draw_lane_lines(axes[3], filled)
        axes[2].legend(loc="lower left", fontsize=8, framealpha=0.6)
        for ax, t in zip(axes, titles):
            ax.set_title(t, fontsize=12)
            ax.axis("off")
        fig.suptitle(f"{dataset} — {f.stem}  ({method} detector, UNION gate, tau={tau:.3f})",
                     fontsize=13)
        fig.tight_layout()
        out = out_dir / f"{f.stem}_{method.lower().replace('-', '')}_union_safety.png"
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}  ({len(ranked)} detections)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all",
                    help="one of DINOv2-kNN/Energy/PixOOD/RbA/MSP, or 'all'")
    ap.add_argument("--dataset", default="all",
                    choices=["all", "laf", "RoadAnomaly21", "RoadObstacle21"])
    ap.add_argument("--stems", default="", help="comma-separated stems (overrides --auto)")
    ap.add_argument("--auto", type=int, default=3, help="auto-pick N multi-detection images")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--no-depth", dest="use_depth", action="store_false")
    ap.add_argument("--det_top_pct", type=float, default=DET_TOP_PCT)
    ap.add_argument("--perimage_all", action="store_true")
    args = ap.parse_args()

    methods = MAIN_METHODS if args.method == "all" else [args.method]
    datasets = ["laf", "RoadAnomaly21", "RoadObstacle21"] if args.dataset == "all" \
        else [args.dataset]
    stems = [s for s in args.stems.split(",") if s] if args.stems else None

    for m in methods:
        if m not in METHODS:
            print(f"[skip] unknown method {m}")
            continue
        for ds in datasets:
            make_figs(m, ds, stems, args.auto, args.top_k, args.use_depth,
                      args.det_top_pct, args.perimage_all)


if __name__ == "__main__":
    main()
