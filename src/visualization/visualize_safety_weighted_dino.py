"""
visualize_safety_weighted_dino.py
---------------------------------
5-panel figure for the "OoD method as DETECTOR -> safety-weighting" experiment:

    RGB | Raw P_ood | Detections (numbered by safety rank) | W_safety | R_safety

Works for any cached method via --method (default DINOv2-kNN; e.g. --method RbA).
Detections come from the same per-dataset threshold as safety_weighted_dino.py
(SMIYC: global 95%-TPR; L&F: per-image top percentile). Each detection is outlined;
the top-K by safety-aware risk R are highlighted and labelled "#k". Figures are
written to the method's folder (DINOv2-kNN -> safety_weighted_dino/figures/,
RbA -> safety_weighted_rba/figures/).

Usage (repo root, `ood` venv):
    ./ood/bin/python src/visualization/visualize_safety_weighted_dino.py --method RbA --dataset RoadAnomaly21 --stems validation0009
    ./ood/bin/python src/visualization/visualize_safety_weighted_dino.py --method RbA --dataset laf --auto 3
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
from scipy import ndimage

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from evaluation.safety_weighted import (  # noqa: E402
    norm01, fill_road, build_gate, ego_lane_map, distance_map, road_boundaries,
    score_map_dir, W_LANE, W_DIST,
)
from evaluation.safety_weighted_dino import (  # noqa: E402
    METHODS, oriented, MIN_DET_SIZE, resolve_threshold, tau_for_image, out_dir_for,
)
from scoring.compute_depth_maps import image_root, find_rgb  # noqa: E402


def detections(s, tau):
    labeled, n = ndimage.label(s >= tau)
    return [labeled == did for did in range(1, n + 1)
            if int((labeled == did).sum()) >= MIN_DET_SIZE]


def draw_lane_lines(ax, filled, show_ext=True):
    """Overlay ORIGINAL fitted road edges (cyan) + right-EXTENDED edge (orange)."""
    b = road_boundaries(filled)
    if b is None:
        return
    rows, xL, xR, xR_ext = b
    ax.plot(xL, rows, color="cyan", lw=1.0, ls="--", alpha=0.9)
    ax.plot(xR, rows, color="cyan", lw=1.6, ls="--", alpha=0.95, label="original road edge")
    if show_ext:
        ax.plot(xR_ext, rows, color="orange", lw=1.6, ls="-", alpha=0.95,
                label="extended right edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="DINOv2-kNN", choices=list(METHODS))
    ap.add_argument("--dataset", required=True,
                    choices=["laf", "RoadAnomaly21", "RoadObstacle21"])
    ap.add_argument("--stems", default="")
    ap.add_argument("--auto", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=6,
                    help="label only the top-K detections by safety rank")
    ap.add_argument("--no-depth", dest="use_depth", action="store_false")
    ap.add_argument("--det_top_pct", type=float, default=None,
                    help="per-image top-percent threshold (matches the eval run)")
    ap.add_argument("--perimage_all", action="store_true",
                    help="force per-image top-pct threshold on ALL datasets")
    ap.add_argument("--out_tag", default="",
                    help="suffix for the output folder, e.g. top20")
    ap.add_argument("--road_gate", choices=["span", "poly"], default="span",
                    help="M_road gate: 'span' (original) or 'poly' (right-extended band)")
    args = ap.parse_args()

    from evaluation.safety_weighted_dino import DET_TOP_PCT  # noqa: E402
    det_top_pct = args.det_top_pct if args.det_top_pct is not None else DET_TOP_PCT

    smap_dir = score_map_dir(args.dataset)
    all_files = sorted(smap_dir.glob("*.npz"))
    if not all_files:
        print(f"[Error] no score maps in {smap_dir}"); sys.exit(1)
    mode = resolve_threshold(args.dataset, all_files, args.method,
                             det_top_pct, args.perimage_all)
    print(f"[Viz] {args.method} / {args.dataset} threshold mode={mode[0]} ({mode[1]})")

    if args.stems:
        wanted = set(args.stems.split(","))
        files = [f for f in all_files if f.stem in wanted]
    else:
        scored = []
        for f in all_files:
            s = oriented(np.load(f), args.method)
            nd = len(detections(s, tau_for_image(mode, s)))
            if nd >= 2:
                scored.append((nd, f))
        scored.sort(key=lambda t: -t[0])
        files = [f for _, f in scored[:args.auto]] or all_files[:args.auto]

    tag = "_".join(t for t in [args.out_tag,
                               "polyroad" if args.road_gate == "poly" else ""] if t)
    out_dir = out_dir_for(args.method, tag) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = image_root(args.dataset)
    titles = ["RGB", f"Raw P_ood ({args.method})", "Detections (#=safety rank)",
              "W_safety", "R_safety"]

    for f in files:
        d = np.load(f)
        s = oriented(d, args.method)
        pred = d["pred_class"].astype(np.int32)
        H, W = s.shape
        filled = fill_road((pred == 0) | (pred == 1))
        Mroad = build_gate(filled, args.road_gate)
        L = ego_lane_map(filled)
        if args.use_depth:
            Dm = distance_map(args.dataset, f.stem, (H, W))
            Wsafe = Mroad * (W_LANE * L + W_DIST * (Dm if Dm is not None else 0.0))
        else:
            Wsafe = Mroad * L
        Wsafe = Wsafe.astype(np.float32)
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
        for k, cm in enumerate(ranked[:args.top_k]):
            overlay[cm] = (*plt.cm.tab10(k % 10)[:3], 0.55)
        axes[2].imshow(overlay)
        for k, cm in enumerate(ranked[:args.top_k]):
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
        if args.road_gate == "poly":      # original vs extended road edge
            draw_lane_lines(axes[2], filled)
            draw_lane_lines(axes[3], filled)
            axes[2].legend(loc="lower left", fontsize=8, framealpha=0.6)
        for ax, t in zip(axes, titles):
            ax.set_title(t, fontsize=12)
            ax.axis("off")
        fig.suptitle(f"{args.dataset} — {f.stem}  ({args.method} detector, tau={tau:.3f})",
                     fontsize=13)
        fig.tight_layout()
        out = out_dir / f"{f.stem}_{args.method.lower().replace('-', '')}_safety.png"
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}  ({len(ranked)} detections)")


if __name__ == "__main__":
    main()
