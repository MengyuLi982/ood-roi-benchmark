"""
visualize_safety_weighted.py
----------------------------
Qualitative 5-panel figure for the safety-aware OoD experiment (Plan B):

    RGB | GT OoD mask | Raw P_ood | Safety weight W_safety | Safety-aware R

Reuses the geometry/normalization from src/evaluation/safety_weighted.py so the
figure shows exactly what the metrics use. Auto-selects multi-object images
(where safety re-ranking can actually change the top object) unless --stems given.

Usage (repo root, `ood` venv):
    python src/visualization/visualize_safety_weighted.py --dataset laf --auto 5
    python src/visualization/visualize_safety_weighted.py --dataset RoadAnomaly21 --method PixOOD --auto 4
    python src/visualization/visualize_safety_weighted.py --dataset laf --stems <stem1>,<stem2>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR  # noqa: E402
from evaluation.safety_weighted import (  # noqa: E402
    METHODS, to_ood_score, norm01, fill_road, build_gate, ego_lane_map, distance_map,
    road_boundaries, score_map_dir, W_LANE, W_DIST,
)
from scoring.compute_depth_maps import image_root, find_rgb  # noqa: E402


def draw_lane_lines(ax, filled, show_ext=True):
    """Overlay the ORIGINAL fitted road edges (cyan) and, if show_ext, the
    right-EXTENDED edge (orange) so the widening is visible."""
    b = road_boundaries(filled)
    if b is None:
        return
    rows, xL, xR, xR_ext = b
    ax.plot(xL, rows, color="cyan", lw=1.0, ls="--", alpha=0.9)
    ax.plot(xR, rows, color="cyan", lw=1.6, ls="--", alpha=0.95, label="original road edge")
    if show_ext:
        ax.plot(xR_ext, rows, color="orange", lw=1.6, ls="-", alpha=0.95,
                label="extended right edge")


def build_maps(dataset, f, method, use_depth, gate_mode="span"):
    d = np.load(f)
    pred = d["pred_class"].astype(np.int32)
    ood = d["ood_label"].astype(np.int32)
    H, W = ood.shape
    filled = fill_road((pred == 0) | (pred == 1))
    Mroad = build_gate(filled, gate_mode)
    L = ego_lane_map(filled)
    if use_depth:
        D = distance_map(dataset, f.stem, (H, W))
        Wsafe = Mroad * (W_LANE * L + W_DIST * (D if D is not None else 0.0))
    else:
        Wsafe = Mroad * L
    p = norm01(to_ood_score(method, d[METHODS[method]].astype(np.float32)))
    R = p * Wsafe
    return ood, p, Wsafe.astype(np.float32), R.astype(np.float32), filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["laf", "RoadAnomaly21", "RoadObstacle21"])
    ap.add_argument("--method", default="Energy", choices=list(METHODS))
    ap.add_argument("--stems", default="", help="comma-separated stems (overrides --auto)")
    ap.add_argument("--auto", type=int, default=5, help="auto-pick N multi-object images")
    ap.add_argument("--no-depth", dest="use_depth", action="store_false")
    ap.add_argument("--road_gate", choices=["span", "poly"], default="span",
                    help="M_road gate: 'span' (original) or 'poly' (right-extended band)")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / ("safety_weighted_polyroad" if args.road_gate == "poly"
                             else "safety_weighted") / "figures"

    smap_dir = score_map_dir(args.dataset)
    files = sorted(smap_dir.glob("*.npz"))
    if not files:
        print(f"[Error] no score maps in {smap_dir}")
        sys.exit(1)

    if args.stems:
        wanted = set(args.stems.split(","))
        files = [f for f in files if f.stem in wanted]
    else:
        scored = []
        for f in files:
            _, n = ndimage.label(np.load(f)["ood_label"] == 1)
            if n >= 2:
                scored.append((n, f))
        scored.sort(key=lambda t: -t[0])
        files = [f for _, f in scored[:args.auto]]
        if not files:  # fallback: any images
            files = sorted(smap_dir.glob("*.npz"))[:args.auto]
    print(f"[Viz] {args.dataset} / {args.method}: {len(files)} images")

    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = image_root(args.dataset)
    titles = ["RGB", "GT OoD", f"Raw P_ood ({args.method})", "W_safety", "R_safety"]

    for f in files:
        ood, p, Wsafe, R, filled = build_maps(args.dataset, f, args.method,
                                              args.use_depth, args.road_gate)
        H, W = ood.shape
        rgb_path = find_rgb(img_root, f.stem) if img_root.exists() else None
        rgb = (np.array(Image.open(rgb_path).convert("RGB").resize((W, H)))
               if rgb_path else np.zeros((H, W, 3), np.uint8))

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(rgb)
        axes[1].imshow(ood, cmap="gray", vmin=0, vmax=1)
        axes[2].imshow(p, cmap="inferno", vmin=0, vmax=1)
        axes[3].imshow(Wsafe, cmap="viridis", vmin=0, vmax=1)
        axes[4].imshow(R, cmap="inferno", vmin=0, vmax=float(max(R.max(), 1e-6)))
        if args.road_gate == "poly":      # original vs extended road edge
            draw_lane_lines(axes[0], filled)
            draw_lane_lines(axes[3], filled)
            axes[0].legend(loc="lower left", fontsize=8, framealpha=0.6)
        for ax, t in zip(axes, titles):
            ax.set_title(t, fontsize=12)
            ax.axis("off")
        fig.suptitle(f"{args.dataset} — {f.stem}", fontsize=13)
        fig.tight_layout()
        mtag = args.method.lower().replace("-", "").replace("v2", "")
        out = out_dir / f"{f.stem}_{mtag}_safety_weighted.png"
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")


if __name__ == "__main__":
    main()
