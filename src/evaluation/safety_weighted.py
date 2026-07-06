"""
safety_weighted.py
------------------
Exploratory post-hoc *safety-aware* OoD evaluation (Plan B).

Turns each method's raw OoD score map P_ood into a safety-aware risk map
    R_safety = P~_ood * W_safety,   W_safety = M_road * (0.7*L + 0.3*D)
where
    P~_ood : per-image percentile-normalized OoD score in [0,1],
    M_road : HOLE-FILLED road(+sidewalk) region (so OoD objects sitting ON the
             road, which break the raw segmentation, are still inside the weight),
    L      : ego-lane relevance, a Gaussian around a SMOOTHED lane center x_c(y)
             with smoothed road width w(y) (no per-row min/max noise),
    D      : distance relevance from a cached monocular-depth map (closer -> higher;
             replaces the crude (y/H)^2). Disabled with --no-depth (then W=M_road*L).

Reads only the cached score maps (results/.../score_maps/<stem>.npz) plus the
depth cache (results/safety_weighted/depth/<dataset>/<stem>.npz). Writes to
results/safety_weighted/.

The headline output is ranking_change_summary.csv: how often the top OoD object
per image changes after safety weighting -- the empirical go/no-go signal.

Usage (repo root, `ood` venv):
    python src/evaluation/safety_weighted.py --dataset laf
    python src/evaluation/safety_weighted.py --dataset RoadAnomaly21 --no-depth
    python src/evaluation/safety_weighted.py --dataset all --limit 20
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import spearmanr

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import SCORE_MAPS_LAF, SMIYC_RESULTS_DIR, RESULTS_DIR  # noqa: E402

OUT_DIR    = RESULTS_DIR / "safety_weighted"
DEPTH_ROOT = OUT_DIR / "depth"

# Method display name -> npz key
METHODS = {"Energy": "energy_map", "MSP": "msp_map", "DINOv2-kNN": "knn_map",
           "PixOOD": "pixood_map", "RbA": "rba_map"}
# msp_map is stored as max-softmax CONFIDENCE (higher = less OoD); the standard
# MSP OoD score is 1 - max_softmax. All other maps are already higher = more OoD.
FLIP = {"MSP"}


def to_ood_score(method: str, raw: np.ndarray) -> np.ndarray:
    return (1.0 - raw) if method in FLIP else raw

ALPHA      = 0.5      # ego-lane Gaussian width as fraction of road width
W_LANE     = 0.7      # weight of ego-lane term
W_DIST     = 0.3      # weight of distance term
BETA_RIGHT = 0.15     # poly-gate: right-boundary extension as fraction of road width
RIGHT_DEG  = 2        # poly-gate: polynomial degree for road-boundary fit
N_BINS     = 2000     # histogram bins (scores already normalized to [0,1])
PCT_LO, PCT_HI = 1.0, 99.0   # percentile normalization
TPR_TARGET = 0.95


# --------------------------------------------------------------------------- #
# Dataset wiring
# --------------------------------------------------------------------------- #
def score_map_dir(dataset: str) -> Path:
    if dataset == "laf":
        return SCORE_MAPS_LAF
    return SMIYC_RESULTS_DIR / dataset / "score_maps"


# --------------------------------------------------------------------------- #
# Score normalization
# --------------------------------------------------------------------------- #
def norm01(m: np.ndarray) -> np.ndarray:
    """Per-image robust percentile normalization to [0,1]."""
    lo, hi = np.percentile(m, PCT_LO), np.percentile(m, PCT_HI)
    return np.clip((m - lo) / (hi - lo + 1e-6), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Road geometry: hole-filled mask + smoothed ego-lane / distance maps
# --------------------------------------------------------------------------- #
def fill_road(road_mask: np.ndarray) -> np.ndarray:
    """Hole-filled road region: morphological close + per-row span fill.
    Ensures objects on the road (holes in the raw mask) are inside M_road."""
    H, W = road_mask.shape
    m = road_mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    filled = np.zeros_like(m)
    for y in range(H):
        xs = np.where(m[y])[0]
        if xs.size:
            filled[y, xs.min():xs.max() + 1] = 1
    return filled.astype(bool)


def road_boundaries(filled: np.ndarray, beta_right: float = BETA_RIGHT,
                    deg: int = RIGHT_DEG):
    """Fit smooth road boundaries from `filled`.

    Returns (rows, xL, xR, xR_ext) where rows are the image rows that contain
    road, xL/xR are the polynomial-fit ORIGINAL left/right road edges, and
    xR_ext is the right edge AFTER the beta_right*width extension. Returns None
    when there are too few road rows to fit (caller falls back to `filled`)."""
    ys, xl, xr = [], [], []
    for y in range(filled.shape[0]):
        xs = np.where(filled[y])[0]
        if xs.size:
            ys.append(y)
            xl.append(float(xs.min()))
            xr.append(float(xs.max()))
    if len(ys) < 5:
        return None
    ys = np.asarray(ys, float)
    d = deg if len(ys) >= 8 else 1
    pL = np.polyfit(ys, np.asarray(xl, float), d)
    pR = np.polyfit(ys, np.asarray(xr, float), d)
    rows = np.arange(int(ys.min()), int(ys.max()) + 1)        # only where road exists
    xL = np.polyval(pL, rows.astype(float))
    xR = np.polyval(pR, rows.astype(float))
    xR_ext = xR + beta_right * np.clip(xR - xL, 0.0, None)     # right-only widening
    return rows, xL, xR, xR_ext


def road_gate(filled: np.ndarray, beta_right: float = BETA_RIGHT,
              deg: int = RIGHT_DEG) -> np.ndarray:
    """Polynomial-band road gate (alternative M_road).

    Fits SMOOTH left/right road boundaries x_L(y), x_R(y) with a degree-`deg`
    polynomial (robust to holes/jagged segmentation) and returns the band between
    them -- with the RIGHT boundary pushed outward by `beta_right * width(y)`.
    Rationale: in right-hand traffic the ego vehicle hugs the right side, so
    objects encroaching from the right shoulder are the dangerous ones; widen the
    gate on that side to be conservative about near-lane objects. The margin is
    width-scaled, so it is naturally larger in the (wide) foreground and smaller
    toward the (narrow) horizon.

    Pure band (NOT unioned with `filled`): the gate is exactly the region between
    the two fitted curves. Falls back to `filled` when too few road rows to fit."""
    H, W = filled.shape
    b = road_boundaries(filled, beta_right, deg)
    if b is None:
        return filled.copy()
    rows, xL, _, xR_ext = b
    xL_i = np.clip(np.round(xL), 0, W - 1).astype(int)
    xR_i = np.clip(np.round(xR_ext), 0, W - 1).astype(int)
    gate = np.zeros((H, W), bool)
    for k, y in enumerate(rows):
        a, bb = xL_i[k], xR_i[k]
        if bb >= a:
            gate[y, a:bb + 1] = True
    return gate


def build_gate(filled: np.ndarray, mode: str = "span") -> np.ndarray:
    """Select the M_road gate: 'span' = hole-filled per-row span (`filled` itself,
    original behavior); 'poly' = polynomial right-extended band (`road_gate`)."""
    return road_gate(filled) if mode == "poly" else filled


def ego_lane_map(filled: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Gaussian ego-lane relevance L(x,y) using polynomial-smoothed x_c(y), w(y)."""
    H, W = filled.shape
    ys, xc_raw, w_raw = [], [], []
    for y in range(H):
        xs = np.where(filled[y])[0]
        if xs.size:
            ys.append(y)
            xc_raw.append(0.5 * (xs.min() + xs.max()))
            w_raw.append(xs.max() - xs.min() + 1.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if len(ys) < 5:
        xc = np.full(H, W / 2.0)
        w = np.full(H, W / 2.0)
    else:
        ys = np.asarray(ys, float)
        deg = 2 if len(ys) >= 8 else 1
        pc = np.polyfit(ys, np.asarray(xc_raw, float), deg)
        pw = np.polyfit(ys, np.asarray(w_raw, float), deg)
        rows = np.arange(H, dtype=float)
        xc = np.polyval(pc, rows)
        w = np.polyval(pw, rows)
    w = np.clip(w, 0.04 * W, None)            # avoid div-by-zero / runaway sigma
    xc_col = xc[:, None]
    w_col = w[:, None]
    L = np.exp(-((xx - xc_col) ** 2) / (2.0 * (alpha * w_col) ** 2))
    return L.astype(np.float32)


def distance_map(dataset: str, stem: str, shape) -> np.ndarray | None:
    """Per-image normalized depth as distance relevance D (closer -> higher)."""
    f = DEPTH_ROOT / dataset / f"{stem}.npz"
    if not f.exists():
        return None
    depth = np.load(f)["depth"].astype(np.float32)
    if depth.shape != tuple(shape):
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return norm01(depth)


# --------------------------------------------------------------------------- #
# Streaming pixel-metric aggregator (AUROC/AP/FPR95 + weighted-miss histogram)
# --------------------------------------------------------------------------- #
class Aggregator:
    def __init__(self, n_bins=N_BINS):
        self.edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.hist_pos = np.zeros(n_bins, np.int64)     # unweighted positive counts
        self.hist_neg = np.zeros(n_bins, np.int64)
        self.hist_pos_w = np.zeros(n_bins, np.float64)  # W-weighted positive mass
        self.n_pos = 0
        self.n_neg = 0
        self.sw_err_sum = 0.0   # sum W*|P~ - Y|
        self.n_pix = 0

    def update(self, p_norm, labels, W):
        p = np.clip(p_norm.ravel(), 0.0, 1.0)
        y = labels.ravel()
        w = W.ravel()
        self.sw_err_sum += float(np.sum(w * np.abs(p - y)))
        self.n_pix += p.size
        pos, neg = y == 1, y == 0
        if pos.any():
            idx = np.clip(np.digitize(p[pos], self.edges) - 1, 0, len(self.hist_pos) - 1)
            np.add.at(self.hist_pos, idx, 1)
            np.add.at(self.hist_pos_w, idx, w[pos])
            self.n_pos += int(pos.sum())
        if neg.any():
            self.hist_neg += np.histogram(p[neg], bins=self.edges)[0]
            self.n_neg += int(neg.sum())

    def metrics(self):
        if self.n_pos == 0 or self.n_neg == 0:
            return dict(auroc=float("nan"), ap=float("nan"), fpr95=float("nan"),
                        sw_error=float("nan"), sw_miss=float("nan"))
        cpos = np.cumsum(self.hist_pos[::-1])[::-1].astype(float)
        cneg = np.cumsum(self.hist_neg[::-1])[::-1].astype(float)
        tpr = cpos / self.n_pos
        fpr = cneg / self.n_neg
        auroc = float(np.trapz(np.r_[0, tpr[::-1], 1], np.r_[0, fpr[::-1], 1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(cpos + cneg > 0, cpos / (cpos + cneg), 1.0)
        rec = np.r_[0, tpr[::-1]]
        prec_f = np.r_[1, prec[::-1]]
        ap = float(np.sum(np.diff(rec) * prec_f[1:]))
        valid = tpr >= TPR_TARGET
        # threshold bin t* = highest-score bin still reaching 95% TPR
        if valid.any():
            t_star = int(np.max(np.where(valid)[0]))
            fpr95 = float(fpr[t_star])
            missed_w = float(self.hist_pos_w[:t_star].sum())  # positives below threshold
        else:
            fpr95 = 1.0
            missed_w = float(self.hist_pos_w.sum())
        total_w = float(self.hist_pos_w.sum()) + 1e-9
        return dict(auroc=auroc, ap=ap, fpr95=fpr95,
                    sw_error=self.sw_err_sum / max(self.n_pix, 1),
                    sw_miss=missed_w / total_w)


# --------------------------------------------------------------------------- #
# Per-dataset evaluation
# --------------------------------------------------------------------------- #
def eval_dataset(dataset: str, use_depth: bool, limit: int,
                 comp_rows: list, rank_rows: list, summary_rows: list,
                 result_rows: list, gate_mode: str = "span"):
    smap_dir = score_map_dir(dataset)
    files = sorted(smap_dir.glob("*.npz"))
    if limit:
        files = files[:limit]
    if not files:
        print(f"[{dataset}] no score maps in {smap_dir} -- skipped.")
        return

    methods = [m for m in METHODS if METHODS[m] in np.load(files[0]).files]
    aggs = {m: Aggregator() for m in methods}
    # ranking bookkeeping per method
    rk = {m: dict(n_multi=0, n_changed=0, spearmans=[]) for m in methods}
    n_depth_missing = 0

    print(f"[{dataset}] {len(files)} images, methods={methods}, depth={use_depth}")
    for n, f in enumerate(files):
        d = np.load(f)
        stem = f.stem
        pred = d["pred_class"].astype(np.int32)
        ood = d["ood_label"].astype(np.int32)
        H, W = ood.shape

        road = (pred == 0) | (pred == 1)
        filled = fill_road(road)
        Mroad = build_gate(filled, gate_mode)     # gate: span (orig) or poly band
        L = ego_lane_map(filled)                  # lane center from ORIGINAL road
        if use_depth:
            D = distance_map(dataset, stem, (H, W))
            if D is None:
                n_depth_missing += 1
                D = np.zeros((H, W), np.float32)
            Wsafe = Mroad * (W_LANE * L + W_DIST * D)
        else:
            Wsafe = Mroad * L
        Wsafe = Wsafe.astype(np.float32)

        # connected OoD components from GT (shared across methods)
        labeled, n_obj = ndimage.label(ood == 1)

        for m in methods:
            p = norm01(to_ood_score(m, d[METHODS[m]].astype(np.float32)))
            aggs[m].update(p, ood, Wsafe)

            if n_obj == 0:
                continue
            comps = []
            for oid in range(1, n_obj + 1):
                cm = labeled == oid
                size = int(cm.sum())
                if size == 0:
                    continue
                p_i = float(p[cm].mean())
                w_i = float(Wsafe[cm].mean())
                comps.append((oid, size, p_i, w_i, p_i * w_i))
                comp_rows.append(dict(dataset=dataset, stem=stem, method=m,
                                      obj_id=oid, size=size,
                                      p_ood=round(p_i, 4), w_safety=round(w_i, 4),
                                      R=round(p_i * w_i, 4)))
            if len(comps) >= 2:
                rk[m]["n_multi"] += 1
                top_raw = max(comps, key=lambda c: c[2])[0]
                top_risk = max(comps, key=lambda c: c[4])[0]
                changed = top_raw != top_risk
                rk[m]["n_changed"] += int(changed)
                if len({c[2] for c in comps}) > 1 and len({c[4] for c in comps}) > 1:
                    rho = spearmanr([c[2] for c in comps], [c[4] for c in comps]).correlation
                    if rho == rho:  # not nan
                        rk[m]["spearmans"].append(rho)
                rank_rows.append(dict(dataset=dataset, stem=stem, method=m,
                                      n_obj=len(comps), top_raw_obj=top_raw,
                                      top_risk_obj=top_risk, changed=int(changed)))
        if (n + 1) % 200 == 0:
            print(f"   ... {n + 1}/{len(files)}")

    if n_depth_missing:
        print(f"[{dataset}] WARNING: {n_depth_missing} images missing depth (D=0 there).")

    for m in methods:
        mt = aggs[m].metrics()
        result_rows.append(dict(dataset=dataset, method=m,
                                AUROC=round(mt["auroc"], 4), AUPRC=round(mt["ap"], 4),
                                FPR95=round(mt["fpr95"], 4),
                                SW_Error=round(mt["sw_error"], 6),
                                SW_Miss=round(mt["sw_miss"], 4)))
        nm = rk[m]["n_multi"]
        nc = rk[m]["n_changed"]
        sp = rk[m]["spearmans"]
        summary_rows.append(dict(dataset=dataset, method=m, n_images=len(files),
                                 n_multi_obj=nm, n_top1_changed=nc,
                                 pct_top1_changed=round(100 * nc / nm, 1) if nm else 0.0,
                                 mean_spearman=round(float(np.mean(sp)), 3) if sp else float("nan")))


def write_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[Saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["all", "laf", "RoadAnomaly21", "RoadObstacle21"])
    ap.add_argument("--no-depth", dest="use_depth", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--road_gate", choices=["span", "poly"], default="span",
                    help="M_road gate: 'span' (original) or 'poly' (right-extended "
                         "polynomial band -> results/safety_weighted_polyroad/)")
    args = ap.parse_args()

    datasets = ["laf", "RoadAnomaly21", "RoadObstacle21"] if args.dataset == "all" \
        else [args.dataset]
    # 'poly' writes to a SEPARATE folder so the original experiment is untouched.
    out_dir = RESULTS_DIR / "safety_weighted_polyroad" if args.road_gate == "poly" \
        else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[road_gate={args.road_gate}]  ->  {out_dir}")

    comp_rows, rank_rows, summary_rows, result_rows = [], [], [], []
    for ds in datasets:
        eval_dataset(ds, args.use_depth, args.limit,
                     comp_rows, rank_rows, summary_rows, result_rows, args.road_gate)

    write_csv(out_dir / "safety_weighted_results.csv", result_rows)
    write_csv(out_dir / "component_risk_ranking.csv", comp_rows)
    write_csv(out_dir / "ranking_comparison.csv", rank_rows)
    write_csv(out_dir / "ranking_change_summary.csv", summary_rows)

    print("\n=== Pixel metrics ===")
    print(f"{'dataset':<15}{'method':<12}{'AUROC':>8}{'AUPRC':>8}{'FPR95':>8}"
          f"{'SW_Err':>10}{'SW_Miss':>9}")
    for r in result_rows:
        print(f"{r['dataset']:<15}{r['method']:<12}{r['AUROC']:>8}{r['AUPRC']:>8}"
              f"{r['FPR95']:>8}{r['SW_Error']:>10}{r['SW_Miss']:>9}")
    print("\n=== Ranking-change summary (top-1 OoD object flips after weighting) ===")
    print(f"{'dataset':<15}{'method':<12}{'#multi':>7}{'#changed':>9}{'%changed':>9}{'spearman':>10}")
    for r in summary_rows:
        print(f"{r['dataset']:<15}{r['method']:<12}{r['n_multi_obj']:>7}"
              f"{r['n_top1_changed']:>9}{r['pct_top1_changed']:>9}{r['mean_spearman']:>10}")


if __name__ == "__main__":
    main()
