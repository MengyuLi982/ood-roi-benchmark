"""
compute_depth_maps.py
---------------------
Precompute monocular relative-depth maps for the safety-aware OoD experiment
(Plan B). Replaces the crude (y/H)^2 distance proxy with a learned depth model.

Model: Depth Anything V2 (HuggingFace transformers, AutoModelForDepthEstimation).
Its output is a relative *inverse-depth* / disparity map where LARGER values are
NEARER. We resize it to the score-map resolution and cache it; the eval
(safety_weighted.py) reads it, normalizes per image to [0,1] and uses it as the
distance relevance D(x,y) (closer -> higher).

Per image, RGB is located by matching the cached score-map stem
(results/.../score_maps/<stem>.npz) back to the dataset image. Output:

    results/safety_weighted/depth/<dataset>/<stem>.npz
        - depth: float16 [H, W]   (raw model output, near = high; NOT normalized)

Usage (from repo root, inside the `ood` venv):
    python src/scoring/compute_depth_maps.py --dataset laf
    python src/scoring/compute_depth_maps.py --dataset RoadAnomaly21 --skip_existing
    python src/scoring/compute_depth_maps.py --dataset RoadObstacle21 --limit 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import (  # noqa: E402
    DATA_LAF, DATA_SMIYC, SCORE_MAPS_LAF, SMIYC_RESULTS_DIR, RESULTS_DIR,
)

DEPTH_ROOT = RESULTS_DIR / "safety_weighted" / "depth"
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# Map dataset key -> (score-map dir, RGB image root, image glob suffixes)
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def score_map_dir(dataset: str) -> Path:
    if dataset == "laf":
        return SCORE_MAPS_LAF
    if dataset in DATA_SMIYC:
        return SMIYC_RESULTS_DIR / dataset / "score_maps"
    raise ValueError(f"Unknown dataset '{dataset}'. Use laf|RoadAnomaly21|RoadObstacle21.")


def image_root(dataset: str) -> Path:
    if dataset == "laf":
        root = DATA_LAF
        base = "leftImg8bit-Objekte" if (root / "leftImg8bit-Objekte").exists() else "leftImg8bit"
        return root / base
    if dataset in DATA_SMIYC:
        return DATA_SMIYC[dataset] / "images"
    raise ValueError(f"Unknown dataset '{dataset}'.")


def pick_device(torch) -> str:
    """Use CUDA only if torch actually has kernels for this GPU's arch.
    (Blackwell sm_120 is unsupported by torch 2.2.2 -> fall back to CPU.)"""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list():
            return "cuda"
        print(f"[Depth] GPU sm_{cap[0]}{cap[1]} unsupported by torch "
              f"{torch.__version__}; using CPU.")
    return "cpu"


def find_rgb(img_root: Path, stem: str) -> Path | None:
    """Find the RGB file whose stem matches the score-map stem."""
    # SMIYC: stem == image stem, images/ is flat.
    for ext in _IMG_EXTS:
        cand = img_root / f"{stem}{ext}"
        if cand.exists():
            return cand
    # Lost & Found: images nested under test/<sequence>/<stem>.png
    for ext in _IMG_EXTS:
        hits = list(img_root.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["laf", "RoadAnomaly21", "RoadObstacle21"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    smap_dir = score_map_dir(args.dataset)
    npz_files = sorted(smap_dir.glob("*.npz"))
    if not npz_files:
        print(f"[Error] No score maps in {smap_dir}. Run scripts/download_score_maps.py first.")
        sys.exit(1)
    if args.limit:
        npz_files = npz_files[:args.limit]

    img_root = image_root(args.dataset)
    if not img_root.exists():
        print(f"[Error] RGB images not found at {img_root}. "
              f"Download the dataset (see data/README.md).")
        sys.exit(1)

    out_dir = DEPTH_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(torch)
    print(f"[Depth] device={device}  model={args.model}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()

    n_done = n_skip = n_missing = 0
    for f in npz_files:
        stem = f.stem
        out_path = out_dir / f"{stem}.npz"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue

        # Target resolution = score-map resolution.
        d = np.load(f)
        H, W = (int(d["shape"][0]), int(d["shape"][1])) if "shape" in d.files \
            else d["ood_label"].shape

        rgb_path = find_rgb(img_root, stem)
        if rgb_path is None:
            print(f"[Skip] no RGB for stem '{stem}'")
            n_missing += 1
            continue

        img = Image.open(rgb_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            pred = model(**inputs).predicted_depth  # [1, h, w], near = high
        depth = F.interpolate(pred.unsqueeze(1).float(), size=(H, W),
                              mode="bicubic", align_corners=False).squeeze().cpu().numpy()

        np.savez_compressed(out_path, depth=depth.astype(np.float16))
        n_done += 1
        if n_done % 50 == 0:
            print(f"[Depth] {n_done} done ...")

    print(f"[Depth] dataset={args.dataset}: {n_done} computed, {n_skip} skipped, "
          f"{n_missing} missing RGB. -> {out_dir}")


if __name__ == "__main__":
    main()
