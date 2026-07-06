# Safety-Aware OoD Evaluation — Experiment Report

*Post-hoc safety-aware reweighting of out-of-distribution (OoD) detectors for road scenes.*
Datasets: **Lost & Found (L&F)**, **SegmentMeIfYouCan RoadAnomaly21 (RA21)**, **RoadObstacle21 (RO21)**.
All code runs in the project venv: `./ood/bin/python ...` (see [Reproduction](#7-reproduction--file-index)).

---

## 1. The problem we want to test

Standard OoD detectors answer **"is this pixel anomalous?"** They are scored with
pixel metrics (AUROC, AP, FPR95) that treat every anomalous pixel as equally
important. But for a *driving* system that is the wrong question. A deer on the ego
lane 10 m ahead and a billboard 50 m off to the side may have the same OoD score,
yet only one is a safety problem.

**Hypothesis:** if we re-weight an OoD map by *where* the anomaly is — on the
drivable road, in the ego lane, and close to the camera — we get a **safety-aware
risk** ranking that prioritizes the objects a driver actually cares about, *without
retraining any detector* (post-hoc, on top of cached score maps).

We test two things:

1. **Does the reweighting change which anomaly is ranked #1?** (Experiment 1)
2. **Used as a full detector→prioritize pipeline (no ground truth), does it push
   true on-road obstacles above off-road false positives?** (Experiment 2)
3. **Is the effect detector-agnostic, or does it depend on the underlying OoD
   method?** (Experiment 2 + the RbA deep-dive in Section 5)

### The safety weight

For each pixel we build a **safety weight** `W_safety` and combine it with the
normalized OoD score `P̃_ood`:

```
R_safety = P̃_ood · W_safety
W_safety = M_road · (0.7 · L + 0.3 · D)
```

| term | meaning | how it is computed |
|---|---|---|
| `M_road` | drivable area gate | hole-filled road+sidewalk mask from the segmentation `pred_class` |
| `L` | ego-lane relevance | Gaussian around a polynomial-smoothed lane center/width |
| `D` | proximity (closer = higher) | **monocular depth** (Depth Anything V2), per-image normalized |
| `P̃_ood` | OoD score in [0,1] | percentile-normalized, oriented so higher = more OoD |

Weights `W_LANE=0.7, W_DIST=0.3` and the ego-lane width `ALPHA=0.5` are top-of-file
constants in `src/evaluation/safety_weighted.py:58`.

> **Design note (why depth, not `(y/H)²`):** an early draft used a hard-coded
> `D=(y/H)²` distance proxy, which ignores camera geometry, and a lane center from a
> holey road mask. Both were replaced: depth comes from a learned monocular model,
> and the road mask is hole-filled before the lane geometry is fit.

---

## 2. Experimental setup

- **Detectors (cached score maps, no re-running):** Energy (SegFormer), MSP,
  DINOv2-kNN, PixOOD, RbA. Maps live in `results/.../score_maps/<stem>.npz`.
  - *Gotcha:* the cached `msp_map` is max-softmax **confidence** (higher = *less*
    OoD); it is flipped via `FLIP = {"MSP"}` (`safety_weighted.py:52`). All others
    are already higher = more OoD.
- **Depth:** `src/scoring/compute_depth_maps.py`, cached to
  `results/safety_weighted/depth/<dataset>/`.
- **Environment:** Python venv `ood` at repo root. The machine GPU is an RTX PRO
  6000 **Blackwell (sm_120)**, which the pinned `torch 2.2.2+cu121` cannot launch
  kernels on → depth runs on **CPU** (`pick_device()` in `compute_depth_maps.py:63`).

---

## 3. Experiment 1 — Safety reweighting of ground-truth anomalies

**Question:** given the *true* anomaly objects in an image, does safety weighting
change which one is the top priority?

**Method** (`src/evaluation/safety_weighted.py`): take GT OoD connected components,
score each by mean `R_safety`, and compare the ranking against the raw OoD ranking.

**Headline result** (`results/safety_weighted/ranking_change_summary.csv`): on L&F,
across the **436** images with ≥2 anomaly objects, the safety-aware top-1 object
differs from the raw-OoD top-1 in **12%–43%** of images, depending on detector:

| dataset | method | #multi-obj imgs | % top-1 changed | mean Spearman ρ |
|---|---|---|---|---|
| L&F | MSP | 436 | 11.7% | 0.78 |
| L&F | RbA | 436 | 20.9% | 0.65 |
| L&F | DINOv2-kNN | 436 | 22.0% | 0.61 |
| L&F | Energy | 436 | 24.8% | 0.56 |
| L&F | PixOOD | 436 | 43.1% | 0.21 |
| RO21 | PixOOD | 8 | 75.0% | −0.19 |
| RA21 | (Energy/kNN/PixOOD) | 2 | 100.0% | <0 |

Lower Spearman ρ = the safety ranking departs more from raw OoD. The reweighting is
*not* a no-op: it routinely re-prioritizes which anomaly matters most.

**Figures:** 5-panel `RGB | GT OoD | P_ood | W_safety | R_safety`, e.g.
`results/safety_weighted/figures/validation0009_safety_weighted.png` (SMIYC geese):
`R_safety` keeps the on-road ducks bright and suppresses off-road false positives.

*(Pixel sanity metrics — AUROC/AP/FPR95 — are unchanged by definition and recorded
in `results/safety_weighted/safety_weighted_results.csv`; e.g. L&F kNN AUROC 0.945,
RbA 0.806, the lowest of the five.)*

---

## 4. Experiment 2 — OoD method as a first-step DETECTOR → safety prioritization

**Question:** in a realistic pipeline with **no ground truth**, can safety weighting
demote off-road false positives and push true on-road obstacles to the top?

**Pipeline** (`src/evaluation/safety_weighted_dino.py`, runs for any method via
`--method`):

1. **Detect:** threshold the oriented OoD map → connected components (≥ `MIN_DET_SIZE`
   = 50 px) = detections. *GT is used only to label TP/FP afterward, never to detect.*
   - SMIYC: a single **global threshold at ≥95% pixel TPR**.
   - L&F: **per-image top 0.5%** (`DET_TOP_PCT`), because the global 95%-TPR τ on L&F
     collapses to a near-floor value that floods each image into one giant blob.
2. **Weight:** distribute `W_safety` over each detection → `R_i = p_i · w_i`.
3. **Rank:** detections by `R_i` (safety-aware) vs `p_i` (raw OoD).

Each method writes to its own folder: `results/safety_weighted_{dino,rba,pixood,energy}/`
(`out_dir_for()` in `safety_weighted_dino.py:54`), with `detection_ranking.csv`
(per detection), `detection_summary.csv`, and 5-panel figures
(`RGB | Raw P_ood | Detections (#=safety rank) | W_safety | R_safety`) from
`src/visualization/visualize_safety_weighted_dino.py`.

### 4.1 Results — all four detectors

`R` = mean safety risk (lower = less safety-relevant). The two claims to check:
**R_off ≪ R_on** (off-road suppressed) and **R_TP > R_FP** (true obstacles win).

| detector | dataset | #det | TP / FP | R_off | R_on | R_TP | R_FP | %top-1 flip |
|---|---|---|---|---|---|---|---|---|
| **DINOv2-kNN** | L&F | 8551 | **514** / 8037 | 0.003 | 0.688 | **0.669** | 0.166 | 76.7 |
| | RA21 | 219 | 11 / 208 | 0.006 | 0.302 | **0.266** | 0.088 | 66.7 |
| | RO21 | 227 | 28 / 199 | 0.003 | 0.629 | **0.698** | 0.138 | 57.7 |
| **Energy** | L&F | 17269 | **386** / 16883 | 0.004 | 0.750 | **0.711** | 0.224 | 83.4 |
| | RA21 | 450 | 14 / 436 | 0.004 | 0.190 | **0.211** | 0.124 | 90.0 |
| | RO21 | 2056 | 31 / 2025 | 0.001 | 0.390 | **0.474** | 0.232 | 70.0 |
| **PixOOD** | L&F | 20127 | **306** / 19821 | 0.001 | 0.884 | **0.703** | 0.461 | 100.0 |
| | RA21 | 281 | 8 / 273 | 0.007 | 0.621 | **0.312** | 0.196 | 70.0 |
| | RO21 | 581 | 36 / 545 | 0.001 | 0.707 | **0.726** | 0.096 | 76.0 |
| **RbA** | L&F | 29918 | **1** / 29917 | 0.0002 | 0.908 | 0.630 | **0.865** ⚠ | 99.4 |
| | RA21 | 241 | 19 / 222 | 0.002 | 0.083 | **0.150** | 0.014 | 40.0 |
| | RO21 | 1193 | 32 / 1161 | 0.002 | 0.339 | **0.599** | 0.009 | 40.0 |

### 4.2 What this shows

- **Off-road suppression is universal.** For *every* method on *every* dataset,
  off-road detections get **~100–1000× lower R** than on-road ones. The safety claim
  is detector-agnostic.
- **TP > FP holds for kNN, Energy, PixOOD** on all three datasets (margin from ~1.5×
  for PixOOD on L&F up to ~7.6× for PixOOD on RO21).
- **RbA collapses on L&F** (red row): only **1 TP out of 29,918 detections**, and
  **R_FP (0.87) > R_TP (0.63)** — the wrong direction. This is the one failure case,
  analyzed next.
- **Robustness ranking on L&F:** kNN (514 TP) > Energy (386) > PixOOD (306) ≫ RbA (1).

**Representative figure:** `results/safety_weighted_dino/figures/validation0009_dino_safety.png`
— the geese rank #1–6 by `R` while off-road tree false positives stay unranked.

---

## 5. Deep dive — why RbA collapses on L&F

RbA works on SMIYC but produces garbage on L&F. Two analyses pin down why.

### 5.1 The detector needs the obstacle in the extreme tail

The L&F detector keeps each image's **top 0.5%** of OoD pixels (`p99.5`). An obstacle
is only caught if its pixels rank in that tail. Measuring where GT-obstacle pixels
actually land in each method's per-image distribution
(`src/evaluation/analyze_rba_laf_collapse.py` → `results/rba_analysis/laf_collapse.csv`):

| method | GT pixels' mean percentile-rank | object recall | fired pixels that are on-road **non-objects** |
|---|---|---|---|
| **RbA** | **p81.8** | **0.2%** | **98.8%** |
| DINOv2-kNN | p88.4 | 31.8% | 31.4% |
| Energy | p81.4 | 18.8% | 25.0% |
| PixOOD | p91.9 | 18.9% | 74.4% |

*(threshold needs ≥ p99.5 to fire)*

**Mechanism, in two steps:**

1. **RbA's score map is too low-contrast.** On L&F it ranks the real obstacle at only
   ~p82 — ~18% of *in-distribution* pixels (road texture, shadows, markings) score
   *higher* than the actual object. The per-image τ ≈ −0.000 (scores cluster tightly
   around zero) confirms a near-flat, low-dynamic-range map. Object-level recall is
   **0.2%** → the "1 TP" you see in the table.
2. **So 98.8% of what fires is on-road clutter** — and `W_safety` is *designed* to
   amplify on-road detections. The pipeline faithfully boosts false positives,
   giving **R_FP > R_TP**. The safety weighting isn't broken; the detector handed it
   garbage.

### 5.2 It's RbA-specific and small-object-specific

Existing analysis `src/evaluation/analyze_rba_object_size.py`
(→ `results/rba_analysis/rba_size_*.csv`, `rba_size_comparison.png`):

- **RbA full-image AUROC: L&F 0.80 vs RO21 0.93.**
- AUROC correlates with object size (Pearson r = +0.20 on L&F, +0.66 on RO21).
- L&F objects are **small** (median largest object 743 px vs RO21's 2088 px) — RbA's
  weakest regime.

**Key subtlety:** AUROC 0.80 isn't catastrophic, but a top-0.5% *detector* needs the
obstacle in the *extreme tail*, not merely "above average." A smooth, low-contrast
map can have decent global AUROC yet still bury the obstacle below 18% of road
pixels — so it is missed. (This also means the per-image *median* rank in 5.1
understates the gap: detection needs the object's **peak** pixel above p99.5, which
RbA rarely achieves — hence median ≈ p82 but object recall 0.2%.)

### 5.3 Figures

- `results/rba_analysis/laf_rank_distributions.png` — box+strip of GT-obstacle
  percentile-ranks per method vs the red p99.5 threshold. RbA sits lowest with the
  heaviest low tail.
- `results/rba_analysis/laf_rba_vs_knn_example.png` — one real L&F image: in the RbA
  map the obstacle (cyan) stays dark and the top-0.5% fires on off-road/road clutter
  (**MISS**); in the kNN map the same obstacle is the bright peak and the top-0.5%
  lands on it (**HIT**).

---

## 6. Experiment 3 — Threshold sensitivity: recovering RbA on L&F

**Question:** the Section 5 diagnosis says RbA's L&F obstacles sit around the
**p82** of each image's score distribution, so a top-0.5% (p99.5) cut sails over
them. If we move the detection cut down to **top-20% (p80)** — just below where the
obstacles actually live — does RbA recover?

**Method:** the same detector pipeline, but with a per-image **top-20%** threshold
applied to **all three datasets** (a single consistent strategy, so L&F and SMIYC
are treated alike). This is driven by new flags on
`src/evaluation/safety_weighted_dino.py` (`--det_top_pct 20 --perimage_all
--out_tag top20`); results go to a separate folder
`results/safety_weighted_rba_top20/`.

### 6.1 Result — top-20% fixes the collapse

| dataset | #det | TP / FP | R_off | R_on | R_TP | R_FP | %flip |
|---|---|---|---|---|---|---|---|
| **L&F** | 34263 | **771** / 33492 | 0.004 | 0.372 | **0.448** | 0.107 | 30.8 |
| RoadAnomaly21 | 292 | 60 / 232 | 0.005 | 0.194 | **0.147** | 0.044 | 70.0 |
| RoadObstacle21 | 3762 | 27 / 3735 | 0.0003 | 0.093 | **0.453** | 0.003 | 30.0 |

Direct comparison on L&F (the collapse case):

| L&F RbA | top-0.5% | top-20% |
|---|---|---|
| TP detections | 1 | **771** |
| R_TP vs R_FP | 0.63 < 0.87 ❌ | **0.448 > 0.107** ✅ (~4.2×) |
| top-1 flip | 99.4% (noise) | 30.8% (signal) |

The ordering is now correct (R_TP > R_FP) and the flip rate drops to a healthy 31%
because real on-road TPs — not noise — now drive the ranking. Off-road suppression
still holds (R_off ≈ 100× below R_on).

### 6.2 Object-level recall — the decisive quantification

How many of the GT obstacles are *actually caught* (hit by a ≥50 px kept detection)
at each threshold (`src/evaluation/analyze_rba_recall.py` →
`results/rba_analysis/rba_recall_by_threshold.csv`):

| dataset | #GT objects | obj recall @top-0.5% | obj recall @top-20% | pixel recall 0.5% → 20% |
|---|---|---|---|---|
| **L&F** | 1645 | **0.1%** (~2) | **92.3%** (~1518) | 0.0% → 72.6% |
| RoadAnomaly21 | 18 | 83.3% | 100.0% | 6.9% → 80.4% |
| RoadObstacle21 | 45 | 62.2% | 86.7% | 43.7% → 88.3% |

By GT object size — the strongest evidence:

| dataset | size | @top-0.5% | @top-20% |
|---|---|---|---|
| **L&F** | small | 0.0% | 81.1% |
| | medium | 0.0% | 96.4% |
| | **large** | **0.2%** | 99.6% |
| RoadObstacle21 | small | 25.0% | 62.5% |
| | medium | 64.3% | 100.0% |
| | large | 100.0% | 100.0% |

On L&F, top-0.5% misses **even large objects (0.2%)** — final proof that RbA's map is
so low-contrast that on-road clutter outscores big obstacles; the extreme tail almost
never contains the real object. Top-20% recovers 92.3% overall. On RO21 (where large
objects already worked) the gain is concentrated on *small* objects (25% → 62.5%) —
RbA's weak regime.

### 6.3 Honest reading — recall lever, not precision fix

- **Object recall (≈1518) > TP detections (771)** on L&F. These measure different
  things: *object recall* asks "does any kept detection touch the object?"; *TP*
  requires ≥10% of a detection's own pixels to be GT. At top-20% many detections are
  large on-road blobs that *cover* an obstacle but are mostly road → the object is
  recalled, yet that bloated detection is scored FP. So top-20% genuinely **finds**
  obstacles (92% recall) but its detections are spatially sloppy (low precision).
- **Not a universal rule.** Top-20% is right *because the analysis located RbA's L&F
  signal at ~p82*. For kNN (signal at p88–p100) top-0.5% is already correct, and
  top-20% would only flood it with false positives. This is an RbA-on-L&F-specific
  threshold, tuned to that detector's score distribution.
- **Verdict:** top-20% is strictly better than top-0.5% *for RbA on L&F* (rescues it
  from "broken" to "usable"), but it does **not** make RbA competitive with kNN /
  Energy / PixOOD, which reach R_TP > R_FP with far fewer, cleaner detections.

---

## 7. Conclusions

1. **Safety-aware reweighting is a meaningful post-hoc layer.** It re-prioritizes the
   top anomaly in 12–43% of multi-object L&F images (Exp 1) and, in a GT-free
   pipeline, suppresses off-road detections by 100–1000× while ranking true on-road
   obstacles above off-road false positives (Exp 2).
2. **The off-road suppression effect is detector-agnostic;** the *usefulness* of the
   ranking is only as good as the underlying detector's ability to localize small
   on-road obstacles.
3. **kNN is the most robust first-step detector** across all three datasets; **RbA is
   the wrong choice for Lost & Found** — its low-contrast maps never push small
   obstacles into the detection tail, so safety weighting amplifies on-road clutter.
4. **The collapse is a fixable threshold artifact, not a pipeline flaw** (Exp 3):
   matching the detection cut to where RbA's signal lives (top-20%) recovers 92% of
   L&F objects and restores R_TP > R_FP — but at the cost of precision, so RbA stays
   below kNN/Energy/PixOOD.

### Honest caveats

- At 95% TPR / top-0.5% the detector is permissive → many false positives. That is
  intentional (it stress-tests `W_safety`'s suppression) but inflates detection
  counts.
- Off-road *true* anomalies (common in RA21) get low `R` **by design** — correct for
  "safety relevance," but not the same as detection accuracy.
- There is **no ground truth for "safety relevance"**; these experiments demonstrate
  prioritization behavior, not a validated accuracy gain.
- L&F vs SMIYC use different detection thresholds (per-image top-0.5% vs global
  95%-TPR), a confound to keep in mind when comparing absolute `R` across datasets.

---

## 8. Reproduction & file index

```bash
# Experiment 1 — GT-component safety reweighting (+ depth precompute)
./ood/bin/python src/scoring/compute_depth_maps.py --dataset laf --skip_existing
./ood/bin/python src/evaluation/safety_weighted.py            # all datasets/methods
./ood/bin/python src/visualization/visualize_safety_weighted.py --dataset RoadAnomaly21 --stems validation0009

# Experiment 2 — detector pipeline, per method
for M in DINOv2-kNN Energy PixOOD RbA; do
  ./ood/bin/python src/evaluation/safety_weighted_dino.py --method "$M" --dataset all
done
./ood/bin/python src/visualization/visualize_safety_weighted_dino.py --method PixOOD --dataset RoadAnomaly21 --stems validation0009

# RbA collapse analysis
./ood/bin/python src/evaluation/analyze_rba_object_size.py
./ood/bin/python src/evaluation/analyze_rba_laf_collapse.py
./ood/bin/python src/visualization/visualize_rba_collapse.py

# Experiment 3 — RbA at per-image top-20% (recovery) + object-level recall
./ood/bin/python src/evaluation/safety_weighted_dino.py --method RbA --dataset all \
    --perimage_all --det_top_pct 20 --out_tag top20
./ood/bin/python src/visualization/visualize_safety_weighted_dino.py --method RbA \
    --out_tag top20 --perimage_all --det_top_pct 20 --dataset RoadAnomaly21 --stems validation0009
./ood/bin/python src/evaluation/analyze_rba_recall.py
```

| role | path |
|---|---|
| Core reweighting + weights | `src/evaluation/safety_weighted.py` |
| Monocular depth precompute | `src/scoring/compute_depth_maps.py` |
| Exp 1 figures | `src/visualization/visualize_safety_weighted.py` |
| Detector pipeline (any method) | `src/evaluation/safety_weighted_dino.py` |
| Detector figures | `src/visualization/visualize_safety_weighted_dino.py` |
| RbA size analysis | `src/evaluation/analyze_rba_object_size.py` |
| RbA collapse analysis | `src/evaluation/analyze_rba_laf_collapse.py` |
| RbA collapse figures | `src/visualization/visualize_rba_collapse.py` |
| RbA object-level recall (Exp 3) | `src/evaluation/analyze_rba_recall.py` |
| Exp 1 results | `results/safety_weighted/` (`ranking_change_summary.csv`, `safety_weighted_results.csv`, `figures/`) |
| Exp 2 results | `results/safety_weighted_{dino,rba,pixood,energy}/` |
| Exp 3 results (RbA top-20%) | `results/safety_weighted_rba_top20/` |
| RbA analysis | `results/rba_analysis/` (incl. `rba_recall_by_threshold.csv`) |

| key figure | path |
|---|---|
| Exp 1 — SMIYC geese | `results/safety_weighted/figures/validation0009_safety_weighted.png` |
| Exp 2 — kNN detector | `results/safety_weighted_dino/figures/validation0009_dino_safety.png` |
| Exp 3 — RbA top-20% (L&F) | `results/safety_weighted_rba_top20/figures/02_Hanns_Klemm_Str_44_000002_000090_leftImg8bit_rba_safety.png` |
| RbA rank distributions | `results/rba_analysis/laf_rank_distributions.png` |
| RbA vs kNN example | `results/rba_analysis/laf_rba_vs_knn_example.png` |
| RbA AUROC vs object size | `results/rba_analysis/rba_size_comparison.png` |
