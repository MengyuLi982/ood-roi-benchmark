# Safety-Aware OoD — Polynomial Right-Extended Road Gate (`polyroad`)

*A variant of the [safety-weighted experiment](safety_weighted_report.md) that replaces the
`M_road` drivable-area gate. Everything else (depth `D`, ego-lane `L`, detectors, thresholds,
metrics) is unchanged. Original results are untouched; this variant writes to separate
`*_polyroad` folders and is selected with `--road_gate poly`.*

---

## 1. What changed and why

In the safety weight `W_safety = M_road · (0.7·L + 0.3·D)`, the original `M_road` is a
**hole-filled per-row span** of the road+sidewalk segmentation (`fill_road`). It is jagged
(follows the raw segmenter) and symmetric — it simply covers whatever road pixels were seen.

We are driving on the **right** (right-hand traffic), so objects encroaching from the **right
shoulder** — right next to the ego vehicle's lane — are the dangerous ones. The new gate
(`road_gate` in `src/evaluation/safety_weighted.py`):

1. **Fits the road boundaries with a polynomial.** Per row it takes the left/right road
   extent, then fits smooth degree-2 curves `x_L(y)`, `x_R(y)` (deg 1 / fallback to the span
   gate when too few road rows). This removes per-row jitter and naturally fills on-road holes
   (objects that break the segmentation) since the band is solid between the two curves.
2. **Extends the right boundary outward:** `x_R'(y) = x_R(y) + β · (x_R(y) − x_L(y))`,
   `β = BETA_RIGHT = 0.15`. The margin is **width-scaled**, so it is wider in the (wide)
   foreground and narrower toward the (narrow) horizon — i.e. most generous exactly where
   near-lane objects matter most. The left boundary is left untouched.

The gate is the **pure polynomial band** (between the two fitted curves), not a union with the
raw mask. The ego-lane term `L` is still computed from the **original** road (lane center is
not dragged rightward), so the change is isolated to the `M_road` gate as intended.

Config: `BETA_RIGHT = 0.15`, `RIGHT_DEG = 2` (top of `safety_weighted.py`).

**Visual:** `results/safety_weighted_polyroad/figures/_compare_02_Hanns_Klemm_Str_44_000001_000020_leftImg8bit.png`
— span gate (jagged) vs poly gate (smooth, right edge pushed toward the parked cars), with the
matching `W_safety` maps.

---

## 2. Effect on Experiment 1 (GT-component reweighting) — L&F

`% top-1 changed` / `mean Spearman ρ`, original **span** vs new **poly** (436 multi-obj imgs):

| method | span %chg | poly %chg | span ρ | poly ρ |
|---|---|---|---|---|
| MSP | 11.7 | 8.7 | 0.78 | 0.85 |
| RbA | 20.9 | 20.0 | 0.65 | 0.665 |
| DINOv2-kNN | 22.0 | 22.0 | 0.61 | 0.645 |
| Energy | 24.8 | 25.2 | 0.56 | 0.583 |
| PixOOD | 43.1 | 48.4 | 0.21 | 0.116 |

The reweighting behavior is essentially preserved; the only notable shift is **PixOOD**, whose
ranking departs a bit further from raw OoD (48.4% vs 43.1%) — consistent with the right margin
pulling in a few more border objects.

---

## 3. Effect on Experiment 2 (detector → safety prioritization)

The detector and its threshold are unchanged, so **detection / TP / FP counts are identical to
the original** — the gate only changes the *weighting*. The claims to check are still
**R_off ≪ R_on** and **R_TP > R_FP**.

`R_TP` / `R_FP`, original **span** → new **poly** (TP/FP counts in parentheses, same for both):

| detector | dataset | TP / FP | R_TP span→poly | R_FP span→poly | verdict |
|---|---|---|---|---|---|
| **DINOv2-kNN** | L&F | 514 / 8037 | 0.669 → **0.679** | 0.166 → 0.177 | ✅ TP>FP |
| | RA21 | 11 / 208 | 0.266 → **0.296** | 0.088 → 0.088 | ✅ |
| | RO21 | 28 / 199 | 0.698 → **0.696** | 0.138 → 0.158 | ✅ |
| **Energy** | L&F | 386 / 16883 | 0.711 → **0.714** | 0.224 → 0.227 | ✅ |
| | RA21 | 14 / 436 | 0.211 → **0.219** | 0.124 → 0.122 | ✅ |
| | RO21 | 31 / 2025 | 0.474 → **0.474** | 0.232 → 0.235 | ✅ |
| **PixOOD** | L&F | 306 / 19821 | 0.703 → **0.708** | 0.461 → 0.457 | ✅ |
| | RA21 | 8 / 273 | 0.312 → **0.341** | 0.196 → 0.199 | ✅ |
| | RO21 | 36 / 545 | 0.726 → **0.732** | 0.096 → **0.180** | ✅ (FP up) |
| **RbA** | L&F | 1 / 29917 | 0.630 → 0.630 | 0.865 → 0.864 | ❌ collapse (unchanged) |
| | RA21 | 19 / 222 | 0.150 → **0.160** | 0.014 → 0.014 | ✅ |
| | RO21 | 32 / 1161 | 0.599 → **0.599** | 0.009 → 0.014 | ✅ |

Off-road suppression is intact across the board (`R_off` ≈ 0.002–0.009, still ~100–1000× below
`R_on`).

### Reading

- **The poly gate keeps every headline conclusion.** `R_TP > R_FP` still holds for
  kNN / Energy / PixOOD on all three datasets; off-road is still strongly suppressed.
- **`R_TP` nudges up almost everywhere** — the smooth band drops the jagged dead-zones that
  occasionally clipped real on-road objects in the span gate.
- **`R_FP` nudges up too, by design.** The right extension grants a little weight to
  right-shoulder false positives. The clearest case is **PixOOD on RO21 (0.096 → 0.180)**:
  road obstacles there sit near the right edge, so the margin captures more nearby clutter.
  `R_TP` (0.732) still dominates, but this is the explicit safety/precision trade — we chose to
  be more cautious about objects near the right of the driving area.
- **RbA on L&F still collapses (1 TP).** Expected: that failure is the detector's low-contrast
  tail (Section 5 of the original report), which the gate cannot fix — the gate only reweights
  detections the detector already produced.

---

## 4. Reproduction

```bash
# Exp 1 — GT-component reweighting, poly gate  -> results/safety_weighted_polyroad/
./ood/bin/python src/evaluation/safety_weighted.py --dataset all --road_gate poly

# Exp 2 — detector pipeline, poly gate  -> results/safety_weighted_{dino,energy,pixood,rba}_polyroad/
for M in DINOv2-kNN Energy PixOOD RbA; do
  ./ood/bin/python src/evaluation/safety_weighted_dino.py --method "$M" --dataset all --road_gate poly
done

# Figures (append --road_gate poly; output lands in the *_polyroad folder)
./ood/bin/python src/visualization/visualize_safety_weighted.py --dataset laf --auto 5 --road_gate poly
./ood/bin/python src/visualization/visualize_safety_weighted_dino.py --method DINOv2-kNN \
    --dataset RoadAnomaly21 --stems validation0009 --road_gate poly
```

Default (`--road_gate span`, or omitting the flag) reproduces the original experiment bit-for-bit.

| role | path |
|---|---|
| New gate `road_gate()` + `build_gate()` + `BETA_RIGHT`/`RIGHT_DEG` | `src/evaluation/safety_weighted.py` |
| Exp 1 poly results | `results/safety_weighted_polyroad/` |
| Exp 2 poly results | `results/safety_weighted_{dino,energy,pixood,rba}_polyroad/` |
| span-vs-poly gate figure | `results/safety_weighted_polyroad/figures/_compare_*.png` |

### Figure index (poly gate)

Visualizers draw the **original road edge (cyan dashed)** vs the **extended right edge
(orange solid)** whenever `--road_gate poly` is passed, so the widening is explicit.

| figure | path |
|---|---|
| Edge close-up (original vs extended, +area highlighted) | `results/safety_weighted_polyroad/figures/_edge_closeup_02_Hanns_Klemm_Str_44_000002_000090_leftImg8bit.png` |
| Gate compare (span vs poly) | `results/safety_weighted_polyroad/figures/_compare_02_Hanns_Klemm_Str_44_000001_000020_leftImg8bit.png` |
| Exp 1, geese, per method | `results/safety_weighted_polyroad/figures/validation0009_{energy,dinoknn,pixood,rba}_safety_weighted.png` |
| Exp 2 detector, geese, per method | `results/safety_weighted_{dino,energy,pixood,rba}_polyroad/figures/validation0009_*_safety.png` |
| Exp 2 detector, L&F scene (kNN) | `results/safety_weighted_dino_polyroad/figures/02_Hanns_Klemm_Str_44_000002_000090_leftImg8bit_dinov2knn_safety.png` |
