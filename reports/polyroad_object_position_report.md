# Why the right-extended road gate barely changes L&F — object-position diagnostic

*Answers: "is the small effect because Lost & Found has few objects near the road lane edge?"
Read-only analysis, method-agnostic (uses only `pred_class` + `ood_label`).*
Code: `src/evaluation/analyze_object_lateral_position.py` → `results/polyroad_analysis/`.

## Answer: yes — L&F obstacles sit in the **center** of the road, not near the right edge

Per GT anomaly object we fit the road boundaries (`road_boundaries`), then measure the object's
lateral position `u_bottom` (0 = lane center, ±1 = road edges) and whether any of its pixels
land in the **extension strip** (the right margin the poly gate adds: `poly_ext & ~poly_band`).

| dataset | #GT obj | median u_bottom | % on-road | % near right edge | **% with any pixel in extension strip** | median right margin (px) |
|---|---|---|---|---|---|---|
| **L&F** | 1645 | **0.008** (dead center) | 93.1% | 9.4% | **7.3%** | 100 |
| RoadAnomaly21 | 18 | 0.018 | 76.5% | 11.8% | 44.4% (small-N, scene-spanning) | 132 |
| RoadObstacle21 | 45 | −0.05 | 100% | 6.7% | 4.4% | 86 |

**On L&F, the median object sits essentially on the lane center (u ≈ 0), 93% are squarely
between the road edges, and only 7.3% have a single pixel in the added right strip.** The median
object has **0%** of its pixels in the strip (`median_frac_in_ext_strip = 0`, and the ECDF jumps
to ~0.93 at frac = 0). So for ~93% of objects the right extension has literally nothing to act
on → the near-no-op you observed. This matches what L&F *is*: "debris in the driving path
ahead," not objects on the shoulder.

### It is *not* that the margin is too small
`median_right_margin_px = 100` for L&F (scatter, bottom-left of the figure: margins sit well
above the 1-px line). The extension does add a real, sizeable strip — there are just no objects
out there to catch. The lateral position (centered), not the margin size, is the cause.

## This explains the two halves of the result separately

- **Why `R_TP` barely moved:** TP risk is driven by *GT objects*, and only 7.3% of them touch
  the strip. `pct_dW_sig = 25.2%` of objects do see *some* weight change, but only 20% of those
  are from the right strip — the rest come from the poly-vs-span **smoothing** of the boundary
  (pixels shifting in/out of the gate on both sides), and it is roughly symmetric (291 objects
  up vs 123 down), so it nets out in aggregate `R`.
- **Why `R_FP` moved most on RO21/PixOOD:** FP risk is about *non-GT* road pixels newly gated,
  i.e. strip **area**, not objects. `mean_strip_frac_of_road` is largest on **RO21 (0.101)** vs
  L&F (0.055) — RO21's wide foreground roads add the most off-object right margin, so right-edge
  false positives there gain the most weight. Consistent with PixOOD/RO21's `R_FP` 0.096→0.180.

## Caveats
- RoadAnomaly21 (n = 18) anomalies are large and scene-spanning, so they trivially touch the
  strip (44%); its road fit is weak. Treat as small-N.
- `dW(span→poly)` conflates the right extension with the poly-vs-span reshaping; the clean
  "extension touched this object" signal is `in_ext_strip`, not `dW`.

## Reproduce
```bash
./ood/bin/python src/evaluation/analyze_object_lateral_position.py
```
Outputs: `object_lateral_position_objects.csv` (per object), `object_lateral_position_summary.csv`
(per dataset), `object_lateral_position.png` (2×2 figure).

## Implication
A bigger `BETA_RIGHT` would **not** help L&F — the objects aren't near the edge at all; it would
only inflate `R_FP`. The right-extension is a correct *safety* prior (be cautious about
right-shoulder intrusions) that L&F simply doesn't exercise. A dataset with shoulder/roadside
anomalies (or the right-of-way cut-in case) is where it would pay off.
