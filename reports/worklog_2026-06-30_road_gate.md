# Worklog — 2026-06-30 — Road-mask (`M_road`) gate experiments

Session record. Goal for the day: rework the **road-mask gate** `M_road` in the safety
weight `W_safety = M_road·(0.7·L + 0.3·D)` to be conservative about objects near the ego
vehicle's **right** edge (right-hand traffic), then understand why it barely helped and try a
stricter variant. All new work writes to **separate folders**; the original
`results/safety_weighted*` experiment is untouched.

Context going in: the original gate (`span`) is a hole-filled, jagged road+sidewalk mask.
Full background in [safety_weighted_report.md](safety_weighted_report.md).

---

## 1. Poly gate — right-extended polynomial road band

**Idea.** Replace the jagged span mask with a smooth gate: fit degree-2 polynomials to the
left/right road boundaries, then push the **right** edge outward by `BETA_RIGHT·width(y)`
(`BETA_RIGHT=0.15`, width-scaled so the margin is bigger in the foreground). Decisions
confirmed with the user: **pure polynomial band** (not unioned), **BETA_RIGHT=0.15**,
**width-only** margin (no extra perspective factor). Lane term `L` stays computed from the
*original* road so the ego-lane center doesn't drift.

**Implemented** (additive; default `span` reproduces the original bit-for-bit):
- `src/evaluation/safety_weighted.py`: new `road_boundaries()` (returns `rows, xL, xR, xR_ext`),
  `road_gate()`, `build_gate()`, constants `BETA_RIGHT`, `RIGHT_DEG`.
- `--road_gate {span,poly}` flag wired into all four scripts (`safety_weighted.py`,
  `safety_weighted_dino.py`, both `visualize_*`). `poly` routes to new `*_polyroad` folders;
  the depth cache path stays at the original folder (no recompute).
- The detector's on-road TP/FP test now uses the active gate (`Mroad[cm].mean()`).

**Result** — the gate only *reweights*, so detection/TP/FP counts are identical to span.
`R_TP > R_FP` held for kNN/Energy/PixOOD on all 3 datasets; off-road still suppressed
~100–1000×. `R_TP` ticked up slightly; `R_FP` ticked up too (by design — the right margin
grants weight to right-shoulder clutter, clearest on PixOOD/RO21 0.096→0.180). RbA on L&F
still collapses (detector-tail issue, gate can't fix). Write-up:
[safety_weighted_polyroad_report.md](safety_weighted_polyroad_report.md).

**Figures + lane overlay.** Added an original-vs-extended edge overlay to both visualizers
(cyan dashed = original road edge, orange = extended right edge) shown when `--road_gate poly`.
`road_boundaries()` exposes the curves; Exp-1 figure filenames now include the method.
Generated per-method figures (geese + L&F scenes) and an edge close-up showing the strip the
extension adds (**+7.9%** road area, all on the right).

---

## 2. Diagnostic — why does the poly gate barely change L&F?

**Question (user).** Is the small effect because Lost & Found has few objects near the road
lane edge? Built a read-only, method-agnostic analysis (only `pred_class` + `ood_label`):
`src/evaluation/analyze_object_lateral_position.py` → `results/polyroad_analysis/`.

For each GT object it measures lateral position `u_bottom` (0=center, ±1=edges) and whether
any pixel lands in the **extension strip** (`road_gate(filled) & ~road_gate(filled,0)`).

**Answer: yes — L&F obstacles sit in the center of the road.**

| dataset | #GT obj | median u_bottom | % on-road | % near right edge | **% with pixel in strip** |
|---|---|---|---|---|---|
| **L&F** | 1645 | 0.008 (dead center) | 93.1% | 9.4% | **7.3%** |
| RoadAnomaly21 | 18 | 0.018 | 76.5% | 11.8% | 44.4% (small-N) |
| RoadObstacle21 | 45 | −0.05 | 100% | 6.7% | 4.4% |

So ~93% of objects are untouchable by the extension → near-no-op. It's **not** a margin-size
issue (median right margin = 100 px). Separately, `R_FP` movement is area-driven:
`mean_strip_frac_of_road` largest on RO21 (0.101 vs L&F 0.055) → explains RO21/PixOOD's bigger
`R_FP` rise. Write-up: [polyroad_object_position_report.md](polyroad_object_position_report.md).

**Discussion — is it dataset or are we missing something?** Mostly dataset, but a few real
gaps: (1) we evaluate global averages while the prior targets a rare subpopulation — *on the
objects it can touch* it does raise weight (+11% on the 120 in-strip objects; 64% of the 22
true right-shoulder objects boosted); (2) the pure poly band can *shrink* coverage elsewhere
(self-cancellation); (3) geometry can't tell a static roadside object from an intruding one —
the real safety signal is **dynamics**, absent here; (4) no GT for "safety relevance";
(5) the right-only assumption (ego right-biased) is unverified.

---

## 3. Union gate — the strictly-more-conservative variant

**Idea (follow-up to gap #2).** `M_road(union) = road_gate(filled) | filled` — the right-
extended poly band **OR** the hole-filled span road. A strict superset of both: never loses
real road, only adds the right strip. Fixes the poly self-cancellation.

**Implemented standalone** (no edits to existing scripts):
- `src/evaluation/safety_weighted_union.py` — adds a `union` gate mode and drives the existing
  detector pipeline with it (overrides the `build_gate` symbol the pipeline looks up). Results
  → `results/safety_weighted_<method>_union/` + comparison `results/gate_comparison_union/`.
- `src/visualization/visualize_safety_weighted_union.py` — 5-panel detector figures with the
  union gate + lane overlay.

**Result — span → poly → union** (detection counts identical throughout):

| method | dataset | #TP | R_TP s→p→**u** | R_FP s→p→**u** | TP/FP ratio s/p/**u** |
|---|---|---|---|---|---|
| kNN | L&F | 514 | 0.669→0.679→**0.681** | 0.166→0.177→**0.181** | 4.03/3.85/**3.75** |
| kNN | RO21 | 28 | 0.698→0.696→**0.703** | 0.138→0.158→**0.161** | 5.07/4.40/**4.37** |
| Energy | L&F | 386 | 0.711→0.714→**0.714** | 0.224→0.227→**0.251** | 3.17/3.14/**2.84** |
| PixOOD | RO21 | 36 | 0.726→0.732→**0.733** | 0.096→0.180→**0.181** | 7.60/4.07/**4.05** |

**Findings.** Union's `R_TP` ≥ poly's everywhere — confirming the self-cancellation diagnosis
(clearest: kNN/RO21, where poly *dipped below* span 0.698→0.696 and union recovers *above* it
0.703). But gains are 3rd-decimal tiny, and `R_FP` rises monotonically span→poly→union (most
inclusive gate = most edge clutter), so the **TP/FP separation is best for plain span** and
erodes with each extension. **Verdict:** even the strictly-better gate barely moves it — the
bottleneck is the dataset (no shoulder objects), not gate construction.

**Figures.** 36 union figures generated (9 per method: 3 each × 3 datasets) into each
`results/safety_weighted_<method>_union/figures/`.

---

## What the three gates are (one-line mental model)

| gate | what it is | coverage vs real road |
|---|---|---|
| **span** | jagged hole-filled road (original) | exactly the real road |
| **poly** | smooth band, right edge extended | can *lose* some real road, *adds* right strip |
| **union** | poly **OR** span | keeps *all* real road **+** adds right strip |

---

## Files created / changed today

**Code (new):**
- `src/evaluation/analyze_object_lateral_position.py`
- `src/evaluation/safety_weighted_union.py`
- `src/visualization/visualize_safety_weighted_union.py`

**Code (additive edits, backward compatible — default behavior unchanged):**
- `src/evaluation/safety_weighted.py` — `road_boundaries`, `road_gate`, `build_gate`,
  `BETA_RIGHT`, `RIGHT_DEG`, `--road_gate`.
- `src/evaluation/safety_weighted_dino.py` — `--road_gate`, gate threaded through, on-road
  test uses active gate.
- `src/visualization/visualize_safety_weighted.py` and `_dino.py` — `--road_gate`,
  `draw_lane_lines` (original vs extended edge), method-tagged Exp-1 filenames.

**Reports (new):**
- `reports/safety_weighted_polyroad_report.md`
- `reports/polyroad_object_position_report.md`
- `reports/worklog_2026-06-30_road_gate.md` (this file)

**Results (new folders, originals untouched):**
- `results/safety_weighted_polyroad/`, `results/safety_weighted_{dino,energy,pixood,rba}_polyroad/`
- `results/polyroad_analysis/`
- `results/safety_weighted_{dino,energy,pixood,rba}_union/`, `results/gate_comparison_union/`

## Reproduce the day

```bash
# 1. Poly gate — Exp1 + Exp2
./ood/bin/python src/evaluation/safety_weighted.py --dataset all --road_gate poly
for M in DINOv2-kNN Energy PixOOD RbA; do
  ./ood/bin/python src/evaluation/safety_weighted_dino.py --method "$M" --dataset all --road_gate poly
done

# 2. Lateral-position diagnostic
./ood/bin/python src/evaluation/analyze_object_lateral_position.py

# 3. Union gate experiment + figures
./ood/bin/python src/evaluation/safety_weighted_union.py --method all --dataset all
./ood/bin/python src/visualization/visualize_safety_weighted_union.py --method all --dataset all --auto 3
```

## Open threads / next steps
- The extension is a sound **recall-vs-precision safety prior** these datasets don't exercise
  (no shoulder/cut-in anomalies). Demonstrating its value needs a dataset with roadside
  anomalies, or adding a **motion/approach (dynamics)** signal so weight reflects intrusion,
  not just position.
- Not pursued unless asked: `BETA_RIGHT` sweep (0.10/0.15/0.25); a side-by-side span/poly/union
  gate figure on one scene; verifying the ego is actually right-biased per scene.
