# Safety-Weighted BCE — a recipe for putting `W_safety` into training

*Design recipe (not a runnable experiment in this repo). It specifies how to move the safety
weight from post-hoc reweighting into an OoD model's **training loss**, why it is/​isn't
expected to help, and exactly what infrastructure a real fine-tune would need. Everything here
is portable to a proper training environment (GPU + in-distribution + outlier data); nothing
below is wired into or executed by this repository.*

Related: [safety_weighted_report.md](safety_weighted_report.md) (post-hoc reweighting),
[safety_weighted_polyroad_report.md](safety_weighted_polyroad_report.md),
[polyroad_object_position_report.md](polyroad_object_position_report.md) (why the spatial prior
barely moves these datasets).

---

## 0. TL;DR

You already use the safety weight **at inference**: `R = p · W_safety`. The training-time
analogue is one substitution — make the *same* `W_safety` a **per-pixel weight inside a proper
loss**:

```
L_safety  =  Σ_x  W(x) · BCE( p(x), y(x) )   /   Σ_x W(x)
```

with `y = ood_label ∈ {0,1}`, `p(x) = σ(logit(x))`, and `W(x)` a **stop-gradient constant** map
(precomputed once, like depth). This makes the model's *decision boundary* safety-aware, not
just its ranking. Three things must be done right for it to be sound: a **residual weight**
`W_safe = 1 + λ·M(0.7L+0.3D)` so off-road pixels still train; **separate** class-imbalance
handling from spatial weighting; and
an honest understanding that per-pixel reweighting of a proper loss changes *what the finite
model prioritizes*, not the asymptotic optimum. Details below.

---

## 1. From post-hoc reweighting to a loss

Post-hoc (today): the detector is frozen; you multiply its normalized score by the weight,
`R = p · W_safety`, and re-rank. This changes the **ordering** of detections but never the
detector's internal representation — a false positive the detector is confident about is only
*down-weighted*, never *unlearned*.

Training-time: fold the weight into the objective so gradients are scaled by safety relevance.
Any per-pixel OoD loss `L_pixel(p, y)` becomes safety-aware as

```
L_safety = ( Σ_x W(x)·L_pixel(p(x), y(x)) ) / ( Σ_x W(x) ).
```

For the chosen **binary cross-entropy**,
`L_pixel = −[ y·log p + (1−y)·log(1−p) ]`, with `p = σ(logit)`. This is exactly the proper,
backpropagated generalization of the quantity the eval harness already accumulates —
`Aggregator.sw_error = mean(W·|p − y|)` (`src/evaluation/safety_weighted.py:226`) — with the L1
surrogate replaced by a proper log-loss and gradients actually flowing into the model.

---

## 2. The three things that make it correct (not naive)

### 2.1 Proper-scoring-rule caveat — *why this is not the same as post-hoc reweighting*

BCE is a **strictly proper scoring rule**: for a fixed pixel, the per-pixel expected loss is
minimized at `p(x) = P(y=1 | x)` — the true posterior. Multiplying each pixel's loss by a
positive constant `W(x)` does **not** move that per-pixel minimizer. So with **unlimited model
capacity and perfect optimization, the safety-weighted loss and the plain loss have the same
global optimum** — the weighting is asymptotically a no-op.

The benefit is entirely a **finite-capacity / finite-optimization** effect: a real network has
limited capacity and is trained with early stopping, so it cannot fit every pixel equally well.
The weight `W` decides **which pixels get fit first and best** — it spends capacity and gradient
budget on the safety-critical region (on-road, near-lane, near-ego). Think of it as importance
sampling over pixels: equivalent to training on a distribution that oversamples high-`W` pixels.

> **Contrast with post-hoc `R = p·W`:** that *does* change the ranking regardless of capacity,
> because it directly rescales outputs at test time. The loss version instead changes *what the
> model learns to represent well*. The two are complementary — you can train with `L_safety`
> **and** still apply `R = p·W` at inference.

**Practical implication.** Expect the loss to help most when (a) the model is capacity-limited
for the safety-critical region, or (b) that region contains *hard* examples the plain loss
under-fits. If the region is already easy, the loss is close to a no-op — which is exactly the
situation on the current datasets (see §6).

### 2.2 The off-road zero-gradient trap

`W_safety = 0` everywhere off-road (`M_road` gates it to zero). If you weight the loss by raw
`W`, **off-road pixels contribute zero loss and zero gradient** — the model is never trained to
keep OoD scores *low* off-road. That silently discards the very false-positive suppression the
post-hoc gate gives you for free, and can make raw (un-gated) scores *worse* off-road after
fine-tuning.

**Fix — residual (recommended).** Never let the weight hit zero; use an *identity + safety
residual* form so every pixel trains at least at the baseline rate:

```
W_safe(x) = 1 + λ · M_road(x) · (0.7·L(x) + 0.3·D(x))            (λ ≥ 0)
```

Off-road (`M=0`) → `W_safe = 1` (full baseline loss, so FP-suppression is still learned);
on-road → `W_safe ∈ [1, 1+λ]`, a boost peaking at the near lane center. `λ` is the emphasis
strength (`λ=0` = plain BCE; region-to-off-road ratio ≤ `(1+λ):1`); pick a gentle `λ≈0.5–2`.

**Equivalence to a floor (why residual is the better knob).** A multiplicative floor
`W' = w_floor + (1−w_floor)·W` equals `w_floor·(1 + ((1−w_floor)/w_floor)·W)`, i.e. it is the
residual form up to a constant factor. Because the loss below normalizes by `ΣW` (scale-
invariant), **floor and residual are the same objective**, with `λ = (1−w_floor)/w_floor`. Note
`w_floor=0.05` secretly means `λ=19` — a very aggressive emphasis; the residual parametrization
exposes this and lets you tune a sane `λ` directly.

**Loss-side only — do NOT use this for the post-hoc scorer.** Training wants off-road *kept*
(baseline 1) so the model learns to suppress off-road FPs. Inference ranking wants the opposite
— off-road *zeroed* — to keep `R_off ≪ R_on`. So use the residual `W_safe` in the loss, but keep
the **multiplicative gate** `M·(0.7L+0.3D)` for the post-hoc `R = p·W` ranking.

### 2.3 Spatial weight ≠ class weight

OoD pixels are rare (heavy negative-class imbalance). `W_safety` is a **spatial** weight; it is
**not** a fix for class imbalance and the two must not be conflated (e.g. do not "boost OoD
recall" by leaning on `W`). Keep them orthogonal:

- imbalance → `pos_weight` in `binary_cross_entropy_with_logits` (or a focal term),
- safety → the per-pixel `W'` multiplier.

The final per-pixel loss carries **both** factors independently.

### 2.4 Don't put `norm01` in the live path

The post-hoc pipeline maps scores to `[0,1]` with a per-image 1/99-percentile clip
(`norm01`, `safety_weighted.py:80`). That clip has **zero-gradient plateaus** and is per-image,
so it's a poor thing to backprop through. For training, drive BCE from **raw logits → sigmoid**
(`binary_cross_entropy_with_logits`), not from `norm01`. Keep `norm01` only for *evaluation*
parity with the existing reports. Also carry over the **MSP orientation** rule
(`to_ood_score`, `FLIP={"MSP"}`, `safety_weighted.py:52–56`): if the head you fine-tune emits a
*confidence* (higher = in-dist), orient it to an OoD probability (`1 − ·`) before BCE so the
label convention `y=1 ⇒ OoD` holds.

---

## 3. Where `W` comes from (precompute + stop-gradient)

`W_safety` is a pure function of the RGB / road geometry — **it does not depend on the OoD model
being trained.** So it is a *constant* w.r.t. the parameters and needs no gradient. Reuse the
existing geometry unchanged from `src/evaluation/safety_weighted.py`:

| term | function | range |
|---|---|---|
| `M_road` gate | `build_gate(filled, mode)` → `road_gate`/`road_boundaries` | {0,1} |
| `L` ego-lane | `ego_lane_map(filled, alpha=ALPHA)` | (0,1] on-road |
| `D` depth | `distance_map(dataset, stem, (H,W))` (cached) | [0,1] |
| blend | `W = M_road · (W_LANE·L + W_DIST·D)`, `W_LANE=0.7`, `W_DIST=0.3` | [0,1] |

These are NumPy (non-differentiable: morphological fill, row min/max, `polyfit`, rasterization)
— **fine**, because `W` is a constant. The correct pattern is the same one depth already uses:
**precompute `W` once per image and cache it**, then load the tensor in the loop and
`W = W.detach()`. No gradient ever flows through road/lane/depth geometry.

**Prerequisite utility (described, not built here):** a
`src/scoring/precompute_safety_weight.py` mirroring `src/scoring/compute_depth_maps.py` — for
each score-map stem it builds `Wsafe = build_gate(fill_road(...)) · (0.7·ego_lane_map + 0.3·
distance_map)` and saves `float16` to `results/safety_weighted/weight/<dataset>/<stem>.npz`
(key `w`). Depth `D` is already cached at `results/safety_weighted/depth/<dataset>/`, so this is
cheap and model-independent. (This is the only runnable piece the recipe needs; it is left as an
opt-in so this deliverable stays a pure document.)

> Note the gate variant is a choice: `span` (original), `poly` (right-extended), or `union`.
> The training `W` should use whichever gate you want the *trained* model to internalize; `span`
> is the safe default, `poly`/`union` bake in the right-shoulder caution.

---

## 4. What a real fine-tune needs (the out-of-repo parts)

This repo has **no trainable model, no in-repo weights, no outlier-training pairing, and CPU-only
torch** — so the fine-tune itself must run elsewhere. What to stand up:

**Model to fine-tune.** The natural in-family target is **Energy / SegFormer**: the `Energy`
method already derives its score from SegFormer logits (`src/scoring/compute_score_maps.py`), so
the most direct realization is to **unfreeze the SegFormer OoD/segmentation head** and train it
with `L_safety`. `src/models/load_model.py:68–71` currently hard-freezes every parameter
(`requires_grad_(False)`) and runs `forward_logits` under `torch.no_grad()` — those are the two
lines to reverse for the head you want to train. (Alternative: a **DINOv2 linear OoD head** — but
that needs patch features exported to cache, which the repo does not currently do.)

**Supervision + outlier pairing (the missing piece).** OoD fine-tuning (Outlier Exposure /
energy-style) pairs **in-distribution** data with an **outlier** source. In this repo:

- InD is available: `data/id` via `CityscapesInDDataset` (all `ood_label=0`).
- A **real OoD-training pairing is not wired.** The GT-bearing pools that exist are the **L&F
  train partition** (53 images + `gtCoarse` labels, loadable via
  `LostAndFoundOoDDataset(split="train")` — currently only ever called with `"test"`) and the
  1096 cached-label **test** npz.
- **SMIYC (RoadAnomaly21 / RoadObstacle21) is evaluation-only — never train on it.**

For a genuine run, pair Cityscapes-InD with an outlier set (e.g. COCO-derived proxies, or the
L&F-train 53) and hold L&F-test + SMIYC out for evaluation.

---

## 5. Reference implementation (illustrative — not wired, not run here)

**The loss.** ~15 lines; `w` carries the safety weight, `pos_weight` handles class imbalance:

```python
import torch
import torch.nn.functional as F

def safety_weighted_bce(logits, labels, w, lam=1.0, pos_weight=None):
    """
    logits : [B,H,W] raw OoD logit (higher => more OoD; orient confidence heads first)
    labels : [B,H,W] in {0,1}, 1 = OoD          (ood_label from the cache)
    w      : [B,H,W] M_road*(0.7L+0.3D) in [0,1] (precomputed, constant)
    lam    : safety emphasis; lam=0 => plain BCE
    """
    w = w.detach()                                   # stop-gradient: W is a constant
    wsafe = 1.0 + lam * w                            # residual: baseline 1 + boost (§2.2)
    bce = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none",
        pos_weight=pos_weight)                        # class imbalance (§2.3)
    return (wsafe * bce).sum() / wsafe.sum().clamp_min(1e-6)  # normalized weighted mean
```

**Training step (sketch).** DataLoader yields `(rgb, ood_label, W)`; `W` is loaded from the
precomputed cache, not recomputed:

```python
# one-time: unfreeze the OoD/seg head (reverse load_model.py:68-71), build optimizer
opt = torch.optim.AdamW(head.parameters(), lr=1e-4)
pos_weight = torch.tensor([neg_over_pos_ratio])       # from data stats, not from W

for rgb, y, w in loader:                               # w = cached W_safety map
    logits = model.ood_logits(rgb)                     # [B,H,W], head is trainable
    loss = safety_weighted_bce(logits, y, w, w_floor=0.05, pos_weight=pos_weight)
    opt.zero_grad(); loss.backward(); opt.step()
```

*(Not executed in this repo — no trainable head/outlier data/GPU here. Port to a training env.)*

---

## 6. Evaluation protocol (reuses the existing harness)

Compare three regimes on the **same cached-eval code** so numbers are directly comparable to the
current reports:

| regime | description |
|---|---|
| **(a) frozen, unweighted** | detector as-is, no `W` — the baseline |
| **(b) frozen + post-hoc `R=p·W`** | today's result ([safety_weighted_report.md](safety_weighted_report.md)) |
| **(c) safety-weighted-trained** | model fine-tuned with `L_safety`, evaluated **both** raw and with post-hoc `R=p·W` |

**Metrics.** The existing `Aggregator` already emits **AUROC / AP / FPR95** plus `sw_error`
(`safety_weighted.py:238–263`), and `safety_weighted_dino.py` gives **`R_TP` / `R_FP`** and
**off-road suppression** (`R_off ≪ R_on`). Report all of these for (a)/(b)/(c). The question the
table answers: *does training-time weighting beat post-hoc weighting on the safety-critical
region, and does it avoid degrading raw off-road behavior (§2.2)?*

**Expected outcome — stated honestly.** The lateral-position diagnostic
([polyroad_object_position_report.md](polyroad_object_position_report.md)) showed L&F/RO21
anomalies sit at the **road center** (median lateral `u_bottom ≈ 0`, ~93% on-road, only ~7% in
the right-extension strip). So on *these* datasets the safety weight mostly re-emphasizes pixels
the detector already handles and **down-weights already-easy off-road negatives** — by the §2.1
argument that is close to a no-op for a proper loss. Predict a **small** headline change,
consistent with the post-hoc result. The loss earns its keep only where the safety-critical
region is genuinely hard or under-represented: datasets with **roadside/shoulder/cut-in
anomalies**, or after adding a **dynamics/approach** term to `W` so the weight reflects
*intrusion*, not just position.

---

## 7. Limitations / portability

- No trainable model, no in-repo weights, no InD↔outlier training pairing; CPU-only torch here —
  the fine-tune must run in a proper GPU training env.
- GT pool is tiny (L&F-train = 53 imgs); SMIYC is eval-only.
- These benchmarks don't exercise the right-shoulder prior, so they under-report the method's
  potential value (see §6).
- The recipe (loss, floor, precompute-and-detach `W`, eval regimes) is written to lift into that
  training env unchanged — only the model/data plumbing of §4 is environment-specific.

---

## 8. Reproduction / next actions

Nothing to run for the document itself. To make it executable:

1. **(opt-in) build the `W` cache:** `src/scoring/precompute_safety_weight.py` →
   `results/safety_weighted/weight/<dataset>/<stem>.npz` (key `w`), one per score-map
   (laf 1096 / RA21 10 / RO21 30), `[H,W]` in `[0,1]`, `0` off-road. Spot-check `w` equals
   `build_gate(fill_road(...))·(0.7·ego_lane_map + 0.3·distance_map)` for a couple of stems.
2. **In a training env:** stand up the InD↔outlier pairing (§4), unfreeze the SegFormer OoD head
   (`load_model.py:68–71`), drop in `safety_weighted_bce` (§5), train.
3. **Evaluate** with regimes (a)/(b)/(c) (§6) on the existing harness.

| role | path |
|---|---|
| this recipe | `reports/safety_weighted_loss_recipe.md` |
| geometry reused for `W` | `src/evaluation/safety_weighted.py` (`build_gate`, `ego_lane_map`, `distance_map`, constants, `Aggregator`) |
| caching pattern to mirror | `src/scoring/compute_depth_maps.py` |
| freeze point to reverse | `src/models/load_model.py:68–71` |
| GT / splits | `src/dataloaders/{laf,smiyc}_datasets.py` |
