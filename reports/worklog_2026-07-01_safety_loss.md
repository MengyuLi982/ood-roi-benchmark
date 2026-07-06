# Worklog — 2026-07-01 — Can the safety score be used to fine-tune the OoD model / as a loss?

Session record. One question today, and the answer we landed on. Full design write-up lives in
[safety_weighted_loss_recipe.md](safety_weighted_loss_recipe.md); this file just records the
dialog and the decision.

---

## Question (user)

> According to the previous report and our plan, how can I use our different safety score to
> fine-tune the OoD model? Or maybe it can be a loss function?

## Answer (short)

Yes — the safety weight `W_safety` you already apply **post-hoc** (`R = p · W_safety`) can be
moved **into training** as a per-pixel weight inside a proper loss. The natural form is a
**safety-weighted BCE**:

```
L_safety = Σ_x  W(x) · BCE( p(x), y(x) )  /  Σ_x W(x)
```

- `y = ood_label ∈ {0,1}` (dense, already in the cache),
- `p(x) = σ(logit(x))` the model's OoD probability,
- `W(x)` = the **same** `W_safety = M_road·(0.7·L + 0.3·D) ∈ [0,1]`, used as a
  **stop-gradient constant** (it depends only on road geometry, not the model, so precompute &
  cache it once — exactly like depth).

Where post-hoc `R = p·W` only re-ranks a frozen model's outputs, the loss version makes the
model's **decision boundary itself** safety-aware. This generalizes the existing
`Aggregator.sw_error = mean(W·|p−y|)` (`safety_weighted.py:226`) from an L1 surrogate to a
proper, backpropagated log-loss.

## The three things that make it correct (not naive)

1. **Proper-scoring-rule caveat.** BCE is a proper scoring rule, so per-pixel reweighting does
   **not** move the per-pixel optimum `p = P(y=1|x)`. With unlimited capacity the weighting is
   an asymptotic no-op — its real effect is **finite-capacity / gradient-budget allocation** to
   safety-critical pixels. (Contrast: post-hoc `R = p·W` changes the ranking regardless of
   capacity. The two are complementary — train with `L_safety`, still reweight at inference.)
2. **Off-road zero-gradient trap.** `W = 0` off-road ⇒ zero loss/gradient there ⇒ the model
   never learns to suppress off-road false positives. Fix with a **weight floor**:
   `W' = w_floor + (1−w_floor)·W`, `w_floor ≈ 0.05`.
3. **Keep imbalance orthogonal to safety.** OoD pixels are rare → handle pos/neg imbalance with
   `pos_weight` (or focal); let `W` handle only *spatial* safety emphasis. Also drive BCE from
   raw logits→sigmoid, not `norm01` (its percentile clip has zero-gradient plateaus).

## Reality check (why the literal fine-tune is out-of-repo)

Two Explore agents confirmed the hard constraints:
- **No trainable model / no training loop / no in-repo weights.** Backbones (SegFormer, DINOv2,
  PixOOD, RbA) are frozen or external.
- **npz caches store only final score maps + `pred_class` + `ood_label`** — **no raw features**
  to feed a head. Torch is **CPU-only** here (Blackwell / 2.2.2).
- A real fine-tune must run in a GPU training env: target the **Energy/SegFormer** head (unfreeze
  `load_model.py:68–71`), pair **Cityscapes-InD** with an **outlier** source (the InD↔outlier
  pairing is **not wired**; the only GT-bearing train pool is L&F-train = 53 imgs w/ gtCoarse;
  SMIYC is eval-only).

## Decisions made this session

- **Deliverable = design recipe** (portable document), not a runnable experiment. →
  `reports/safety_weighted_loss_recipe.md` (loss math, the 3 correctness points, precompute-`W`
  plan, model+data requirements, reference PyTorch snippet, eval protocol, limitations).
- **Loss form = safety-weighted BCE.**
- The one runnable prerequisite — `src/scoring/precompute_safety_weight.py` to cache `W` maps to
  `results/safety_weighted/weight/` — was **described but not built** (opt-in, gated).

## Expected effect (honest)

Small on the current datasets: the lateral-position diagnostic
([polyroad_object_position_report.md](polyroad_object_position_report.md)) showed L&F/RO21
anomalies sit road-center, so the spatial weight mostly re-emphasizes already-easy pixels — the
§ proper-scoring-rule argument then predicts near-no-op. The loss pays off with roadside/
shoulder/cut-in anomalies, or after adding a **dynamics/approach** term to `W`.

## Files this session

- `reports/safety_weighted_loss_recipe.md` — the full recipe (main deliverable).
- `reports/worklog_2026-07-01_safety_loss.md` — this record.
- Memory updated: `safety-weighted-experiment.md` (loss-recipe entry).
