# Gate 11 — preserve the measured operator fingerprint

Gate 10 found a real geometric signal and then failed behaviorally.

Its Q-span selector produced much better conditioned measured cue-response signatures, but the selected examples were subsequently reduced back to ordinary pointwise lower-bound constraints. Different selected banks could therefore collapse to the same active constraint surface and the same accepted parameter trajectory.

Gate 11 tests the next, narrower claim:

> If a bounded learner measures a compact response-operator fingerprint of an old capability and preserves that fingerprint directly during a write, does it retain unseen cue access better than preserving examples alone?

This is not a claim that the fingerprint is the exact `Q`, `HQB`, or any hidden operator in CausalHorizon. It is a black-box finite-response sketch motivated by that decomposition.

## Fixed setup

- Same handwritten-digits model, cue construction, evaluator and 24 adjustable hidden-bias parameters as Gates 9–10.
- Independent seeds: `251, 263, 277`.
- Stored old-cue pool: 60 records.
- Progress checkpoints: `+0.05, +0.10, +0.20, +0.40` calibration log-probability.
- Scalar callback budget: `45,000` per method.
- Update-attempt cap: `24`.
- Finite-model correction count: `2`.
- Signature panel: 12 deterministic orthonormal parameter directions.
- Signature amplitude: `epsilon = 0.03`.
- Fingerprint relative tolerance: `15%` of the frozen baseline mode gain, with absolute floor `5e-4`.

The held-out test split and unseen cue labels remain evaluator-only.

## Frozen response geometry

At the initial parameter vector `theta_0`, measure the same Gate-10 signature matrix

```math
S_{jk}=m_j(\theta_0+\epsilon d_k)-m_j(\theta_0),
```

where `m_j` is the classification margin of old cue `j` and the rows `d_k` are the 12 fixed orthonormal temporary parameter probes.

Compute

```math
S = U \Sigma V^T.
```

For every numerically supported singular mode `r` (up to 12), form an actual 24-dimensional parameter direction

```math
w_r = \sum_k V_{rk} d_k.
```

Choose a distinct pivot cue `i_r` with large leverage `|U_{i_r r}|` for that mode. The frozen mode-gain query is

```math
\phi_r(\theta)
=
m_{i_r}(\theta+\epsilon w_r)-m_{i_r}(\theta).
```

The `w_r`, pivot cue, and baseline value `phi_r(theta_0)` are frozen for the entire run. They are not recomputed after learning steps, so the contract cannot drift with the learner.

## Ordinary anchors

Operator gain alone does not fix the response intercept. Therefore the full Gate-11 guard also retains one ordinary fragile margin anchor per digit class: the lowest-initial-margin cue in that class. Each keeps its original one-sided lower bound.

This creates two new methods:

- `anchor_guard`: the ten frozen class anchors only;
- `operator_guard`: the same ten anchors **plus** the frozen singular-mode gain contracts.

The difference between these methods isolates the effect of protecting measured response geometry rather than simply adding a different replay selector.

## Guard behavior

The existing measured `BehavioralUpdateGuard` is reused. Mode-gain queries are ordinary black-box scalar callbacks from the guard's perspective, except each gain query is charged as two scalar margin measurements because it evaluates the pivot cue at `theta` and `theta + epsilon w_r`.

For each mode gain, preserve

```math
|\phi_r(\theta)-\phi_r(\theta_0)|
\le
\max(5\times10^{-4},\;0.15|\phi_r(\theta_0)|).
```

Every finite-difference query, rejected candidate, model correction and final validation is charged. There is no analytic Jacobian and no access to held-out labels.

## Controls

On the same new seeds, rerun:

- `qspan_scanned` from Gate 10 — 20 examples selected for signature span, then protected only as pointwise margin inequalities;
- `all_pool` — all 60 old-cue lower-bound constraints;
- `unprotected` — equal task machinery without old-behavior protection.

These mechanisms have different acquisition structures. Gate 11 therefore does **not** claim equal per-attempt scan cost. Actual callback calls and model-example evaluations are reported alongside matched-progress behavior.

## Primary comparison

At every checkpoint reached by both methods, compare `operator_guard` first against `anchor_guard`, then against `qspan_scanned`.

Primary outcomes remain the Gate-9/10 outcomes:

- loss rate among initially correct familiar `full/upper/lower` cues;
- loss rate among initially correct unseen masks `upper3/lower3/left4/right4`.

Secondary outcomes:

- achieved new-task progress;
- callback calls and model-example evaluations;
- maximum frozen fingerprint drift;
- number/rank of frozen modes;
- whether the operator guard merely blocks learning.

No interpolation is used. A checkpoint not reached is missing coverage, not successful preservation.

## What counts as support

A useful positive result requires all of the following:

1. the frozen operator guard reaches nontrivial new-task progress;
2. its mode-gain contracts remain within their preregistered tolerances;
3. at matched progress, unseen cue losses are lower than the anchor-only guard on more than one seed;
4. the advantage is not explained solely by more callback budget or by failing to learn.

Beating Q-span examples as well would be stronger evidence that preserving measured response geometry is more useful than merely selecting examples from that geometry.

## What kills the current idea

Any of these are informative failures:

- `operator_guard` and `anchor_guard` take effectively identical parameter paths;
- mode-gain preservation adds no unseen-cue benefit;
- it preserves unseen cues only by preventing progress;
- the gains can remain stable while unseen access still deteriorates;
- the compact mode sketch is too expensive relative to ordinary replay;
- all-pool pointwise replay dominates it at comparable progress and cost.

If the gains are genuinely preserved yet unseen access still fails, the next missing object is likely not a static local response fingerprint. The natural escalation would be a **state-conditioned / nonlinear / trajectory-conditioned response operator**, not another replay ranking rule.

## Claim boundary

SVD/Jacobian sketches, leverage pivots, finite-difference sensitivity measurements, trust-region constrained updates and continual-learning replay are established techniques. Gate 11 claims no novelty for those ingredients.

The experiment asks one narrow question:

> Does directly retaining a frozen low-dimensional black-box response operator protect capability beyond retaining old examples that were chosen using the same geometry?
