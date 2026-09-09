# Gate 12 — preserve the actual action on a measured basis

Gate 11 successfully preserved a frozen local parameter-response fingerprint and still failed to improve held-out unseen cue access. That result exposed a conceptual mismatch.

CausalHorizon's linear corollary says, schematically,

```math
\Delta y(b)=H Q b.
```

If a retained bank spans the relevant `Qb` entry space, then the useful condition is that the **actual candidate change** vanish on that basis:

```math
H Q B_{\mathrm{basis}} \approx 0.
```

It does not require the old parameter Jacobian itself to remain fixed.

Gate 12 therefore uses Gate-10 response geometry only to identify a basis and then protects the **actual old responses on that frozen basis** as tight two-sided contracts.

## Fixed setup

- Same handwritten-digits model, cue construction, evaluator and 24 adjustable hidden-bias parameters as Gates 9–11.
- Independent seeds: `293, 307, 331`.
- Stored old-cue pool: 60 records.
- Frozen active basis capacity: 20 records, exactly two per digit class.
- Progress checkpoints: `+0.05, +0.10, +0.20, +0.40` calibration log-probability.
- Scalar callback budget: `45,000` per method.
- Update-attempt cap: `24`.
- Measured finite-model correction count: `2`.
- Signature panel: 12 deterministic orthonormal parameter directions.
- Signature amplitude: `epsilon = 0.03`.
- Basis-response tolerance for cue `j`:

```math
\tau_j=\max(0.002,\;0.05|m_j(\theta_0)|).
```

The held-out test split and unseen cue labels remain evaluator-only.

## Frozen basis discovery

At `theta_0`, both new methods pay for the same Gate-10 response-signature scan:

```math
S_{jk}=m_j(\theta_0+\epsilon d_k)-m_j(\theta_0).
```

Two frozen 20-record banks are then formed:

- `qspan_equal`: Gate-10 D-optimal / log-determinant signature-span selection, still exactly two cues per class;
- `boundary_equal`: the two lowest-initial-margin cues per class, ignoring the measured signature geometry after paying for it.

The selected records and their baseline responses are frozen for the entire run. The bank is not reselected after learning.

## What is protected

For every selected cue `j`, retain the baseline classification margin

```math
m_j(\theta_0)
```

and require every accepted write to satisfy

```math
|m_j(\theta)-m_j(\theta_0)|\le\tau_j.
```

The existing `BehavioralUpdateGuard` is run in its equality-projection mode. Its local model therefore attempts to restore the baseline responses themselves, not merely keep them above a one-sided survival bound. Every finite-difference measurement, rejected nonlinear candidate, correction and validation is charged.

This is the key distinction from Gate 10:

```text
Gate 10: geometry -> choose examples -> keep each above a lower bound
Gate 12: geometry -> choose basis    -> keep actual basis action near zero
```

## Controls

On the same new seeds, rerun:

- `qspan_scanned` from Gate 10 — geometry-selected examples protected only by one-sided lower bounds and reselected during learning;
- `all_pool` — all 60 one-sided old-cue lower-bound constraints;
- `unprotected` — task learning without old-behavior protection.

Gate 12 does not claim identical acquisition structures across all mechanisms. Actual callback calls and model-example evaluations are reported.

## Primary comparisons

At every checkpoint reached by both methods:

1. compare `qspan_equal` against `boundary_equal` to test whether the measured basis geometry matters under the stronger action-preservation contract;
2. compare `qspan_equal` against Gate-10 `qspan_scanned` to test whether preserving actual basis responses is better than preserving one-sided survival margins on geometry-selected examples.

Primary outcomes remain:

- loss rate among initially correct familiar `full/upper/lower` cues;
- loss rate among initially correct unseen masks `upper3/lower3/left4/right4`.

Secondary outcomes:

- achieved new-task progress;
- callback calls and model-example evaluations;
- maximum selected-basis response drift;
- measured signature conditioning of the frozen bank;
- whether near-equality preservation simply blocks learning.

No interpolation is used. Failure to reach a checkpoint is missing coverage, not perfect preservation.

## What counts as support

A useful positive result requires:

1. `qspan_equal` reaches nontrivial new-task progress;
2. accepted basis responses remain inside the preregistered two-sided tolerances;
3. at matched progress, unseen cue losses are lower than `boundary_equal` on more than one seed;
4. the advantage is not explained solely by extra measurement cost or reduced progress.

Beating Gate-10 `qspan_scanned` at matched progress would be stronger evidence that **protecting action on a basis** is more useful than merely selecting examples from that basis.

## What kills the static-basis idea

Informative failures include:

- Q-span and boundary equalities behave the same;
- the Q-span bank is better conditioned but gives no unseen-cue advantage;
- tight basis preservation blocks learning before useful progress;
- selected basis responses remain stable while unseen cues still fail;
- one-sided all-pool replay does as well or better at comparable progress and cost.

If the measured basis is well conditioned, its actual responses are genuinely preserved, and unseen access still fails, then the remaining hypothesis becomes much sharper:

> the relevant entry geometry is not a fixed cue subspace at `theta_0`; it is state-, history-, and/or proposed-write-dependent.

That is the point to test a dynamic object such as

```math
Q(\theta, x, h, \Delta\theta)
```

or a trajectory-conditioned response map rather than another static replay bank.

## Claim boundary

D-optimal subset selection, Jacobian/finite-difference sketches, equality-constrained trust-region updates and replay are established tools. Gate 12 claims no novelty for those ingredients and does not assume the nonlinear classifier satisfies CausalHorizon's exact theorem.

The narrow question is:

> If measured response geometry identifies a frozen basis, does directly preserving the actual responses on that basis protect unseen capability better than preserving ordinary replay inequalities?
