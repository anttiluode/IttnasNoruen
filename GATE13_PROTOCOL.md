# Gate 13 — rolling proposed-write-conditioned causal basis

Gate 12 showed that a fixed, well-conditioned Q-like cue basis can be held almost exactly while unseen cue access still deteriorates. It also showed a small local advantage for Q-span at the first `+0.05` checkpoint that did not persist as learning moved away from `theta_0`.

Gate 13 tests the next hypothesis directly:

> the useful old-capability entry geometry is not static; it must be re-estimated at the current state, and may depend on the proposed write itself.

## Preregistered methods

All three methods pay for the same per-attempt acquisition skeleton before the protected update:

1. measure an ordinary unprotected proposal toward the current new-task target;
2. measure all 60 stored old cues at the current parameter state;
3. measure the same 60 cues under a 12-direction finite perturbation panel;
4. choose or reuse a 20-cue two-per-class basis;
5. anchor the relevant pre-write responses;
6. run the same equality guard and finite validation.

The methods differ only in what geometry determines the basis and whether the anchors are frozen or rolling.

### `static_scanned_equal`

- A Q-span basis is selected once at `theta_0` from a generic 12-direction panel.
- Its baseline responses at `theta_0` remain the protected equalities for the whole run.
- Every later attempt still pays for the proposal and a fresh 12-direction scan, but ignores those scans for selection. This equalizes acquisition cost with the rolling methods.

### `rolling_current_equal`

- Before each atomic write, measure a fresh generic 12-direction response-signature panel at the current `theta_t`.
- Select a fresh Q-span basis from that current geometry.
- Anchor each selected cue to its **current pre-write** response.
- Only then allow the protected update.

### `rolling_write_equal`

- First measure the ordinary proposed write `Delta theta_t`.
- Construct the 12-direction panel with the normalized proposed direction as its first axis, plus 11 deterministic orthogonal directions.
- Measure the old-cue response signatures under that write-conditioned panel.
- Select a fresh Q-span basis and anchor its **current pre-write** responses.
- Only then run the protected update.

The basis may be rebuilt only before an atomic candidate write. It may not use the accepted post-write state to redefine what mattered for that write.

## Fixed design

- Dataset/model/task family: unchanged handwritten-digit setup from Gates 9–12.
- Independent seeds: `347, 359, 373`.
- Stored candidate pool: 60 old cues.
- Active basis: 20 cues, exactly two per class.
- Adjustable parameters: 24 hidden-bias offsets.
- Progress checkpoints: `+0.05, +0.10, +0.20, +0.40`.
- Per-method scalar callback budget: `45,000`.
- Attempt cap: `24`.
- Signature probes: `12`.
- Signature amplitude: `epsilon = 0.03`.
- Equality tolerance: `max(0.002, 0.05 * abs(pre_write_margin))`.
- Trust radius and guard implementation: unchanged from Gate 12.

## Primary test

At every checkpoint reached by both methods, compare:

```text
rolling_write_equal - rolling_current_equal
```

for unseen-cue loss among initially correct unseen masks. Negative difference favors proposed-write conditioning.

Secondary matched-progress comparison:

```text
rolling_write_equal - static_scanned_equal
```

The intended positive pattern is:

1. rolling bases actually change over time;
2. proposed-write-conditioned and generic-current bases are not identical;
3. all accepted local basis contracts are genuinely preserved;
4. `rolling_write_equal` loses fewer unseen cues at matched achieved progress without receiving more scalar measurements.

## What would count against dynamic Q

The hypothesis is weakened if:

- the rolling bases barely change;
- proposed-write conditioning chooses essentially the same basis as the generic current-state panel;
- local contracts are preserved but unseen access is no better than static or rolling-current controls;
- any apparent preservation advantage comes only from blocking learning or spending more measurements.

If the write-conditioned version also fails while its per-write basis is well conditioned and actually preserved, the next escalation should be trajectory/history/state observability rather than another cue-selection rule.

## Claim boundary

The measured signatures remain a black-box proxy for CausalHorizon's exact `Q`; this nonlinear classifier is outside the theorem's exact linear setting. Gate 13 tests only whether making that proxy **current-state and proposed-write dependent** improves capability preservation in this controlled task.
