# Gate 14 — local preservation law or dynamic Q geometry?

Gate 13 produced the first large improvement in this sequence: protection rebuilt at the current state preserved unseen cue access much better than a globally frozen `theta_0` basis. But Gate 13 changed two things at once:

1. the selected basis could change with the current state;
2. protected scalar responses were re-anchored to their current pre-write values rather than permanently frozen at `theta_0`.

Gate 14 separates those mechanisms.

## Question

Is the Gate-13 gain caused by discovering a state-dependent Q-like basis, or is the important rule simply:

> preserve the current old capability locally across each write, rather than demand that a few old scalar responses remain globally fixed forever?

## Preregistered methods

Every method uses rolling **pre-write response anchors**. Every method also pays for the same per-attempt ordinary proposal and the same fresh 12-direction signature scan. The only difference is which 20 old cues become the protected basis for that write.

### `frozen_qspan_rolling_anchor`

- Select one Q-span basis at `theta_0`.
- Keep those same 20 cue identities for the entire run.
- Before each atomic write, remeasure their **current** responses and use those current values as equality anchors.
- Pay for the same fresh 12-direction scan as the other methods, but ignore it for basis selection.

This isolates the effect of rolling anchors without dynamic basis selection.

### `random_rolling_anchor`

- Before each atomic write, select a fresh deterministic random 20-cue basis, exactly two cues per class.
- Anchor those selected cues to their current pre-write responses.
- Pay for and ignore the same fresh response-signature geometry.

This tests whether simple basis churn / time-distributed coverage explains any benefit of moving the protected set.

### `qspan_rolling_anchor`

- Before each atomic write, use the fresh current-state 12-direction response signatures to select a Q-span basis, exactly two cues per class.
- Anchor those selected cues to their current pre-write responses.

This is the clean dynamic-geometry condition.

## Fixed design

- Dataset/model/task family: unchanged handwritten-digit system from Gates 9–13.
- Independent seeds: `389, 401, 419`.
- Stored candidate pool: 60 cues.
- Active protected basis: 20 cues, exactly two per class.
- Adjustable parameters: 24 hidden-bias offsets.
- Progress checkpoints: `+0.05, +0.10, +0.20, +0.40`.
- Per-method scalar callback budget: `45,000`.
- Attempt cap: `24`.
- Signature probes: `12` deterministic orthonormal directions per attempt.
- Signature amplitude: `epsilon = 0.03`.
- Equality tolerance: `max(0.002, 0.05 * abs(current_pre_write_margin))`.
- Same equality guard, finite candidate validation, and trust radius as Gates 12–13.

## Equal acquisition rule

For every atomic attempt, all three methods pay for:

1. an ordinary unprotected proposal toward the current new-task target;
2. all 60 current old-cue margins;
3. all 60 old-cue responses under 12 temporary orthonormal parameter probes;
4. the protected equality-guard measurements and rejected candidates.

The frozen and random methods may ignore the signature geometry, but ignored measurements still cost calls.

The basis and current response anchors must be fixed **before** the protected candidate is generated or observed.

## Primary comparison

At matched achieved progress:

```text
qspan_rolling_anchor - frozen_qspan_rolling_anchor
```

Primary outcome: unseen-cue loss among initially correct unseen masks. Negative favors dynamic Q-span selection.

## Secondary comparisons

```text
qspan_rolling_anchor - random_rolling_anchor
random_rolling_anchor - frozen_qspan_rolling_anchor
```

These distinguish geometry from mere rotation of the protected set.

## Interpretation matrix

### If frozen ≈ random ≈ qspan

The Gate-13 gain came mainly from **rolling local anchors**. The useful rule is differential/local preservation, not dynamic Q selection.

### If random ≈ qspan < frozen

Moving the protected set matters, but Q-like geometry is not special. The mechanism is likely broader temporal coverage / rotating replay.

### If qspan < random ≈ frozen

Current-state response geometry is doing specific useful work. This is the strongest evidence so far for a state-dependent Q-like protection basis.

### If qspan < random < frozen

Both rotation and measured geometry contribute.

## Additional diagnostics

Record for every accepted or rejected attempt:

- selected cue IDs;
- overlap with the original frozen Q-span basis;
- overlap with the previous attempt's basis;
- measured signature rank/condition/residual for the selected basis;
- maximum local equality violation;
- scalar callback cost;
- achieved new-task progress;
- seen and unseen access only at evaluator checkpoints.

## Claim boundary

This remains a black-box proxy experiment on a small nonlinear classifier. The measured signature is not asserted to be CausalHorizon's exact `Q`. Gate 14 tests only whether **state-dependent measured basis selection** adds preservation value once every method already follows the same rolling local response-preservation law.
