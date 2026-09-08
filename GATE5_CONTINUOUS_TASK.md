# Gate 5 — Continuous changing-substrate task

Gate 4 is a privileged mathematical reference. Gate 5 must remove that privilege.

The task must combine the interactions that earlier gates separated:

- material feedback is continuously enabled;
- self-commands continue while external events occur;
- the echo predictor continues learning while material changes;
- several useful responses share adjustable material;
- teaching is derived from measured task error, not scheduled signs;
- delayed feedback can arrive with multiple outstanding experiences;
- receipt memory has a fixed capacity;
- all comparison methods receive the same command copy, windows, measurements and delayed feedback;
- no method receives the protected sensitivity matrix `P` or a hidden causal branch label.

## Reference task

Keep the four material gains and overlapping response bank from Gate 4:

```math
y_1=g_1+g_2,
```

```math
y_2=g_2+g_3,
```

```math
y_3=g_3+g_4,
```

with protected targets all equal to `2`, while

```math
y_{new}=g_1+g_3
```

must move from `2` to `3`.

The privileged Gate 4 solution is

```math
d=[0.5,-0.5,0.5,-0.5].
```

Gate 5 is successful only if a constrained learner can approach the same behavioral result without being handed that direction or the matrix that defines it.

## Candidate methods

Every method gets the same information budget.

1. **Naive local error** — updates only material active in the new task.
2. **Replay protection** — after a proposed update, replay a bounded subset of protected probes and use observed drift to correct the proposal.
3. **Learned sensitivity sketch** — maintain a bounded low-rank estimate of how recent material changes moved protected responses.
4. **Active probe variant** — spend a counted intervention to estimate which change directions are safer.
5. **Private-dither variant** — use the Sigh intervention only for self/world separation, not as free knowledge of the preservation direction.

The experiment should not assume one of these wins.

## Continuous stream

A run should interleave:

```text
self command
external disturbance
protected-task probe
new-task probe
material update
predictor adaptation
quiet interval
late feedback for an earlier receipt
another action before that feedback arrives
```

There should be no global reset of eligibility, predictor or material between episodes unless the reset is itself part of the declared resource budget.

## Teaching signal

For the new task, use actual remaining error:

```math
\delta_{new}=3-y_{new}.
```

For protected tasks, deviation from their retained targets is also measured:

```math
\delta_i=2-y_i.
```

A method may decide how to combine these signals only using information it is explicitly allowed to retain or remeasure.

## Metrics

A method does not pass merely because the new answer improves.

Track:

- new-task squared error;
- maximum and RMS protected-response drift;
- material norm and maximum branch gain;
- echo-prediction error;
- off-target structural write after command/external confounds;
- receipt overflows / dropped feedback under fixed capacity;
- number and energy of active probes;
- recovery after the target relationship reverses;
- stability as feedback delay increases.

A useful scalar score may be reported, but raw components must remain visible so a method cannot hide catastrophic forgetting behind new-task improvement.

## Required attackers

- **No compatible direction:** protect four independent responses in four material dimensions. The learner should report conflict or fail gracefully rather than amplify indefinitely.
- **Receipt overflow:** allow more simultaneously pending consequences than the declared capacity.
- **Command-correlated external event:** naive echo learning should become confounded.
- **World follows private dither:** the dither remedy must lose its causal leverage, matching the known Sigh limit.
- **Longer feedback delay:** an update gain stable at short delay should eventually become oscillatory unless reduced.
- **Representation sharing:** every material coordinate must influence more than one useful response.

## Substantial-result criterion

The interesting result is not that a null-space projection exists; Gate 4 already supplies that mathematics.

A substantial result would be evidence that, under the same bounded signals and memory given to the controls, one mechanism can:

1. infer enough about shared response constraints to make coordinated changes;
2. improve the new task;
3. preserve old useful behavior;
4. keep material magnitude controlled;
5. continue tracking its own changing self-response;
6. assign delayed feedback without unlimited history;
7. expose when the requested change is unidentifiable or incompatible.

That would directly address the program's current question:

> **Can a system learn which coordinated changes its own material can tolerate—and use that knowledge while continuing to act through the material it is changing?**
