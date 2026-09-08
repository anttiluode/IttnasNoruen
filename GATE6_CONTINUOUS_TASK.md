# Gate 6 — Continuous changing-substrate task

**Implementation status:** [`gate6_continuous_learning.py`](gate6_continuous_learning.py)
now exercises this contract in the gain model. See [`RESULTS.md`](RESULTS.md) for
the matched stream, negative results and counted resources. Windows and version IDs
are supplied; material is fixed within a four-tick balanced intervention; digital
receipts do not decay; replay sensitivities occupy a bounded measured sketch.
The coupled nonlinear experiment is separate and is not claimed to satisfy this
continuous local-interface contract.

Gate 4 is the privileged mathematical reference. Gate 5 supplies an implemented replay-driven compatibility mechanism in the present diagonal gain model. Gate 6 must now force that mechanism to survive the interactions that earlier gates still separate.

The task must combine:

- material feedback continuously enabled;
- self-commands continue while external events occur;
- the echo predictor continues learning while material changes;
- several useful responses share adjustable material;
- teaching is derived from measured task error, not scheduled signs;
- delayed feedback can arrive with multiple outstanding experiences;
- receipt memory has a fixed capacity;
- replay memory has a fixed capacity and replay bandwidth is counted;
- all comparison methods receive the same command copy, windows, measurements and delayed feedback;
- no method receives the protected sensitivity matrix `P`, its null space, or a hidden causal branch label.

## Reference task

Keep the four material gains and overlapping response bank:

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

Gate 4's privileged direction is

```math
d=[0.5,-0.5,0.5,-0.5].
```

Gate 5 already shows that replay can approximate this direction without receiving `P` when replay sensitivity measurements are clean.

Gate 6 is successful only if a constrained learner can retain the behavioral benefit while self-prediction, changing material, delayed credit, finite memory and overlapping external activity are all active.

## Candidate methods

Every method gets the same information budget.

1. **Naive local error** — updates only material active in the new task.
2. **Replay-coordinated proposal** — Gate 5 mechanism with bounded protected replay.
3. **Write-then-repair** — modifies real material immediately and repairs retained behaviors afterward.
4. **Learned sensitivity sketch** — bounded low-rank estimate of how material changes move protected responses.
5. **Active probe variant** — spends counted interventions to estimate safer change directions.
6. **Private-dither variant** — uses the Sigh intervention for self/world separation only, not as free knowledge of the preservation direction.

The experiment should not assume Gate 5 wins.

## Continuous stream

A run should interleave:

```text
self command
external disturbance
protected-task replay
new-task probe
temporary proposal correction
material update
predictor adaptation
quiet interval
late feedback for an earlier receipt
another action before that feedback arrives
```

There should be no global reset of eligibility, predictor or material between episodes unless that reset is itself counted.

## Teaching signal

For the new task, use actual remaining error:

```math
\delta_{new}=3-y_{new}.
```

For protected tasks, deviation from retained targets is also measurable:

```math
\delta_i=2-y_i.
```

A method may decide how to combine those signals only using information it is explicitly allowed to retain or remeasure.

## Replay accounting

The Gate 5 dense replay representation stores one N-coordinate probe plus one scalar reference answer per protected question.

For `M` protected questions:

```math
M(N+1)
```

scalar values are retained, apart from IDs/bookkeeping.

Replay also consumes actions and time. Track:

- replay count;
- replay probe energy;
- replay age / staleness;
- protected memories dropped when the replay bank is full.

A result does not count if it preserves old behavior using unlimited replay bandwidth or unlimited protected-task memory.

## Metrics

Track raw components, not only a scalar score:

- new-task squared error;
- maximum and RMS protected-response drift;
- material norm and maximum branch gain;
- echo-prediction error;
- off-target structural write after command/external confounds;
- receipt overflows / dropped feedback under fixed capacity;
- replay-memory overflows / dropped protected behavior;
- replay count and probe energy;
- number and energy of active causal probes;
- recovery after the target relationship reverses;
- stability as feedback delay increases.

## Required attackers

- **No compatible direction:** protect four independent responses in four material dimensions. The learner should report conflict or fail gracefully rather than amplify indefinitely.
- **Receipt overflow:** allow more simultaneously pending consequences than the declared capacity.
- **Replay overflow:** require more protected behaviors than the declared replay bank can retain.
- **Command-correlated external event:** naive echo learning should become confounded.
- **World follows private dither:** the dither remedy must lose causal leverage, matching the known Sigh limit.
- **Longer feedback delay:** an update gain stable at short delay should eventually become oscillatory unless reduced.
- **Stale replay sensitivity:** change readout/response geometry after proposal acquisition and measure resulting protected drift.
- **Representation sharing:** every material coordinate must influence more than one useful response.

## Substantial-result criterion

The interesting result is no longer that a null-space projection exists, nor that replay can implement it in a clean diagonal gain model.

A substantial result would be evidence that, under the same bounded signals and memory given to controls, one mechanism can:

1. infer enough about shared response constraints to make coordinated changes;
2. improve the new task;
3. preserve old useful behavior;
4. keep material magnitude controlled;
5. continue tracking its own changing self-response;
6. assign delayed feedback without unlimited history;
7. protect behavior without unlimited replay memory or replay bandwidth;
8. expose when the requested change is unidentifiable or incompatible.

That directly addresses the program's current question:

> **Can a system learn which coordinated changes its own material can tolerate—and use that knowledge while continuing to act through the material it is changing?**
