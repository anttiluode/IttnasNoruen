# Gate 13 findings — rolling local protection helps; proposed-write conditioning is only a small early signal

Gate 13 was preregistered in `GATE13_PROTOCOL.md` before the result run and executed on independent seeds `347, 359, 373`.

The implementation passed all audits. The result is mixed but materially more interesting than another static-basis null.

## Frozen result

GitHub Actions run: `34373316487`

Source branch head used by the PR run: `dc7d951378a31fae46bdb42b8c47b0a939ff864d`

Artifact: `gate13-receipt`, artifact id `10112838734`

Raw `gate13.json` SHA-256:

```text
ce22efba20520e2746f4f587ffa5f5a05f4ef06b6fab4dedaa7703cfe68a283b
```

Artifact ZIP SHA-256:

```text
43af90aab158f0a3438b2397fe879d7ca563ee0feaca438cef97223dae45418a
```

The dedicated Gate-13 tests passed (`5 passed`) and the normal repository CI passed as well.

All experiment checks passed:

- callback budgets respected;
- 60-record pool preserved;
- checkpoint progress measured rather than interpolated;
- active basis stayed at 20 cues;
- every rolling basis remained two cues per class;
- the basis was chosen before the protected candidate was generated/observed;
- accepted local equality contracts were validated;
- the first axis of the write-conditioned panel was the measured proposal direction.

## The three methods

`static_scanned_equal` reused the Gate-12-style frozen Q-span basis and its `theta_0` response anchors, but still paid for ignored fresh scans on every attempt.

`rolling_current_equal` rebuilt a Q-span basis from fresh generic response geometry at the current parameter state before every atomic write and anchored those cues to their current pre-write responses.

`rolling_write_equal` did the same, except the first probe axis was the actual ordinary proposed write direction for that atomic update.

Thus the static method did **not** win any acquisition-budget discount from remaining static.

## First result: the rolling bases really moved

The frozen control had exactly one basis on every seed.

The rolling methods used:

| seed | rolling current distinct bases | rolling write distinct bases |
| ---: | ---: | ---: |
| 347 | 13 | 16 |
| 359 | 23 | 23 |
| 373 | 23 | 23 |

Mean Jaccard overlap with the original frozen basis was:

| seed | rolling current | rolling write |
| ---: | ---: | ---: |
| 347 | 0.791 | 0.833 |
| 359 | 0.706 | 0.753 |
| 373 | 0.739 | 0.754 |

So the current measured geometry was not just rediscovering the same memory bank.

## Second result: rolling local protection strongly beat the frozen basis

The final states are not matched-progress comparisons, but they show the scale of the difference:

| seed | method | progress | calls | final unseen loss |
| ---: | --- | ---: | ---: | ---: |
| 347 | static | 0.4000 | 40,533 | 0.07965 |
| 347 | rolling current | 0.4000 | 25,367 | 0.03761 |
| 347 | rolling write | 0.4000 | 31,112 | 0.03761 |
| 359 | static | 0.3266 | 45,000 | 0.09506 |
| 359 | rolling current | 0.2951 | 45,000 | 0.04936 |
| 359 | rolling write | 0.2000 | 45,000 | 0.03473 |
| 373 | static | 0.0746 | 45,000 | 0.06238 |
| 373 | rolling current | 0.1283 | 45,000 | 0.03214 |
| 373 | rolling write | 0.1708 | 45,000 | 0.04726 |

The cleaner matched-progress comparison is also favorable to rolling-write versus static.

At `+0.05`, all three seeds paired:

```text
rolling_write - static unseen loss = -0.0140603
```

At `+0.10`, seeds 347 and 359 paired:

```text
rolling_write - static unseen loss = -0.0190965
```

At `+0.20`, seeds 347 and 359 paired:

```text
rolling_write - static unseen loss = -0.0242433
```

On seed 347 at `+0.40`:

```text
rolling_write - static unseen loss = -0.0420354
```

Negative favors rolling-write.

This is the first gate in this sequence where abandoning the fixed `theta_0` protection object is associated with a large and persistent capability-preservation improvement.

## Third result: proposed-write conditioning is only a modest early improvement over ordinary rolling geometry

The preregistered primary comparison was:

```text
rolling_write_equal - rolling_current_equal
```

At `+0.05`, all three seeds paired:

```text
seen loss difference   = +0.0007067
unseen loss difference = -0.0006753
call difference        = +626.7
```

The unseen effect is tiny. Per seed, write-conditioning was better on 347 and 359 and worse on 373.

At `+0.10`, all three seeds again paired:

```text
seen loss difference   = 0.0000000
unseen loss difference = -0.0042760
call difference        = -626.7
```

Per seed unseen losses were:

| seed | rolling current | rolling write | write minus current |
| ---: | ---: | ---: | ---: |
| 347 | 0.01991 | 0.00885 | -0.01106 |
| 359 | 0.02194 | 0.01828 | -0.00366 |
| 373 | 0.02647 | 0.02836 | +0.00189 |

So two of three seeds favored proposed-write conditioning at `+0.10`, but not all three.

At `+0.20`, seeds 347 and 359 paired and the unseen difference was exactly:

```text
0.0000000
```

At `+0.40`, only seed 347 paired and the unseen difference was again zero.

Therefore the strong version of the proposed-write-conditioned hypothesis is **not supported** by Gate 13. The actual write direction gives a small early signal, but no durable advantage over simply rebuilding the basis at the current state.

## The write-conditioned panel was not vacuous

Across runs, the Q-span selector enriched cues responsive to the panel's first axis. Mean selected-versus-pool absolute first-axis response ratios for the write-conditioned runs were approximately:

```text
seed 347: 2.22x
seed 359: 2.66x
seed 373: 2.77x
```

So the write axis materially affected the measured geometry. Yet that extra conditioning did not buy a later preservation advantage over rolling-current geometry.

## The important interpretation

Gates 10–12 progressively ruled out increasingly strong **static** objects:

```text
static example selection
    -> insufficient
static parameter-response fingerprint
    -> insufficient
static Q-like basis with actual responses pinned
    -> insufficient
```

Gate 13 now says something new:

> **A protection object rebuilt at the current state can outperform a fixed protection object by a large margin.**

But the data do **not** yet justify the stronger sentence:

> the exact proposed write direction is the missing Q.

The write-conditioned basis only improved the ordinary rolling basis modestly and transiently.

## A confound we should not hide

Gate 13 intentionally changed two things together when moving from static to rolling:

1. **basis selection** became current-state dependent;
2. **response anchors** became local pre-write values rather than permanent `theta_0` values.

Therefore the large rolling-versus-static advantage cannot yet be attributed specifically to basis reselection.

It could be that the crucial mechanism is simply:

```text
protect local change now
rather than
force a finite set of old scalar values to remain globally frozen forever
```

That distinction matters.

A clean next experiment should separate these factors:

```text
frozen basis + rolling pre-write anchors
vs
rolling basis + rolling pre-write anchors
vs
write-conditioned rolling basis + rolling pre-write anchors
```

If frozen-basis/rolling-anchor already gets the benefit, then the important object is not dynamic `Q` selection but a local differential preservation law.

If rolling basis still wins at matched progress and acquisition, then the evidence for genuinely state-dependent entry geometry becomes much stronger.

## Claim boundary

This remains one small nonlinear classifier family with three resampled splits, and the signatures remain only a black-box proxy for CausalHorizon's exact event-entry map.

What Gate 13 cleanly supports is narrower:

1. the rolling basis changed substantially over learning;
2. all local protected contracts were genuinely enforced;
3. both rolling methods preserved unseen access much better than the fixed Gate-12-style basis in these runs;
4. explicit proposed-write conditioning added only a small early advantage over generic current-state rolling geometry;
5. the experiment does not yet separate dynamic basis selection from rolling pre-write anchoring.
