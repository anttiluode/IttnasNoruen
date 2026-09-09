# Gate 12 findings — a frozen basis is locally informative but not sufficient

Gate 12 was preregistered before the result run in `GATE12_PROTOCOL.md` and executed on independent seeds `293, 307, 331`.

The implementation passed. The strong static-basis hypothesis did not.

## Frozen result

GitHub Actions run: `34371630452`

Source branch head used by the PR run: `c8766d03a421f7b6175a461063af3c815aab42d3`

Artifact: `gate12-receipt`, artifact id `10112168031`

Raw `gate12.json` SHA-256:

```text
33a98967169dd7ccc9488e2f985e299cf633e973a646947c77d8d2bfeeb060d6
```

Artifact ZIP SHA-256:

```text
49e459a598cafe554b4954cf6f8cb422742a9ae38dac1c13dbf7f513306e2726
```

All preregistered checks passed:

- bounded 60-record pool;
- callback budgets respected;
- measured checkpoints valid;
- frozen 20-record basis capacity respected;
- Q-span bank remained exactly two cues per class;
- accepted basis contracts were validated;
- final selected-basis responses remained within tolerance.

The dedicated Gate-12 unit tests also passed.

## What changed from Gates 10 and 11

Gate 10 used measured response geometry to select examples and then kept those examples only above one-sided survival bounds.

Gate 11 froze the local parameter-response fingerprint itself. It successfully held that fingerprint fixed but did not improve unseen access.

Gate 12 tested the cleaner CausalHorizon-motivated condition:

```text
measured geometry
    -> choose a frozen basis
    -> preserve the actual old responses on that basis
```

For each selected cue `j`, the guard retained the baseline margin `m_j(theta_0)` and attempted to keep

```math
\Delta m_j \approx 0.
```

The equality projection in practice kept the selected responses much closer than the preregistered tolerance required.

## The selected Q-span basis really was held almost exactly

For `qspan_equal`, maximum selected-basis drift divided by its allowed tolerance was:

| seed | checkpoint | max tolerance ratio |
| ---: | ---: | ---: |
| 293 | `+0.05` | `1.49e-5` |
| 307 | `+0.05` | `9.49e-9` |
| 307 | `+0.10` | `4.32e-9` |
| 307 | `+0.20` | `5.80e-8` |
| 331 | `+0.05` | `2.80e-7` |
| 331 | `+0.10` | `2.56e-3` |
| 331 | `+0.20` | `2.34e-3` |
| 331 | `+0.40` | `6.54e-4` |

All had zero basis violations.

This is much stronger than merely saying that the selected memories remained correctly classified. Their scalar margins were numerically pinned near their original values while learning proceeded elsewhere.

## Yet unseen access still moved

Final runs:

| seed | method | progress | calls | final unseen loss |
| ---: | --- | ---: | ---: | ---: |
| 293 | boundary_equal | `0.12380` | `26,883` | `0.05828` |
| 293 | qspan_equal | `0.09759` | `27,219` | `0.07925` |
| 307 | boundary_equal | `0.24399` | `26,673` | `0.05742` |
| 307 | qspan_equal | `0.33483` | `26,652` | `0.07656` |
| 331 | boundary_equal | `0.36263` | `26,673` | `0.02929` |
| 331 | qspan_equal | `0.40000` | `21,045` | `0.05858` |

Those final values are not matched-progress comparisons, but they make the structural point visible: the selected basis can remain essentially unchanged while unseen parts of the capability deteriorate.

## Matched progress: a small local signal, then no durable advantage

At `+0.05`, all three seeds give a paired Q-span-equality versus boundary-equality comparison.

Mean `qspan_equal - boundary_equal` differences were:

```text
seen loss   = +0.00000147
unseen loss = -0.00342874
calls       = -714
```

Negative unseen difference favors Q-span. So at the first small checkpoint the measured geometry has a modest descriptive advantage over choosing the fragile boundary cues.

Per seed at `+0.05`:

| seed | boundary unseen loss | Q-span unseen loss | Q-span minus boundary |
| ---: | ---: | ---: | ---: |
| 293 | `0.02797` | `0.02098` | `-0.00699` |
| 307 | `0.01675` | `0.00718` | `-0.00957` |
| 331 | `0.00837` | `0.01464` | `+0.00628` |

So even the first-checkpoint effect is not uniform across seeds.

At `+0.10`, only seeds 307 and 331 provide paired coverage. The mean unseen difference reverses:

```text
qspan_equal - boundary_equal unseen loss = +0.00119117
```

At `+0.20`, the same two seeds give:

```text
qspan_equal - boundary_equal unseen loss = +0.00253749
```

Q-span remains better than boundary on seed 307 but worse on seed 331. There is no robust monotonic advantage as the trajectory moves away from `theta_0`.

## The stronger comparison is against Gate-10 Q-span replay

At matched `+0.05` on seeds 307 and 331:

```text
qspan_equal - qspan_scanned unseen loss = +0.00403395
```

At matched `+0.10`:

```text
qspan_equal - qspan_scanned unseen loss = +0.00702188
```

So pinning the original 20 Q-span basis responses almost exactly was **worse** for unseen access than Gate 10's looser, repeatedly rescanned one-sided Q-span replay at the same achieved progress.

The same differences appear against `all_pool` at those checkpoints.

This does not prove that dynamic reselection is the cause, because the methods also differ in contract type and acquisition structure. But it rules out the simple claim that stronger preservation of a fixed Q-like basis must improve general capability preservation.

## The paths were genuinely different

Q-span-equality and boundary-equality banks overlap heavily—17/20 selected cues on seeds 293 and 307, and 18/20 on seed 331—but the few differing constraints still produce materially different writes.

Parameter distances between the two methods at matched checkpoints were:

| seed | +0.05 | +0.10 | +0.20 |
| ---: | ---: | ---: | ---: |
| 293 | `0.4234` | — | — |
| 307 | `0.1419` | `0.2462` | `0.3673` |
| 331 | `0.3005` | `0.8035` | `1.2731` |

So this is not another identical-trajectory null.

## What the result says

The cleanest counterexample is seed 307.

At `+0.20`, the twenty Q-span basis responses are preserved to a maximum drift of only about

```text
5.8e-8 of their allowed tolerance
```

while unseen-cue loss is still

```text
0.03828.
```

On seed 331 at `+0.40`, the basis is still pinned to about `6.5e-4` of tolerance while unseen loss reaches `0.05858`.

Therefore, in this nonlinear learner:

> **Preserving a fixed, well-measured set of old scalar responses does not imply preservation of the old capability outside that set, even when those responses were chosen for measured span/conditioning.**

That is the important result.

## Why this now points toward a dynamic Q

Gate 10 established that measured response geometry is real: it changes conditioning dramatically.

Gate 11 established that freezing the local parameter-response geometry itself is not sufficient.

Gate 12 establishes that even freezing the **actual responses on a Q-like static basis** is not sufficient.

But Gate 12 also contains a small clue: Q-span is modestly better than boundary selection at the very first `+0.05` checkpoint on average, and that advantage does not persist as learning proceeds.

That is exactly the pattern one would expect if the useful entry geometry is locally meaningful but changes along the learning trajectory.

This is not proof from three seeds, but it motivates a much sharper next object:

```math
Q(\theta, x, h, \Delta\theta)
```

rather than one fixed `Q(theta_0)` or one fixed replay bank.

The candidate write itself should probably enter the measurement protocol. A cue can be irrelevant to one proposed write and critical to another.

## Gate 13 should not be another replay selector

A cleaner next experiment is a **rolling, proposed-write-conditioned causal basis**.

For each atomic update:

1. at the current `theta_t`, first measure the ordinary proposed new-task write `Delta theta_t`;
2. measure old-cue responses under a small panel containing the actual proposed direction plus orthogonal counted perturbations;
3. construct a rank-revealing basis for the directions that this particular write can affect;
4. freeze that basis and its current responses **before** the candidate is applied;
5. compensate/project the candidate to preserve that pre-write basis;
6. validate actual finite responses and commit or reject;
7. only after commit may the next atomic write build a new basis.

The key anti-cheating rule is that the basis for a proposed write is chosen before seeing the accepted post-write state. Re-estimation cannot retroactively redefine damage as irrelevant.

The primary comparison should be:

```text
static qspan_equal
vs
rolling current-state qspan
vs
proposed-write-conditioned rolling qspan
```

at matched progress and matched acquisition budget.

If the proposed-write-conditioned version wins, that would be the first evidence that the missing object is genuinely dynamic rather than merely a better static memory bank.

If it also fails while its per-write basis is well conditioned and actually preserved, then the next escalation is trajectory/history/state observability rather than cue geometry alone.

## Claim boundary

Gate 12 is still one small classifier test bed with three preregistered seeds. The `+0.05` local advantage is descriptive and should not be inflated into a general result.

What is cleanly supported is the negative statement:

1. the Q-span basis was frozen before learning;
2. its selected scalar responses were preserved extraordinarily tightly;
3. the learning trajectories differed materially from the boundary control;
4. unseen cue access nevertheless deteriorated;
5. therefore this fixed scalar-response basis is not sufficient to stand for the old capability in this nonlinear setting.
