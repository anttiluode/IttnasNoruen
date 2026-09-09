# Gate 9 — nonlinear compensation helps; replay selection remains unproven

[Generated results](SELECTED_REPLAY_RESULTS.md) and [protocol](GATE9_PROTOCOL.md)
record the complete comparison. The learner can now choose replay records based
on measured consequences of a temporary update. This does not establish better
protection of unseen cue routes.

## Result of the intended comparison

The informed selector reaches the +0.05 task checkpoint on all three new splits.
Against acquisition-matched random selection, its loss rate on previously correct
held-out cues is higher by 0.071 percentage points for familiar masks and 0.069
points for unseen masks. These tiny descriptive differences establish no benefit.
At later checkpoints only one seed supports a paired comparison with the informed
selector. Missing progress is displayed rather than replaced with a protection win.

Stored-pool recognition can remain intact while unseen cue responses fail. The
informed method loses zero of its 60 stored classifications in these runs, while
its final held-out losses average five familiar-mask and 3.67 unseen-mask responses.
Those final states have unequal new-task progress across methods and must not be
used as a matched-learning ranking.

The random method nearly reaches +0.40 in all three seeds. Its seed-151 progress
is 0.399951, which narrowly misses the fixed 1e-6 checkpoint tolerance. That is a
strict reporting miss, not evidence of a large learning deficit. The raw progress
and stop reason are retained.

## A distinct mechanism failure exposed during the run

The first run made several guards stop at identical points. The recorded audit
found that Dykstra and an independent SLSQP solve agreed on the linear subproblem
([runnable audit](audit_replay_boundary.py)).
The candidate still violated a true nonlinear margin by about 0.02745. The
projection had not failed; the local linear model was inadequate for that step.

This exact example exposes the issue:

```math
\text{protected response}=b-a^2\geq0,\qquad
\text{new response}=a.
```

At `(a,b)=(0,0)`, the linear model permits increasing `a` with `b=0`. Every nonzero
such step violates the real protected response. A feasible write exists:
`a=0.1, b=0.01`. The learner must supply the compensating change in `b`.

The guard now uses a rejected trial to measure

```math
e_{\rm model}=f_{\theta+z}(q)-f_\theta(q)-\hat J_\theta(q)z.
```

It shifts its local prediction by that discrepancy, reprojects the temporary
step, and measures the new actual candidate. It allows at most two corrections
before shortening the new-task step. The original acceptance limits never move.
Every trial is charged and an exhausted budget returns unchanged parameters.

This is a finite nonlinear model correction, not a new biological credit signal
or a general convergence guarantee. It uses software's ability to evaluate
temporary parameter vectors. It can still fail when curvature, nonsmooth margins,
conflicting requirements or measurement limits defeat the local search.

## What the repair changes

For the fixed mixed guard on seed 179, progress rises from about 0.082 to 0.40 and
scalar calls fall from 7,812 to 6,363. On the harder seeds, progress improves but
the same budget still limits learning. The initially observed failure and the
corrected run are both preserved. The correction was made after inspecting the
first outcomes; the data, selectors, checkpoints and task settings were not tuned.

This separates three questions that otherwise looked like one:

1. Did we select a useful old question?
2. Did its measured sensitivity predict the finite update adequately?
3. Did the retained questions cover the useful behavior outside the bank?

Gate 9 improves the second mechanism. It does not resolve the first or third.

## What Kompressori contributes

Kompressori warns that preserving known responses can miss altered susceptibility
to future inputs. Here the related practical lesson is that a first-order
preserving direction need not define a finite preserving write. The derivatives
are different: Kompressori studies input-response geometry; this correction
models changes with respect to adjustable parameters. The connection is a shared
need to test the actual finite response, not an identity between the two operators.

Selection by anticipated interference is established prior work (MIR). Low-rank
overlap is not tested here and cannot replace behavioral validation by assertion.
A future compact sketch must show that it predicts consequential interference on
independent directions, under an explicit cost allowance.

The current common cap is **scalar callbacks**, not equal compute. New-task
callbacks evaluate 64 images and old-cue callbacks evaluate one. The report also
counts example evaluations; scanned selection can be substantially more expensive.
These are three splits of one small dataset, with externally supplied labeled
memory and cue generators. Recall-time cue construction and continuous self/world
credit remain separate experiments.
