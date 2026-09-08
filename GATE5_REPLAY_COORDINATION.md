# Gate 5 — Replay-driven candidate plasticity

Gate 4 answered the compatibility question with a privileged matrix `P`:

```math
P\Delta g = 0,
```

while the new task must still change:

```math
p^T\Delta g \ne 0.
```

Gate 5 removes that main privilege in the present four-gain model. The learner is **not** handed `P`, its null space, or the final direction `[0.5,-0.5,0.5,-0.5]`.

Instead it reconstructs the needed compatibility operation one protected question at a time through replay.

## The local measurement shortcut

For the current response law

```math
x_j = g_j(s_j u + v_j)
```

and fixed soma readout

```math
y = \sum_j c_j x_j,
```

a protected replay is performed with `u=0`. The self predictor therefore contributes no return during that measurement and

```math
r_j = x_j = g_j v_j.
```

The local sensitivity of the soma answer to branch gain is

```math
a_j = \frac{\partial y}{\partial g_j}=c_jv_j.
```

Because branch `j` knows its current gain and receives its local residual, the same quantity can be recovered as

```math
a_j = c_j\frac{r_j}{g_j}.
```

This is a shortcut provided by the diagonal gain model. It is not assumed to remain available in a coupled nonlinear dendrite.

## Temporary proposal instead of immediate material change

Each branch gets one extra fast scalar:

```math
z_j = \text{proposed gain change at branch }j.
```

The new question initializes the proposal from its measured local sensitivity.

A protected replay then gives local sensitivity `a_j`. Each branch forms the local product

```math
a_j z_j.
```

Those products are summed:

```math
h=\sum_j a_jz_j.
```

The squared sensitivities are also summed:

```math
n=\sum_j a_j^2.
```

If `n>0`, one correction scalar is broadcast and every branch applies

```math
z_j\leftarrow z_j-a_j\frac{h}{n}.
```

This removes the component of the current proposal that would change the replayed answer.

Because fixing one protected response can disturb another, the system cycles through the retained questions repeatedly. In matrix language each replay applies

```math
F_i=I-\frac{a_i a_i^T}{\lVert a_i\rVert^2},
```

but the implementation never constructs the matrix `P` or multiplies by a stored projector. It obtains one row through physical replay, performs local multiplications, aggregates two scalars and broadcasts one correction.

## Deterministic Gate 5 result

The protected questions are

```math
y_1=g_1+g_2,
```

```math
y_2=g_2+g_3,
```

```math
y_3=g_3+g_4,
```

and the new question is

```math
y_{new}=g_1+g_3.
```

Starting at

```math
g=[1,1,1,1],
```

20 cycles through the three protected questions cost **60 protected replays**.

The learned temporary proposal becomes approximately

```text
[+0.50000036, -0.49999988, +0.49999964, -0.49999964]
```

without receiving the privileged matrix `P`.

The evaluator is still allowed to compute the Gate 4 reference after the fact. Cosine similarity between the learned proposal and the privileged reference is greater than `0.9999999999998`.

A unit task-error commit changes the new answer from `2` to `3` while maximum protected-answer drift is approximately

```text
4.77e-7
```

from the finite number of replay projections.

The dense replay bank explicitly stores three 4-coordinate probe descriptions plus three reference answers:

```text
3 * (4 + 1) = 15 scalar values
```

and the temporary proposal costs another four scalars.

## Task error stops the material-growth loop

Gate 5 also keeps the task-error teaching rule from Gate 4:

```math
\delta=3-y_{new}.
```

With a `0.5` commit rate, 20 structural updates leave

```text
new response       ~= 2.99999905
remaining error    ~= 9.54e-7
max old drift      ~= 4.77e-7
max branch gain    ~= 1.50
```

rather than the runaway gain seen under endlessly scheduled positive teaching.

During those same updates a self command is issued and the echo predictor keeps adapting. Starting without a separately scheduled post-write relearning phase, its final deterministic self-prediction error is about

```text
0.00612
```

after 20 online updates. This is only a tracking smoke test, not yet the command-confounded continuous attacker.

## The direction is not hard-coded

Changing the fixed readout contributions to

```text
[0.7, 1.1, 0.9, 1.3]
```

changes the compatible proposal to approximately

```text
[+0.71428571, -0.45454545, +0.55555556, -0.38461538]
```

using the same replay algorithm. It again matches the privileged evaluator reference and preserves the three measured old answers to below `1e-6` in the deterministic gate.

This control matters because the alternating signs are not stored as the answer. They are assembled from the current replayed response geometry.

## Failure controls

### Full-rank protection

If four independent responses protect all four material directions, repeated replay drives the proposal norm to numerical zero.

The learner should therefore expose **conflict / no compatible change**, not keep amplifying material.

### Omit one protected question

If the third protected response is never replayed, the learner cannot magically protect it. In the current construction, that omitted answer drifts by approximately

```text
0.5
```

after the new task is solved.

Protection costs information.

### Write first, repair afterward

A comparison method changes the actual gains immediately and then alternates repairs of old and new answers.

It eventually reaches the same compatible solution in this linear system, but the old response bank temporarily moves by

```text
0.5
```

before repair.

The temporary proposal therefore has a concrete role:

> **calculate a coordinated change in fast state before exposing slow material to it.**

That is the direct bridge back to Sigh: fast repeated operators remove incompatible components of a candidate write; only the surviving component consolidates.

## What machinery did we add?

| resource | explicit cost / source |
|---|---|
| local replay sensitivity | local replay residual, current local gain, fixed readout coupling |
| temporary proposal | one scalar `z_j` per branch |
| coordination | local `a_j z_j`, scalar sums `h` and `n`, one broadcast correction scalar |
| protected history | replayable probe description plus reference answer |
| replay bandwidth | counted physical replays; 60 in the reference acquisition |
| slow write permission | measured new-task error at commit time |
| delayed pending credit | **not integrated here**; existing receipt mechanism remains separate |

This is not free local learning. The scalar aggregation/broadcast pathway is additional machinery. The replay descriptions are memory. The current sensitivity shortcut depends on the simple gain law.

## What has actually been resolved?

Before Gate 5, the repo knew that a compatible direction existed only because Gate 4 was handed `P`.

Gate 5 gives the current small system an executable way to **acquire enough of that constraint information through interaction**:

```text
new task produces candidate write
            |
            v
      temporary proposal z
            |
old question replay -> local sensitivities -> scalar incompatibility
            |                                  |
            +---------- correction <------------+
            |
        replay again
            |
            v
proposal invisible to retained old questions
            |
      task error authorizes commit
            |
            v
        slow material
```

The central operation is therefore no longer missing in the four-gain model.

## Why it is still a mechanism prototype

Passing Gate 5 does **not** establish a general learning neuron.

The current response law is instantaneous and diagonal. Replays use zero self-command to make sensitivity measurement clean. The protected probes and their reference answers are replayable. The readout coupling and local gains are available. Noise, finite sensitivity-estimation error, command-correlated external events, delayed receipt feedback and coupled nonlinear propagation are not all active simultaneously.

So the correct status is:

> **Replay-driven compatible plasticity is now an implemented mechanism in the present model, not merely a privileged mathematical reference.**

The next substantial test is to make that mechanism survive the continuous changing-substrate task under the same bounded information budget as its controls.
