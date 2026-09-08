# IttnasNoruen

**Expectation + delayed credit + replay + candidate plasticity + slow material.**

This repo is the integration point for the recent Sigh / Jello / Child / FunctionalArbors / Active Dendrite line. It is not a claim that we have built a biological neuron. It is a deliberately small executable object that asks a sharper question:

> **Under what information and memory constraints can a system keep learning about—and through—a substrate that its own learning continually changes?**

The current answer has four separate requirements:

1. **Distinguishability** — available actions and measurements must separate causes that require different updates.
2. **History** — enough information must survive until delayed feedback arrives.
3. **Compatibility** — there must be an achievable material change that improves the new response while preserving responses that still matter.
4. **Stability** — delayed feedback must drive that change slowly enough for the learning loop to converge.

Gate 5 now implements a concrete mechanism for item 3 in the present four-gain model: **replay old questions to reshape a temporary proposal for plasticity before committing it to slow material.**

## The small changing substrate

For branch-like compartment `j`:

```math
\hat x_j = k_j u
```

```math
r_j=x_j-\hat x_j
```

```math
e_j\leftarrow\gamma e_j+r_j
```

```math
k_j\leftarrow k_j+\eta_k u r_j
```

Slow material changes only when a signed teaching signal authorizes a write:

```math
w_j\leftarrow w_j+\eta_w\delta e_j.
```

Material changes future physical response through the toy gain

```math
g_j=1+\beta w_j
```

and

```math
x_j=g_j(s_j u+v_j).
```

This gain law is intentionally not a biophysical conductance model. Its job is to force the important coupling:

> **Writing material changes the response that the self-predictor must subsequently explain.**

A fixed soma readout measures future behavior rather than inspecting `w` directly.

## Gate 0 — Efference copy cleans delayed credit

A predictable self-command echo and an external event occur together. With a learned efference-copy predictor, predictable self activity is largely removed from the eligibility used by delayed teaching. Without it, command-generated activity contaminates the structural write.

This is a historical control with clean self-only calibration.

## Gate 1 — Surprise is not permission to write

Change the physical self-response so the echo predictor becomes wrong. The predictor adapts, but without a delayed teaching event the slow material stays fixed.

This establishes only:

> **prediction error may train a fast self-model without immediately becoming structural memory.**

Astra's attacker then showed that the same residual can remain eligible and be written later by an unrelated consequence.

## Gate 2 — Addressed reversal reference

One stored coordinate is reversed while another remains unchanged. This remains an intentionally easy addressed control.

## Gate 3 — Material earns meaning; receipts separate pending windows

Gate 3 closes the material-to-behavior loop and exposes a second credit problem.

A single aggregate eligibility vector answers:

> what unexplained activity is still around?

It does not answer:

> which earlier activity is this particular delayed consequence about?

Gate 3 therefore adds bounded **receipts**: retained temporal/action windows whose delayed feedback names the receipt, not a branch label. Separate pending receipts survive separate consequences.

The declared limitation is important: a receipt supplies a window identity. It does not solve arbitrary causal ambiguity inside that window.

`IttnasNoruen(max_receipts=K)` makes this memory explicit. With `N` branches and `K` outstanding receipts, the raw receipt traces contain `N*K` scalar values plus IDs/bookkeeping.

## Gate 4 — Exact compatible-change reference

The next wall was not credit attribution. Even after identifying the right experience, **what may the system change without damaging something else?**

Four gains begin at

```math
g=[1,1,1,1].
```

Three overlapping old responses must remain `2`:

```math
y_1=g_1+g_2,
```

```math
y_2=g_2+g_3,
```

```math
y_3=g_3+g_4.
```

A new response

```math
y_{new}=g_1+g_3
```

must move from `2` to `3`.

If protected sensitivities are rows of `P`, a safe local change must satisfy

```math
P\Delta g=0
```

while the new task requires

```math
p^T\Delta g\ne0.
```

The privileged reference computes

```math
d=[0.5,-0.5,0.5,-0.5].
```

Using remaining task error

```math
\delta=3-y_{new}
```

and

```math
g\leftarrow g+\eta\delta d
```

makes learning stop as behavior becomes correct instead of producing the runaway positive-teaching amplification loop.

Gate 4 also includes a full-rank impossibility control and the delayed-feedback stability reference

```math
\delta_{t+1}=\delta_t-a\delta_{t-D},
```

whose restricted stability boundary is

```math
0<a<2\sin\left(\frac{\pi}{4D+2}\right).
```

Detailed derivation: [`COMPATIBLE_CHANGE_REFERENCE.md`](COMPATIBLE_CHANGE_REFERENCE.md).

## Gate 5 — Replay-driven candidate plasticity

Gate 4 was still privileged because it received `P`. Gate 5 removes that main privilege in the current gain model.

### Local sensitivity from replay

During a protected replay with zero self-command,

```math
r_j=g_jv_j.
```

For fixed readout

```math
y=\sum_j c_j g_j v_j,
```

branch `j` can recover its local gain sensitivity as

```math
a_j=c_j\frac{r_j}{g_j}.
```

The learner therefore obtains **one protected sensitivity row at a time through replay**. It is never handed the complete matrix.

This shortcut is specific to the diagonal gain law.

### Fast proposal, slow commit

Each branch gets one temporary scalar `z_j`, a proposed gain change.

A protected replay forms

```math
h=\sum_j a_jz_j
```

and

```math
n=\sum_j a_j^2.
```

Then one scalar correction is broadcast:

```math
z_j\leftarrow z_j-a_j\frac{h}{n}.
```

Cycling through old questions repeatedly removes proposal components that would disturb them. Only afterward does task error authorize a slow material commit.

This is the computational bridge to Sigh:

> **fast repeated operators remove incompatible components of a candidate write; only the surviving component consolidates.**

### Deterministic receipt

Twenty cycles through the three protected questions cost **60 protected replays** and recover approximately

```text
[+0.50000036, -0.49999988, +0.49999964, -0.49999964]
```

without giving the learner `P` or the privileged direction.

Against the evaluator's Gate 4 reference, cosine similarity is greater than `0.9999999999998`.

A unit task-error commit moves the new response from `2` to `3` while maximum protected-response drift is about

```text
4.77e-7
```

from the finite replay count.

The dense protected replay bank costs

```text
3 * (4 probe coordinates + 1 reference answer) = 15 scalar values
```

plus IDs/bookkeeping. The temporary proposal costs another four scalars.

With error-driven commits at rate `0.5`, after 20 updates:

```text
new response       ~= 2.99999905
remaining error    ~= 9.54e-7
max old drift      ~= 4.77e-7
max branch gain    ~= 1.50
self-echo error    ~= 0.00612
```

The echo predictor is updated during the same sequence rather than in a separate post-write relearning phase.

### It is not a hard-coded alternating-sign trick

Changing the fixed readout contributions to

```text
[0.7, 1.1, 0.9, 1.3]
```

makes the same replay algorithm discover approximately

```text
[+0.71428571, -0.45454545, +0.55555556, -0.38461538]
```

and it again matches the privileged evaluator reference.

### Attackers

- **Full-rank protection:** four independent protected responses drive the proposal to zero. The system exposes conflict rather than amplifying forever.
- **Omitted replay:** leave one old question out and that untested answer drifts by about `0.5`.
- **Write then repair:** modifying real material first and repairing afterward eventually converges in this linear system, but old behavior moves by `0.5` transiently. The temporary proposal avoids exposing the slow material to that intermediate damage.

Full mechanism, accounting and caveats: [`GATE5_REPLAY_COORDINATION.md`](GATE5_REPLAY_COORDINATION.md).

Run:

```bash
python gates.py
python gate4_compatible_change.py
python gate5_replay_coordination.py
```

## What Gate 5 changes conceptually

The program has accumulated four separations:

> **surprise is not permission to write.**

> **eligibility that survives is not necessarily eligibility that belongs to the consequence that arrived.**

> **knowing which experience received feedback is not knowing which coordinated material change will preserve other useful behavior.**

> **a compatible change can be calculated in temporary fast state before slow material is exposed to it.**

That last operation was missing before Gate 5. In this model it is now executable rather than merely supplied by an oracle matrix.

## Explicit resources and remaining privileges

| quantity | source here | remaining restriction |
|---|---|---|
| command copy `u` | explicit efference-copy input | physical pathway not modeled |
| local activity `x_j` | branch-local observable | instantaneous diagonal response |
| predicted self echo | learned local coefficient `k_j` | command-correlated world remains an attacker |
| residual `r_j` | local subtraction | comparator is supplied |
| receipt trace | declared action/time window | causal ambiguity inside a window remains |
| receipt capacity | explicit `max_receipts` | no learned compression/eviction |
| temporary proposal `z_j` | one fast scalar per branch | additional state |
| replay sensitivity `a_j` | replay residual / local gain / fixed readout coupling | gain-model-specific shortcut |
| preservation correction | local multiply + two scalar sums + one broadcast | additional coordination pathway |
| protected replay memory | stored probe + reference answer | external/internal storage cost must be counted |
| teaching `delta` | measured task error at commit | full delayed integration is next |
| slow material `w_j` | persistent branch state | still coordinate-like, not a propagating arbor |

## Is it still a toy?

Yes in substrate complexity; **less so in mechanism**.

Gate 5 is not a general neuron result. The response law is instantaneous and diagonal, protected replays are reproducible, replay sensitivity is measured in a clean zero-command condition, and noise / command-confounded external events / delayed receipt feedback / nonlinear propagation are not all active simultaneously.

But the compatibility operation is no longer missing or handed in as `P`:

> **Replay-driven compatible plasticity is now an implemented mechanism in the present model.**

That moves the repo from "collection of integration controls" toward a **mechanism prototype**.

The point where the result becomes substantially harder to dismiss as a toy is Gate 6: make the same operation survive a continuous changing substrate under bounded replay and receipt memory, then move it into a coupled propagating arbor and eventually Operaattori morphology.

## Research contract

A future gate is not allowed to claim success unless we can answer:

1. What signal does each changing element actually receive?
2. Where does its prediction come from?
3. What counted state carries relevant pending history?
4. What identifies which history delayed consequence refers to?
5. What authorizes the write?
6. How does the write change later physical response?
7. What other useful responses share that material?
8. Does a compatible change exist?
9. How is compatibility information acquired rather than supplied?
10. What replay/measurement/memory cost did that acquisition require?
11. Is delayed learning stable at the chosen gain?
12. Can the system still update when the world changes again?

## Roadmap

```text
Gate 0  efference copy + delayed credit reference
Gate 1  predictor adapts without immediate slow write
Gate 2  addressed reversal reference
Gate 3  material->behavior + bounded delayed-credit receipts
Gate 4  privileged compatible-change + impossibility + delay-stability reference
Gate 5  replay discovers coordinated candidate plasticity without receiving P
Gate 6  continuous changing-substrate task: echo + external overlap + bounded receipts + bounded replay + delayed task error
Gate 7  private dither inside Gate 6 against command-correlated external causes
Gate 8  replace coordinate branches with a coupled propagating arbor and overlapping representations
Gate 9  transplant permitted local signals onto audited Operaattori morphology
Gate 10 nonlinear dendritic operating regimes; active queries expose hidden distinctions
Gate 11 bounded downstream readout chooses another experiment from residual geometry
```

Gate 6 contract: [`GATE6_CONTINUOUS_TASK.md`](GATE6_CONTINUOUS_TASK.md).

The intended destination remains concrete:

> **A morphology-shaped learning substrate that predicts the consequences of its own actions, keeps enough bounded history to use delayed feedback, calculates candidate structural changes against retained behavior before consolidating them, and remains stable and learnable after those changes alter the dynamics it must predict.**
