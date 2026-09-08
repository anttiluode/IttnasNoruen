# IttnasNoruen

**Expectation + eligibility + consequence + slow material, now closed back onto behavior.**

This repo is the integration step suggested by the recent Sigh / Jello / Child / FunctionalArbors / Active Dendrite work. It is not a claim that we have built a biological neuron. It is a deliberately small executable object that forces previously separate mechanisms to share one accounting of signals, memory, prediction, credit and future response.

The immediate question is now:

> **Under what information and memory constraints can a system keep learning about—and through—a substrate that its own learning continually changes?**

The current four-branch program makes that question concrete. Continued learning needs at least four things to line up:

1. available actions/measurements must distinguish causes that require different updates;
2. enough history must survive until delayed feedback arrives;
3. an achievable material change must improve the new response while preserving responses that still matter;
4. delayed feedback must drive that change at a stable rate.

The first three gates integrate prediction, receipts and material feedback. **Gate 4 is an exact mathematical reference for item 3 and the delayed-feedback stability part of item 4. It is explicitly not yet a locally discovered mechanism.**

## The object

For branch-like compartment `j`, the model exposes:

```text
outgoing command u
      |
      +--------------------> local echo predictor k_j
      |                              |
      |                              v
      |                         predicted self echo
      |                              |
      v                              |
material-shaped local response x_j -(-)----> residual r_j
                                             |
                                             +----> aggregate eligibility e_j  (attacker/reference)
                                             |
                                             +----> receipt-local trace        (counted causal window)
                                                              |
                                                   delayed teaching delta
                                                              |
                                                              v
                                                       slow material w_j
                                                              |
                                                              +----> changes future local gain
```

The predictor and residual are:

```math
\hat x_j = k_j u
```

```math
r_j = x_j - \hat x_j
```

The original aggregate eligibility remains:

```math
e_j \leftarrow \gamma e_j + r_j
```

The self-return model is a separate fast learner:

```math
k_j \leftarrow k_j + \eta_k u r_j
```

The slow material changes only when a delayed **signed teaching signal** is applied. In this repo, the teaching value is not treated as an undifferentiated raw reward. Calling the consequence method is the write-permission event.

```math
w_j \leftarrow w_j + \eta_w \delta e_j
```

Gate 3 adds a second credit representation: a **receipt**. A receipt is a counted temporal/action record containing the residual vector accumulated during a declared causal window. The delayed teaching signal later names the receipt, **not a branch**.

The crucial new loop is that material is no longer merely inspected. It changes later physical response through a deliberately simple toy gain:

```math
g_j = 1 + \beta w_j
```

```math
x_j = g_j\,(s_j u + v_j)
```

where `s_j u` is the command-linked physical echo and `v_j` is external drive. This is not proposed as a biophysical conductance law. Its job is to enforce the coupling that Sigh made unavoidable:

> **Writing material changes the response that the self-predictor must subsequently explain.**

A fixed soma readout measures branch activity. Slow material therefore has to earn its meaning by changing a future measured response rather than by being inspected directly.

## Gate 0 — Efference copy cleans delayed credit

Historical reference control. A known command creates a reproducible spatial echo across four branch-like compartments. An external event is added on one branch. The consequence arrives only after several quiet ticks.

Two otherwise identical learners receive the same activity:

1. **EFF_COPY** learns the command-linked return during a self-only calibration phase and subtracts its predicted echo before forming eligibility.
2. **NO_COPY** has no self-return predictor, so command-generated activity and external activity enter the same eligibility trace.

Gate 0 deliberately keeps material-to-activity feedback off so its old interpretation remains stable. It is a control, not the integrated destination.

## Gate 1 — Prediction error is not automatically memory

Historical control. After calibration, the physical self-echo changes. The predictor becomes wrong and adapts.

There is no delayed teaching signal, so slow material does not change.

This proves only the timing separation:

> **prediction error can train the fast predictor without immediately becoming slow material.**

Astra's review exposed the stronger problem: that same residual can remain eligible and be written later by an unrelated valid consequence. Gate 3 attacks that explicitly.

## Gate 2 — Addressed reversal reference

Historical addressed control. One stored coordinate is reversed while another stays unchanged.

This remains a deliberately easy reference because coordinates are separate and events are isolated. Gate 3 upgrades preservation to a future-response measurement through the same fixed soma readout.

## Gate 3 — Material earns meaning; credit receipts attack overlap

Astra's review identified two missing couplings:

1. learned material was written and inspected but did not affect the activity used by later gates;
2. one aggregate eligibility vector could not distinguish several surviving candidate causes.

Gate 3 therefore contains four linked tests rather than inventing four new theories.

### 3A. The write changes future behavior

A consequence-linked event writes one branch. We then ask a held-out question through the fixed soma readout.

Success requires:

- the future soma response changes because material changed the branch gain;
- that write makes the previously calibrated self-predictor stale;
- the fast predictor relearns the new self-response;
- relearning the predictor alone does not create another slow-material write.

This closes the missing loop:

```text
prediction -> residual -> eligible write -> material -> future activity -> new prediction problem
```

### 3B. A valid consequence can authorize the wrong surviving residual

We reproduce Astra's attacker:

1. calibrate the self predictor;
2. change the self-response, producing a stale-model residual;
3. before that residual has decayed, present a real external event on branch 2;
4. later deliver the valid teaching signal for the external event.

The legacy aggregate trace writes roughly the same off-target pattern Astra reported. The problem is not lack of write permission. The problem is that permission applies to **everything still in the one trace**.

The receipt version opens a new counted causal window only for the later external event. Its delayed teaching signal applies to that retained window rather than to the older residual.

This is not a free causal oracle. The system is being given a temporal/action boundary.

### 3C. Multiple outstanding consequences

Two events occur one tick apart, and neither consequence arrives yet.

The legacy aggregate trace gives the first consequence to both events. Because `apply_consequence()` clears the entire aggregate trace, the second consequence then finds nothing left.

With receipts:

```text
receipt A: event A trace -------- waits -------- consequence A
receipt B: event B trace -------- waits -------- consequence B
```

Both traces decay with time, but one consequence does not erase the other receipt.

Again, the declared resource is important: delayed feedback identifies the earlier **receipt/action window**, not the hidden branch that caused success. Arbitrarily overlapping unlabeled causes remain ambiguous.

### 3D. Preservation is measured as behavior

Two addressed relations first affect the future response bank. One relation is then revised.

The test no longer asks merely whether an untouched coefficient stayed numerically constant. It measures both held-out questions through the fixed soma readout and requires:

- the protected response to remain stable;
- the revised response to actually change.

This still does not solve the full Jello shared-representation interference problem. It upgrades the reference metric to the right kind of object: **future behavior**.

## Gate 4 — Compatible change reference

Astra's next attack exposed a deeper issue: even perfect credit attribution does not tell us **which coordinated material change can improve one response without damaging the responses that share that material**.

Gate 4 turns the old question

> **What may I change without damaging something else?**

into an exact linear reference.

Four gains start at

```math
g=[1,1,1,1].
```

Three overlapping responses must stay at `2`:

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

Put the protected sensitivities in a matrix `P`. A preserving update must satisfy

```math
P\Delta g=0.
```

The new response must also change:

```math
p^T\Delta g\ne0.
```

The exact compatible direction is

```math
d=[0.5,-0.5,0.5,-0.5].
```

It strengthens branches 1 and 3 while compensating on 2 and 4. Every branch participates in an old response; preservation is therefore no longer trivial coordinate independence.

If teaching is remaining behavioral error,

```math
\delta=3-y_{new},
```

and

```math
g\leftarrow g+\eta\delta d,
```

with `eta=0.5`, then

```text
2 -> 2.5 -> 2.75 -> 2.875 -> ... -> 3
```

while the protected response bank stays fixed to numerical precision.

The gate also includes two attackers:

- **naive active-only strengthening** reaches toward the new target but damages protected responses;
- **full-rank protection** has no non-zero safe direction at all, showing that some requested changes are genuinely incompatible without more capacity or a representation change.

Finally, Gate 4 checks the exact delayed-error recurrence

```math
\delta_{t+1}=\delta_t-a\delta_{t-D}
```

for which this restricted linear reference is stable only when

```math
0<a<2\sin\left(\frac{\pi}{4D+2}\right).
```

The resulting boundaries are `2` for zero delay, `1` for one-step delay and about `0.445042` for three-step delay. The code verifies the characteristic roots immediately below and above those boundaries.

**Important:** Gate 4 is a mathematical target, not a mechanism already discovered by the local learner. The code is given `P`. The substantive missing problem is how a bounded changing substrate obtains or embodies enough information about protected response sensitivities to approximate this coordinated direction.

Full derivation and research contract: [`COMPATIBLE_CHANGE_REFERENCE.md`](COMPATIBLE_CHANGE_REFERENCE.md).

Run the integration controls and reference:

```bash
python gates.py
python gate4_compatible_change.py
```

## The new distinctions

The first integration object separated:

> **surprise** from **permission to write**.

Astra's overlap attack exposed:

> **eligibility that merely survives** is not the same as **eligibility that belongs to the consequence that arrived**.

Gate 4 adds a third separation:

> **knowing which experience received feedback** is not the same as **knowing which coordinated material change can improve it without damaging other useful responses**.

A receipt can help with the former. Preservation requires information about how other useful responses share the material being changed.

## Bounded receipt memory

The receipt store is no longer silently unbounded. `IttnasNoruen` accepts:

```python
max_receipts=K
```

and refuses to open another receipt when that budget is exhausted.

With `N` branches and `K` outstanding receipts, raw receipt traces contain `N*K` scalar values, plus identifiers and bookkeeping. For this four-branch toy that is `4K` trace values.

This does not solve attribution. It makes the cost explicit: if feedback can be delayed beyond the available receipt budget, the system must limit outstanding actions, compress/overwrite records, or accept that some consequences can no longer be assigned.

## What is local and what is still privileged?

| quantity | where it comes from here | remaining restriction |
|---|---|---|
| command copy `u` | explicit efference-copy input | which pathway carries it? |
| local activity `x_j` | material-shaped branch-local observable | toy gain, not yet cable/NMDA physics |
| predicted self echo | local learned coefficient `k_j` | clean calibration remains in Gates 0-3 references |
| residual `r_j` | local subtraction | physical comparator still supplied |
| aggregate eligibility | one decaying local vector | confounds overlapping causes |
| receipt trace | declared action/time window | boundary + receipt identity are counted resources |
| receipt capacity | explicit `max_receipts` | compression/eviction policy not yet learned |
| teaching `delta` | delayed signed teaching signal | Gates 0-3 schedule it externally; Gate 4 uses declared task error |
| material `w_j` | persistent addressed state | still one coordinate per branch |
| soma answer | fixed weighted branch readout | not yet an actual morphology-based soma |
| protected sensitivity matrix `P` | supplied only to Gate 4 reference | local system does not yet know how to obtain/embody it |

The repo should become **more constrained** as it grows, not less.

## Why this repo exists

The older projects found pieces separately:

- [SighImageSuper](https://github.com/anttiluode/SighImageSuper): learned self-echoes, structural memory, interrogation damage, confounded prediction, private intervention, and the crucial material->response->prediction coupling.
- [JelloBrain](https://github.com/anttiluode/JelloBrain): propagated activity writes future routes; selective write permission reduces self-reinforcement but does not solve shared remapping.
- [Child](https://github.com/anttiluode/Child): local history predicts future local state; delayed interpretation requires counted pending records.
- [FunctionalArbors](https://github.com/anttiluode/FunctionalArbors): consequence can travel back through an arbor, while identifying the eligible cause remains harder.
- [Operaattori](https://github.com/anttiluode/Operaattori): morphology and biophysics supply the electrical substrate to transplant into later.
- [OperaattoriAktiivinenDendriitti](https://github.com/anttiluode/OperaattoriAktiivinenDendriitti): asks which intervention makes hidden structural distinctions observable under nuisance and limited measurements.
- [HigherLevelOfAbstraction](https://github.com/anttiluode/HigherLevelOfAbstraction): error structure should choose another experiment before deciding what representation to rewrite.

`IttnasNoruen` is where these pieces are forced to interfere with one another.

## Research contract

A future gate is not allowed to claim success unless we can answer:

1. **What signal does each changing element actually receive?**
2. **Where does its prediction come from?**
3. **What physical or counted state carries the relevant history?**
4. **What identifies which history a delayed consequence refers to?**
5. **What authorizes the write, and is that signal reward, teaching error, or permission?**
6. **How does the write change later physical response?**
7. **What other useful responses does that change disturb?**
8. **Does a compatible change exist under the current representation?**
9. **What information lets the system approximate that change without being handed `P`?**
10. **Is the delayed learning loop stable at the chosen effective gain?**
11. **Can the system still update when the external relationship changes?**

## Roadmap

```text
Gate 0  efference copy + aggregate eligibility + delayed consequence reference
Gate 1  stale predictor adapts without immediate slow write
Gate 2  addressed reversal reference
Gate 3  material->activity coupling + stale-credit attacker + multiple receipts + behavioral preservation
Gate 4  exact compatible-change + impossibility + delayed-feedback stability reference
Gate 5  one continuous four-branch task: derive task error, bounded receipts, material feedback, continuous echo learning
Gate 6  private dither inside that task against command-correlated external causes
Gate 7  learn/embody protected-response constraints instead of receiving P
Gate 8  replace coordinate branches with an explicit propagating arbor and overlapping representations
Gate 9  transplant local signals onto audited Operaattori morphology
Gate 10 nonlinear dendritic operating regimes; choose interventions that expose hidden distinctions
Gate 11 bounded downstream readout chooses another intervention from residual geometry
```

The next substantial experiment is **Gate 5**, not a larger arbor. It must put the difficult ingredients in one stream and compare methods under the same information budget:

- material feedback continuously enabled;
- self-commands and external events can overlap;
- echo prediction continues while material changes;
- teaching comes from measured task error, not scheduled positive/negative signs;
- useful old responses overlap the material needed by the new response;
- receipt capacity is fixed;
- private dither is allowed only as a counted intervention and retains its known failure when the world follows the dither too;
- success means new-task improvement, preserved old responses, controlled material magnitude and stable delayed learning.

The intended destination is not "a neuron that computes Bayesian inference." It is more concrete:

> **A morphology-shaped learning substrate that predicts local consequences of its own actions, keeps distinct causal traces when its available timing/actions permit that distinction, discovers coordinated material changes that preserve still-useful behavior, and remains stable and learnable after those changes alter the dynamics it must predict.**
