# IttnasNoruen

**Expectation + eligibility + consequence + slow material, now closed back onto behavior.**

This repo is the integration step suggested by the recent Sigh / Jello / Child / FunctionalArbors / Active Dendrite work. It is not a claim that we have built a biological neuron. It is a deliberately small executable object that forces previously separate mechanisms to share one accounting of signals, memory, prediction, credit and future response.

The immediate question is now:

> **Can one local learning system predict the consequences of its own action, preserve candidate causes until delayed consequence, change slow material only when authorized, and then survive the fact that its own write changes the future activity it must predict?**

That is narrower than "build a neuron" and stronger than another analogy.

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

Run everything:

```bash
python gates.py
```

## The new distinction: survival is not belonging

The first integration object separated:

> **surprise** from **permission to write**.

Astra's overlap attack exposes the next separation:

> **eligibility that merely survives** is not the same as **eligibility that belongs to the consequence that arrived**.

A single decaying vector answers "what recent unexplained activity is still around?"

It does not automatically answer:

> "which earlier activity is this delayed teaching signal about?"

That is the old FunctionalArbors credit problem in a tiny executable form.

The first explicit answer tested here is a counted temporal/action receipt. It is useful when the system has action or episode boundaries. It is insufficient when several unlabeled causes genuinely overlap inside the same window.

## What is local and what is still privileged?

| quantity | where it comes from here | remaining restriction |
|---|---|---|
| command copy `u` | explicit efference-copy input | which pathway carries it? |
| local activity `x_j` | material-shaped branch-local observable | toy gain, not yet cable/NMDA physics |
| predicted self echo | local learned coefficient `k_j` | clean calibration remains in Gates 0-3 references |
| residual `r_j` | local subtraction | physical comparator still supplied |
| aggregate eligibility | one decaying local vector | confounds overlapping causes |
| receipt trace | declared action/time window | boundary + receipt identity are counted resources |
| teaching `delta` | delayed signed teaching signal | source/delivery not yet biological |
| material `w_j` | persistent addressed state | still one coordinate per branch |
| soma answer | fixed weighted branch readout | not yet an actual morphology-based soma |

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
8. **Can the system still update when the external relationship changes?**

## Roadmap

```text
Gate 0  efference copy + aggregate eligibility + delayed consequence reference
Gate 1  stale predictor adapts without immediate slow write
Gate 2  addressed reversal reference
Gate 3  material->activity coupling + stale-credit attacker + multiple receipts + behavioral preservation
Gate 4  remove clean self-only calibration; continuous echo learning with external events
Gate 5  private dither / intervention against command-correlated external causes
Gate 6  replace coordinate branches with an explicit propagating arbor and overlapping representations
Gate 7  transplant local signals onto audited Operaattori morphology
Gate 8  nonlinear dendritic operating regimes; choose interventions that expose hidden distinctions
Gate 9  bounded downstream readout chooses another intervention from residual geometry
```

The intended destination is not "a neuron that computes Bayesian inference." It is more concrete:

> **A morphology-shaped learning substrate that predicts local consequences of its own actions, keeps distinct causal traces when its available timing/actions permit that distinction, lets delayed teaching modify the right surviving candidate rather than every recent surprise, and remains useful after its own writes change the dynamics it must predict.**
