# IttnasNoruen

**Expectation + eligibility + consequence + slow material.**

This repo is the integration step suggested by the recent Sigh / Jello / Child / Active Dendrite work. It is not a claim that we have built a biological neuron. It is a deliberately small executable object that forces several previously separate mechanisms to share one accounting of signals and memory.

The immediate question is:

> **Can one local learning system predict the consequences of its own action, preserve unexplained local activity long enough for delayed consequence, and change slow material without teaching that material the predictable self-echo?**

That is narrower than "build a neuron" and stronger than another analogy.

## The object

For branch-like compartment `j`, the first model exposes only these quantities:

```text
outgoing command u
      |
      +--------------------> local echo predictor k_j
      |                              |
      |                              v
      |                         predicted self echo
      |                              |
      v                              |
physical/local response x_j --------(-)----> residual r_j
                                             |
                                             v
                                      eligibility e_j
                                             |
                                  delayed consequence delta
                                             |
                                             v
                                      slow material w_j
```

The equations are intentionally simple:

```math
\hat x_j = k_j u
```

```math
r_j = x_j - \hat x_j
```

```math
e_j \leftarrow \gamma e_j + r_j
```

The self-return model is a separate fast learner:

```math
k_j \leftarrow k_j + \eta_k u r_j
```

The slow material does **not** learn merely because a prediction error exists. It changes only when a delayed consequence authorizes the surviving eligibility:

```math
w_j \leftarrow w_j + \eta_w \delta e_j
```

This separation is the point.

> **Prediction supplies an expectation. Eligibility associates a later consequence with earlier local activity. Consequence supplies a reason to change. Slow material stores the change.**

A single scalar called `surprise` is not asked to do all four jobs.

## Gate 0 — Efference copy cleans delayed credit

A known command creates a reproducible spatial echo across four branch-like compartments. An external event is added on one branch. The consequence arrives only after several quiet ticks.

Two otherwise identical learners receive the same activity:

1. **EFF_COPY** learns the command-linked return during a self-only calibration phase and subtracts its predicted echo before forming eligibility.
2. **NO_COPY** has no self-return predictor, so command-generated activity and external activity enter the same eligibility trace.

The gate measures how much of the resulting slow material is assigned to the actually perturbed branch.

This is not a free biological result. Gate 0 explicitly grants a command copy and one local predictor per branch. It also uses a declared self-only calibration phase. Those are counted resources, not hidden oracles.

Run:

```bash
python gates.py
```

## Gate 1 — Prediction error is not automatically memory

After the echo predictor has calibrated, the physical self-echo changes. The predictor becomes wrong and must adapt.

There is **no external consequence**.

Success means:

- prediction error initially rises;
- the fast echo model learns the new return;
- the slow material does not change merely because the predictor was surprised.

This is a direct attack on the tempting but unsafe rule:

```text
prediction error -> rewrite everything
```

## Gate 2 — Update one distinction without erasing another

The system first stores a consequence-linked event on one branch and a separate event on another. Then the consequence associated with the second branch reverses sign.

Success means:

- the reversible branch changes sign;
- the already useful branch remains effectively unchanged.

This is only an addressed toy, so it does not solve interference in a shared dendritic substrate. It gives that harder future experiment a reference behavior.

## What is local and what is still privileged?

The current object receives no label for the external event and never inspects a hidden matrix. But it still assumes more access than one soma normally has:

| quantity | where it comes from here | future biological question |
|---|---|---|
| command copy `u` | explicit efference-copy input | which pathway carries it? |
| local activity `x_j` | branch-local observable | what variable is actually available locally? |
| predicted self echo | local learned coefficient `k_j` | what circuit/synapse implements this predictor? |
| residual `r_j` | local subtraction | can a local comparator be implemented physically? |
| eligibility `e_j` | decaying local trace | what molecular/electrical state carries it? |
| consequence `delta` | delayed scalar teaching signal | how is it delivered and gated? |
| material `w_j` | persistent addressed state | what synaptic/branch parameter should it correspond to? |

The repo should become *more* constrained as it grows, not less.

## Why this repo exists

The older projects found pieces of this object separately:

- [SighImageSuper](https://github.com/anttiluode/SighImageSuper): learned self-echoes, structural memory, interrogation damage, confounded prediction, private intervention.
- [JelloBrain](https://github.com/anttiluode/JelloBrain): propagated activity writes future routes; selective write permission can prevent self-reinforcement but does not solve every remapping problem.
- [Child](https://github.com/anttiluode/Child): local history can make a future local state predictable; delayed interpretation requires counted traces.
- [FunctionalArbors](https://github.com/anttiluode/FunctionalArbors): consequence can be transported back through an arbor, while causal eligibility remains the harder identification problem.
- [Operaattori](https://github.com/anttiluode/Operaattori): morphology and biophysics give us a real electrical substrate to transplant into later.
- [OperaattoriAktiivinenDendriitti](https://github.com/anttiluode/OperaattoriAktiivinenDendriitti): asks which intervention makes a hidden structural distinction observable under nuisance and limited measurements.
- [HigherLevelOfAbstraction](https://github.com/anttiluode/HigherLevelOfAbstraction): expectation tests what the current abstraction preserves; structured error motivates another experiment before rewriting the representation.

`IttnasNoruen` is where these pieces are forced into one loop.

## Research contract

A future gate is not allowed to claim success unless we can answer:

1. **What signal does each changing element actually receive?**
2. **Where does its prediction come from?**
3. **What physical or counted state carries the relevant history?**
4. **What authorizes a write?**
5. **What other useful responses does that write disturb?**
6. **Can the system still update when the external relationship changes?**

## Roadmap

```text
Gate 0  efference copy + eligibility + delayed consequence in one loop
Gate 1  stale predictor adapts without surprise becoming structural memory
Gate 2  consequence reversal while a separate stored distinction survives
Gate 3  remove the clean self-only calibration phase
Gate 4  private dither / intervention to separate self cause from correlated world cause
Gate 5  replace vector branches with an explicit propagating arbor
Gate 6  transplant the local signals onto audited Operaattori morphology
Gate 7  add nonlinear dendritic operating regimes and ask which distinctions become observable
Gate 8  bounded downstream readout chooses another intervention when residual geometry is ambiguous
```

The intended destination is not "a neuron that computes Bayesian inference." It is more concrete:

> **A morphology-shaped learning substrate that predicts the local consequences of its own actions, preserves causal traces long enough for delayed consequence, changes only where evidence authorizes change, and remains capable of learning when the world stops matching its expectation.**
