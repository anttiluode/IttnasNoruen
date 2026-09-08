# Learning under behavioral constraints

The reusable object in this repository is an update process that asks two different
questions: what new behavior should improve, and which established responses must
remain within their declared tolerances?

For a system with adjustable material or software parameters `theta`, write

```math
f_\theta(q)=\text{response to question }q.
```

A bounded reference bank stores `(q_i, b_i, epsilon_i)`. It defines a measured
behavioral contract:

```math
\mathcal C=\{\theta: |f_\theta(q_i)-b_i|\leq\epsilon_i\text{ for every retained }i\}.
```

The learner seeks an update that improves the new task while staying in this set.
This is constrained adaptation. The four-branch construction is one instance;
neither images, diagonal gains nor neuron terminology are essential to the definition.

## Three different discrepancies

These must not be collapsed into one variable called surprise.

| Signal | Comparison | Job |
| --- | --- | --- |
| Self-prediction residual | actual activity minus the predicted effect of the copied command | estimate what current action explains |
| Task error | desired task output minus measured task output | authorize improvement of the relevant behavior |
| Preservation discrepancy | current/candidate response minus a retained reference response | constrain damage to an existing commitment |

An efference copy is a copy of a command. A predictor uses that copy to form an
expected response. A replay bank supplies a historical or normative reference.
They can share machinery, but one is not automatically the other.

Replaying a question detects a response change through the available interface.
It attributes that change to a material update only when the relevant other causes
are controlled or modeled: input realization, state, noise, readout and apparatus
drift. With a fixed linear input/output interface, a before/after measurement sees
`C (A_new - A_old) B q`; it does not reveal the whole operator difference.

## What fast proposal dynamics accomplish

At a fixed operating point, measured sensitivities form a local approximation:

```math
f_{\theta+\Delta}(q_i)\approx f_\theta(q_i)+a_i^\top\Delta.
```

Gate 5 acquires `a_i` from an unusually simple local gain law. Gate 6 uses noisy
replay to refresh a bounded sketch while acting and learning. The software guard
instead estimates sensitivities by counted parameter perturbations of a scalar
response callback.

Each replay can remove the harmful part of a proposed change. Its orthogonal
projector leaves the locally preserving directions invariant. This supplies a
precise link to Sigh: fast dynamics can select a candidate change before that
change becomes slow structure. It is a numerical construction, not evidence that
dendrites implement the same circuit.

Nearly parallel constraints make a fixed number of cyclic projections slow. The
general software guard therefore orthogonalizes its bounded measured sketch. This
adds central arithmetic and basis storage. It is not described as purely local
plasticity.

## A tangent is not a certificate for a finite nonlinear update

The guard includes a restoration component for existing reference drift, a proposed
new-task component, a trust radius, and direct candidate evaluation. A step may be
shortened or rejected when actual responses violate the contract. Budget exhaustion
returns the original parameters rather than a partially checked update.

The software callback can evaluate temporary parameter vectors without committing
live material. This is a useful capability for a trained network or simulator. A
physical implementation would have to supply reversible perturbations, a justified
predictor or a separate test system, and account for their costs and back-action.
The current code does not provide that biological mechanism.

## The higher-level limitation: reference coverage

A finite bank protects responses at retained questions. It does not protect an
arbitrary family of unseen questions. Two systems can agree on every stored probe
and disagree elsewhere. No update rule removes that ambiguity without additional
assumptions or measurements.

For example, if the response difference as a function of query is known to be
`L`-Lipschitz and the retained questions cover the relevant query set within radius
`rho`, a pointwise bank bound `epsilon` implies a bound `epsilon + L*rho` over that
set. Here `L` bounds the difference function, not just one model. Neither this
regularity bound nor sufficient coverage is established by the present experiments.

The handwritten-digit experiment measures this gap explicitly. Anchor preservation,
average held-out accuracy and worst held-out response drift are separate metrics.
Good averages can conceal large changes to individual answers.

Consequently, a next useful mechanism would allocate the replay budget to questions
that reveal likely interference. It should be tested against fixed and random banks
under the same total observations. This is the link back to Active Dendrite: choose
an intervention that reveals a currently hidden distinction, now a distinction
between acceptable and damaging updates.

## Status and prior work

This repository now contains an executable continuous integration benchmark, a
portable software guard, a nonlinear propagation test, and a real-data classifier
example. Those are different capability levels, and their privileges are listed
separately in the [results report](RESULTS.md).

The projection principle is established work. We do not claim a new general solution
to continual learning or a new biological learning law:

- [Halperin, The Product of Projection Operators (1962)](https://acta.bibl.u-szeged.hu/13958/)
  establishes the cyclic projection limit underlying the ideal construction.
- [Strohmer and Vershynin, A randomized Kaczmarz algorithm with exponential convergence](https://arxiv.org/abs/math/0702226)
  studies row-wise correction without requiring the whole system at once.
- [Lopez-Paz and Ranzato, Gradient Episodic Memory (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)
  constrains new learning using retained experience.
- [Farajtabar et al., Orthogonal Gradient Descent (2020)](https://proceedings.mlr.press/v108/farajtabar20a.html)
  explicitly projects gradients to preserve earlier outputs locally.

The program's testable emphasis is the interaction among changing material, learned
self-prediction, bounded delayed credit, measured compatibility, reference coverage
and the cost of obtaining the evidence that approves a change.

The language of relations is useful here: a stored commitment is a relation between
an intervention and a response. It does not imply that physical storage disappears,
or establish an ontological or quantum-mechanical claim about what objects are.
