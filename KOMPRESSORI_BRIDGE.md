# From retained answers to retained response geometry

The [Kompressori handoff](https://github.com/anttiluode/Kompressori/tree/7cca26e6677e2f6ee6351f320555102baf7bd8f9)
asks whether compressing a state preserves how it responds to the next perturbation.
IttnasNoruen asks whether changing parameters preserves useful behavior. These
meet at a testable gap: agreement on old inputs does not ensure agreement nearby.

Kompressori reports a high-rank finite-time operator on its 100 sampled input
directions, but a much lower-rank change after a local event. It also reports that
separated changes nearly add while nearby finite changes interact. Those results
concern its particular field and sampled subspace. They do not establish low rank
for arbitrary learning updates. This handoff has not independently reproduced them.

## Executable counterexample

[response_geometry_reference.py](response_geometry_reference.py) isolates the gap:

```math
f_k(q)=q+kq(q-1).
```

Answers at `q=0` and `q=1` are identical for **every** `k`. Their slopes are `1-k`
and `1+k`. Two old answers alone cannot expose this changing sensitivity.

The learner starts at `k=0` and is asked to change the answer at `q=0.5` from 0.5
to -1.5. It receives the scalar callback and measured perturbations, without a
supplied derivative or update direction.

| Contract | New answer after one accepted update | Old-answer drift | Largest absolute anchor slope | Calls including baseline |
| --- | ---: | ---: | ---: | ---: |
| Two anchors | -1.5 | 0 | 9 | 14 |
| Anchors plus four nearby probes | 0.4375 | 0 | 1.25 | 69 |

Both methods have a 200-call cap. The second retains six questions instead of two
and requires nearby responses to remain within 0.01. The extra contract restricts
new-task progress; it does not provide equal learning for free.
[Frozen receipt](results/response_geometry_reference.json).

This is a mathematical reference, not a new empirical success. It makes the
attacker explicit: **an update can preserve every recorded answer while changing
how the system responds to the next nearby cue.**

## Two derivatives, different jobs

The guard measures parameter sensitivities `a_i = gradient_theta f_theta(q_i)`.
Response geometry concerns input sensitivities:

```math
J_\theta(q)=D_q f_\theta(q),\qquad
\Delta J=J_{\theta+\Delta\theta}-J_\theta.
```

Estimating the first does not automatically protect the second. Nearby probes
constrain a few consequences of `Delta J`; they do not certify a whole Jacobian,
all input directions or dynamical stability. Kompressori's failed held-out
response-probe result warns against claiming that a few directions suffice.

The next experiment can allocate a fixed probe budget to directions likely to
reveal damage from the proposed update, then score unseen directions independently.
Any compact update sketch must pay for acquisition, storage and approximation
error. Compare it with ordinary replay and random nearby probes at matched
new-task progress. Gate 8's partial-cue losses are a real-data motivation for this.

## Euler/Navier–Stokes inspiration

OpenAI's [official repository](https://github.com/openai/NavierStokesAndEuler)
publishes Lean formalizations and states forced Navier–Stokes and unforced Euler
blowup results. We have not independently audited the proofs. IttnasNoruen uses
neither those equations nor their singularity construction. The useful question
is whether a change prepares the system to amplify a later perturbation. The
scalar reference establishes its own limited claim directly.
