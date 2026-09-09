# Gate 10 — measured Q-coverage replay

Gate 9 left three distinct problems:

1. **coverage** — did the retained questions represent the old capability?
2. **finite prediction** — did the local model predict the proposed finite write?
3. **compensation** — can a coordinated finite write satisfy the retained contracts?

Gate 9 improved the second/third with measured nonlinear correction, while replay selected by predicted damage showed no held-out access advantage.

Gate 10 tests a different selector motivated by the `QB` term in CausalHorizon.

## Hypothesis

CausalHorizon gives, for a fixed linear event/update/readout system,

```math
\Delta y(b)=H Q b.
```

A bank represents a linear query family when its projected entries `Q b_j` span the relevant event-entry space. IttnasNoruen's classifier does **not** expose that exact `Q`.

Gate 10 therefore uses a black-box proxy. For every candidate old cue `b_j`, measure its response to the same panel of temporary parameter probes `d_k`:

```math
s_j[k] = m_j(\theta+\epsilon d_k)-m_j(\theta),
```

where `m_j` is that cue's stored classification margin.

The signature `s_j` is not asserted to equal `Q b_j`. It is an experimentally accessible sketch of how the current model's response geometry sees the cue.

The new selector asks for a 20-record bank whose measured signature vectors are jointly well-conditioned rather than individually vulnerable.

## Fixed design

- Dataset/model/splits: same handwritten-digit setup as Gate 9.
- New independent seeds: `211, 223, 239`.
- Candidate storage: same bounded 60-record pool, two fragile correct cues per class and view.
- Active replay capacity: 20 records for ordinary methods, exactly two per class.
- Adjustable parameters: the same 24 hidden-bias offsets.
- Progress checkpoints: `+0.05, +0.10, +0.20, +0.40` calibration log-probability.
- Online scalar-callback cap: `45,000`.
- Update-attempt cap: `24`.
- Measured finite-model corrections: `2` as in corrected Gate 9.
- Signature panel: 12 deterministic orthonormal parameter directions.
- Signature amplitude: `epsilon = 0.03`.

The signature directions are generated only from the split seed and are identical across scanned methods.

## Scanned methods and equal scan cost

The four replay selectors are:

- `random_scanned`
- `boundary_scanned`
- `interference_scanned` — Gate-9-style predicted finite damage
- `qspan_scanned` — new coverage/conditioning selector

Every scanned method pays for the same operations before selecting its bank:

1. one temporary ordinary new-task proposal;
2. current margins for all 60 stored cues;
3. the 12-probe response signature for all 60 cues;
4. proposed margins for all 60 cues.

Only the ranking rule differs. Ignored measurements still cost calls.

`all_pool` keeps all 60 constraints and avoids the selector scan; this is a stronger-storage control with a different cost structure, not a matched selector. `unprotected` and `frozen` remain behavior references.

## Q-span selector

Let `s_i in R^12` be candidate cue signatures. The selector greedily maximizes

```math
\log\det\left(\lambda I + \sum_{i\in S}s_i s_i^T\right)
```

under the same two-per-class / 20-total constraint.

This is a D-optimal/volume criterion. It rewards adding response directions that expand or condition the measured span rather than repeatedly selecting the largest individual damage.

The implementation also records:

- selected signature rank;
- ambient pool signature rank;
- minimum singular value;
- condition number when full rank;
- residual fraction of the full pool outside the selected row span;
- regularized log determinant.

These diagnostics are descriptive. The held-out access outcome remains the actual test.

## Outcome comparison

At each checkpoint reached by both methods, pair `qspan_scanned` against:

- `random_scanned`
- `boundary_scanned`
- `interference_scanned`
- `all_pool`
- `unprotected`

Primary preservation outcomes are exactly the Gate-9 families:

- loss rate among initially correct familiar `full/upper/lower` cues;
- loss rate among initially correct unseen masks `upper3/lower3/left4/right4`.

No interpolation is used. Failure to reach a checkpoint is missing coverage, not perfect protection.

## What would count as support

The strongest positive result would be:

1. Q-span banks show better measured signature conditioning/coverage than vulnerability/random controls;
2. that advantage predicts fewer **held-out unseen cue losses** at equal achieved new-task progress;
3. the effect survives more than one seed and does not depend on greater measurement budget.

A geometry improvement without held-out behavioral improvement is not enough.

## What would kill the idea

Useful negative outcomes include:

- Q-span and ordinary selectors have similar signature geometry;
- Q-span improves scan-space conditioning but not unseen access;
- Q-span protects unseen cues only by blocking learning;
- Gate-9 vulnerability selection does as well or better at matched progress;
- the selected geometry is unstable across attempts/seeds.

Any of these means the measured signature is a poor proxy for the relevant `Q`, or that coverage of this local response geometry is not the bottleneck.

## Claim boundary

D-optimal experiment design, determinant/volume subset selection, gradient/Jacobian sketches, replay selection and continual-learning interference are established subjects.

Gate 10 does **not** claim to recover the exact CausalHorizon event-entry map, nor to establish a new general replay algorithm. Its narrow question is:

> Does selecting a bounded replay bank for span/conditioning of measured response signatures protect unseen cue access better than selecting examples by current vulnerability?
