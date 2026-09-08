# Compatible Change Reference

This note nails down the mathematical target exposed by Astra's review.

It is **not** a claim that `IttnasNoruen` already discovers the required update from local signals. It is the reference answer the continuous learner must eventually reproduce or approximate under counted information and memory constraints.

The question is:

> **Under what information and memory constraints can a system keep learning about—and through—a substrate that its own learning continually changes?**

For the present four-branch linear reference, continued learning is possible when four conditions all hold:

1. **Distinguishability:** available observations/interventions separate causes that require different updates.
2. **History:** enough information survives until delayed feedback arrives to identify the relevant pending experience.
3. **Compatibility:** there exists an achievable material-change direction that improves the new response while preserving responses that still matter.
4. **Stability:** delayed feedback applies that change slowly enough that the closed learning loop does not oscillate or diverge.

The new code in `gate4_compatible_change.py` makes conditions 3 and 4 exact and testable. The existing efference-copy / receipt machinery addresses parts of conditions 1 and 2, but does not yet solve them in one continuous task.

## 1. What may I change without damaging something else?

Let four material gains begin at

```math
g=[1,1,1,1].
```

Three overlapping responses are already useful and must stay fixed:

```math
y_1=g_1+g_2,
```

```math
y_2=g_2+g_3,
```

```math
y_3=g_3+g_4.
```

Initially all three equal `2`.

A new task requires

```math
y_{new}=g_1+g_3
```

to move from `2` to `3`.

The protected-response sensitivities are

```math
P=
\begin{bmatrix}
1&1&0&0\\
0&1&1&0\\
0&0&1&1
\end{bmatrix}.
```

A preserving material change must satisfy

```math
P\,\Delta g=0.
```

The new response has sensitivity

```math
p=[1,0,1,0].
```

Useful learning additionally requires

```math
p^T\Delta g\ne0.
```

For this construction, projecting the proposed new-task direction into the null space of `P` gives

```math
d=[0.5,-0.5,0.5,-0.5].
```

This direction obeys

```math
Pd=0
```

while

```math
p^Td=1.
```

Therefore the coordinated change

```math
\Delta g=[+0.5,-0.5,+0.5,-0.5]
```

produces

```math
g=[1.5,0.5,1.5,0.5]
```

and exactly preserves the old response bank:

| response | before | after |
|---|---:|---:|
| `g1 + g2` | 2 | 2 |
| `g2 + g3` | 2 | 2 |
| `g3 + g4` | 2 | 2 |
| `g1 + g3` | 2 | 3 |

This is the first exact version of the old question:

> **Change the material along a direction that leaves the protected responses invariant.**

The important point is overlap: every branch participates in at least one protected response. Preservation is no longer trivial independence between coordinates.

## 2. The impossibility criterion

A compatible direction is not guaranteed to exist.

If the protected sensitivities span the whole adjustable material space, then

```math
\mathrm{null}(P)=\{0\}.
```

No non-zero update can preserve all protected responses.

`gate4_compatible_change.py` includes an exact full-rank control using four independent protected responses in four dimensions. The projected safe direction has norm zero.

More training cannot fix this incompatibility. The system must do at least one of the following:

- relax a preservation requirement;
- add adjustable capacity;
- change representation;
- change which measurements/actions define the task.

For nonlinear systems, the same sensitivity argument becomes local: a direction safe for a small step need not remain safe after a large move.

## 3. Error-driven teaching removes the runaway positive-instruction loop

The current material rule exposed an amplification loop when every episode supplies another fixed positive teaching sign.

The reference instead uses remaining task error:

```math
\delta=3-y_{new}.
```

Then update only along the preserving direction:

```math
g\leftarrow g+\eta\,\delta\,d.
```

With

```math
\eta=0.5,
```

the new response follows

```text
2 -> 2.5 -> 2.75 -> 2.875 -> ... -> 3
```

while all three protected responses remain exactly `2` in the deterministic reference.

After 20 updates the gate reaches the target to within about `1e-6` while the measured protected-response drift remains at numerical precision.

This gives the teaching signal a stopping meaning:

> **As behavior becomes correct, the signal driving further structural change vanishes.**

That is qualitatively different from issuing an unlimited stream of positive reinforcement.

## 4. A naive local update shows why compensation matters

If the learner simply strengthens the branches active in the new probe, it can increase `g1 + g3`, but it also changes old responses that share branches 1 and 3.

The Gate 4 reference records this attacker explicitly.

So the missing information is not merely "which new event got feedback?"

A receipt can answer:

> **Which earlier action/window is this delayed feedback about?**

Protection requires another kind of knowledge:

> **Which other useful responses depend on the same material I am about to change?**

Those are different problems.

## 5. Where can the compensating direction come from?

The reference code is deliberately privileged: it is given the protected sensitivity matrix `P`.

`IttnasNoruen` does **not** yet know how to obtain or embody `P` from its present permitted signals.

Candidate mechanisms to test rather than assume include:

- replay of protected questions and measured response changes;
- stored local response sensitivities;
- additional active probes;
- geometry/circuit coupling that automatically produces compensating changes;
- a bounded approximation learned from previous safe and unsafe updates.

This is the substantive architectural gap.

The interesting future result is not the null-space projection itself; projection against protected directions is established continual-learning mathematics. The interesting result would be showing how a constrained changing substrate can acquire enough information to approximate a compatible direction using the same limited signals and actions it is trying to learn through.

## 6. Distinguishability is a separate requirement

A residual may be caused by:

- an external event;
- changed self-response because material changed;
- an inaccurate echo predictor;
- calibration/noise.

If every available intervention and measurement makes two candidate causes respond identically, no update rule can reliably distinguish them.

The Sigh private-dither control gives one sufficient construction: perturb the self-command in a direction that changes the self-response while the confounding external process does not follow that perturbation.

If the external process follows the private dither too, the separation disappears. That failure remains a declared identifiability limit.

So there is no universal rule:

```text
residual -> external evidence
```

The system needs informative interventions, independently grounded feedback, or a justified model restriction.

## 7. Pending-history memory is now explicitly bounded

The receipt implementation originally used an unbounded dictionary.

`IttnasNoruen` now accepts

```python
max_receipts=K
```

and refuses to open a new receipt when that capacity is exhausted.

With `N` branches and `K` outstanding receipts, the raw receipt traces contain

```math
N K
```

stored scalar values, plus identifiers and bookkeeping.

For the four-branch reference this is `4K` trace values.

A smaller representation may suffice when the task admits compression. What cannot be discarded is information that distinguishes histories which later demand different updates.

Capacity therefore creates a real operational tradeoff:

- limit outstanding actions;
- compress pending records;
- overwrite/forget some records;
- or accept that some delayed feedback becomes unassignable.

## 8. Delay constrains the stable learning rate

Correct attribution is not enough if feedback arrives late.

For the scalar reference recurrence

```math
\delta_{t+1}=\delta_t-a\,\delta_{t-D},
```

with feedback delay `D`, the characteristic polynomial is

```math
z^{D+1}-z^D+a=0.
```

For this restricted recurrence, stability requires

```math
0<a<2\sin\left(\frac{\pi}{4D+2}\right).
```

The exact reference boundaries used by the gate are:

| delay `D` | maximum stable effective gain |
|---:|---:|
| 0 | 2.000000 |
| 1 | 1.000000 |
| 3 | 0.445042 |

The code checks the characteristic roots immediately below and above each boundary: below, spectral radius is less than one; above, it is greater than one.

This is not a universal learning-rate law. It is an exact warning from a small delayed-feedback recurrence:

> **A learning rate that is stable with immediate feedback can oscillate or diverge when the same correctly identified feedback arrives late.**

## 9. What the next experiment must demonstrate

Gate 4 is a mathematical reference, not the breakthrough claim.

The next continuous four-branch task should force all of the difficult interactions to coexist:

```text
self-command
     |
     v
changing material ----> material-shaped response
     ^                         |
     |                         v
 delayed task error <---- bounded readout
     ^                         |
     |                         v
bounded receipts        echo prediction
     |                         |
     +----------- residual ----+
```

Requirements:

- material feedback stays enabled continuously;
- self-commands and external events overlap;
- the echo predictor keeps learning while material changes;
- teaching is derived from measured task error rather than scheduled signs;
- old responses share adjustable material with the new response;
- receipts have a fixed capacity;
- comparison methods receive the same windows, observations and feedback;
- private dither is one optional intervention, including its known attacker;
- success requires target improvement, preservation, calibrated attribution and bounded material growth.

The decisive missing result is:

> **Can the system obtain a useful approximation to the compensating direction from the information it is actually allowed to possess?**

That is where `IttnasNoruen` stops being a collection of integration controls and starts testing a potentially substantial mechanism.
