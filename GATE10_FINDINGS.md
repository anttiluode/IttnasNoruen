# Gate 10 findings — measured Q-coverage replay

The protocol was fixed before the full run in [`GATE10_PROTOCOL.md`](GATE10_PROTOCOL.md). The implementation is [`gate10_q_coverage.py`](gate10_q_coverage.py).

## Result

**Measured Q-span conditioning improved strongly. Held-out access did not.**

The full preregistered seeds were `211, 223, 239`. All budget, capacity, selected-contract, measured-progress, shared-signature-protocol and two-per-class checks passed. The full run was GitHub Actions run `34368355530` at head `4f787395fb34a81fd3f36e9d1eeada2c7ad019ae`.

The raw receipt produced by that run has SHA-256:

```text
f7879f5dc3affa5c2360ec144ced9fdc9f35f8d11738ce48150c4809344f3649
```

The uploaded Actions artifact ZIP has digest:

```text
sha256:4ce668cf7c954109253c872747cf1d23f5e4380e523da0850ba00fa9b8a11f2a
```

A compact frozen summary is in [`results/gate10/summary.json`](results/gate10/summary.json).

---

## 1. The new selector really changed the measured geometry

Every scanned method used the same 12 temporary parameter probes and paid for the same scan. All selected 20-record banks already had **rank 12 / ambient rank 12** on the first scan, so rank itself did not distinguish the methods.

Conditioning did.

| method | seed 211 condition | seed 223 | seed 239 | mean |
|---|---:|---:|---:|---:|
| random | 178.00 | 413.26 | 42.07 | 211.11 |
| Gate-9 interference | 201.60 | 176.87 | 42.93 | 140.47 |
| boundary | 21.04 | 25.51 | **8.01** | 18.19 |
| **Q-span** | **19.64** | **20.85** | 9.85 | **16.78** |

Relative to Gate-9 interference selection, Q-span improved the first-scan condition number by about **10.3x, 8.5x and 4.4x** across the three seeds. Relative to random selection the improvements were about **9.1x, 19.8x and 4.3x**.

The regularized log-determinant told the same story. Mean first-scan values were roughly:

```text
Q-span       -71.64
boundary     -72.38
random       -95.56
interference -91.45
```

So the D-optimal selector did the engineering job it was given.

But an unexpected control also did nearly as well: **the simple fragile-boundary bank was already very well conditioned** in this measured signature space. That matters below.

---

## 2. Better measured geometry did not improve held-out cue access

At matched progress `+0.05`, `+0.10` and `+0.20`, all three seeds support a Q-span versus Gate-9 interference comparison.

The difference in familiar-cue loss was:

```text
0.000000
0.000000
0.000000
```

and the difference in unseen-mask loss was:

```text
0.000000
0.000000
0.000000
```

At `+0.40`, only seed 239 gives a paired Q-span/interference comparison, and the differences are again exactly zero.

Q-span and the boundary selector are also tied on seen and unseen loss at every common checkpoint in this three-seed panel.

Against random replay, Q-span is not better either. At `+0.10` it has about **+0.000749** higher unseen loss; at `+0.20` it has about **-0.000352** lower familiar loss but **+0.000759** higher unseen loss. At `+0.40` only seed 239 pairs, where Q-span has about **+0.002278** higher unseen loss. These are tiny descriptive differences, not evidence for an adverse general effect, but they certainly do not establish an advantage.

So the preregistered behavioral hypothesis fails:

> **Better span/conditioning of this measured response-signature proxy did not translate into better preservation of unseen cue access.**

---

## 3. Learning progress also exposes the null

Final achieved progress for Q-span was:

```text
seed 211   0.212500   no acceptable step observed
seed 223   0.315964   measurement budget exhausted
seed 239   0.400000   all checkpoints reached
```

Gate-9 interference selection reached essentially the **same progress on all three seeds**.

The all-60 pool also lands on the same protected trajectory in important cases while spending substantially more measurements. More stored examples therefore still do not solve the capability-coverage problem by themselves.

---

## 4. Different banks often collapsed to the same update

The first selected Q-span and interference banks are visibly different. Nevertheless the accepted parameter paths often become identical or nearly identical.

Examples:

- seed 211 final `||theta_Q - theta_interference|| ~= 4e-13`
- seed 239 final `||theta_Q - theta_interference|| ~= 2.8e-16`
- seed 223 differs slightly at the budget-limited endpoint, but the `+0.05`, `+0.10` and `+0.20` checkpoint paths coincide to numerical precision or near it.

This gives the most useful diagnosis of Gate 10:

> **The learner measured a better coverage geometry, but its update rule often had no reason to use that geometry.**

The selected records enter the guard as pointwise lower-bound inequalities. If the same subset of constraints becomes active—or none of the geometric differences becomes active—very different spanning banks can produce the same feasible update.

Gate 10 therefore tested **bank selection**, not preservation of the entire measured span.

---

## 5. What this says about the CausalHorizon `Q` idea

It does **not** falsify the exact CausalHorizon coverage corollary.

In the linear theorem, if

```math
\Delta y(b)=H Q b
```

and a bank `B_bank` spans the relevant `Q b` space, then enforcing

```math
H Q B_{bank}=0
```

protects the entire spanned family by linearity.

Gate 10 did something weaker:

1. approximate `Q b` with finite margin-response signatures;
2. use those signatures only to **choose examples**;
3. return to an ordinary pointwise behavioral guard.

The span itself was never turned into an enforced object.

That gap now looks central.

---

## 6. Next gate: protect an operator/basis, not merely examples

The clean follow-up is no longer another replay selector.

Use the measured signature matrix to build a compact basis for the old capability, then make the guard preserve that basis directly.

One software version could be:

```text
candidate old cues
      |
      v
measured response-signature matrix S
      |
      v
rank-revealing basis / operator fingerprint
      |
      +--> synthesize basis probes or virtual constraints
      |
      v
finite candidate update
      |
      v
re-measure basis response after the update
      |
      v
accept only if the protected response operator remains within tolerance
```

The comparison should be against:

- the same 20 raw examples;
- the full 60 raw examples;
- Gate-9 vulnerability replay;
- Q-span example replay;
- a compact **basis/operator guard** using the same or lower total query/storage budget.

Held-out cue access still decides whether it worked.

This is closely related to the empirical-baseline/operator-fingerprint lesson in Active Dendrite: a panel of answers may be more useful when compressed into the map that generated them than when retained only as unrelated episodes.

---

## Claim boundary

D-optimal design, determinant subset selection, Jacobian/gradient sketches, reduced bases and subspace-preserving continual-learning methods are established areas.

Gate 10 establishes only this narrow result in this benchmark:

> **A D-optimal selector substantially improves conditioning of a measured cue-response signature bank, but that improvement does not improve held-out cue preservation when the chosen records are subsequently used only as ordinary pointwise constraints.**

That is a negative behavioral result and a positive diagnostic result.
