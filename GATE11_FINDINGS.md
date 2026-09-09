# Gate 11 findings — the static operator fingerprint is not enough

Gate 11 was preregistered before the result run in `GATE11_PROTOCOL.md` and executed on independent seeds `251, 263, 277`.

The implementation did what it was supposed to do. The behavioral hypothesis failed.

## Frozen result

GitHub Actions run: `34370992870`

Source branch head used by the PR run: `d1a53dafa5e27ece7d0ee681a2ae3367b3734da3`

Artifact: `gate11-receipt`, artifact id `10111922276`

Raw `gate11.json` SHA-256:

```text
decb92d863012c7c5ef9015d2738e6f99681502ff7c47d4e5842b90ccdedc757
```

Artifact ZIP SHA-256:

```text
b10298c7bc68734ad19db4429aba0a5e238eebd21f9ba0e6d97b29dfaaada9a7
```

All implementation/audit checks passed:

- bounded 60-record pool;
- online callback budgets respected;
- measured checkpoints valid;
- frozen singular modes present;
- accepted operator constraints validated;
- final protected fingerprint remained within its preregistered tolerance.

The full repo CI also passed on Python 3.11 and 3.12.

## What Gate 11 protected

At `theta_0`, Gate 11 measured the Gate-10 response-signature matrix

```math
S_{jk}=m_j(\theta_0+\epsilon d_k)-m_j(\theta_0)
```

for 60 old cues and 12 orthonormal temporary parameter probes. It froze the supported SVD modes of `S`, turned each right singular vector into a 24-dimensional parameter direction, chose a high-leverage pivot cue, and defined a finite mode-gain query

```math
\phi_r(\theta)
=
m_{i_r}(\theta+\epsilon w_r)-m_{i_r}(\theta).
```

`operator_guard` protected ten ordinary fragile class anchors plus all twelve frozen mode gains. `anchor_guard` paid for the same initial geometry acquisition but protected only the ten anchors.

Each gain was allowed to drift by at most

```math
\max(5\times10^{-4},\;0.15|\phi_r(\theta_0)|).
```

## The fingerprint really was preserved

Final maximum tolerance ratios for `operator_guard` were:

| seed | max fingerprint tolerance ratio | violations |
| ---: | ---: | ---: |
| 251 | `1.0000000000026` | 0 |
| 263 | `1.0000000000002` | 0 |
| 277 | `0.3782` | 0 |

By contrast, the unprotected fingerprint drift inside `anchor_guard` reached:

| seed | anchor-guard fingerprint ratio |
| ---: | ---: |
| 251 | `21.2440` |
| 263 | `2.9349` |
| 277 | `0.4636` |

So this is not a failure to enforce the new contract. We successfully held the measured operator sketch nearly fixed where it wanted to move.

## It did not preserve unseen access better

At matched `+0.05` progress, both `operator_guard` and `anchor_guard` reached the checkpoint on seeds 251 and 263.

Mean `operator_guard - anchor_guard` loss differences were:

```text
seen   = 0.000000
unseen = +0.00122549
```

Positive is worse for the operator guard.

The entire unseen difference came from seed 251:

| seed | method | unseen loss at +0.05 |
| ---: | --- | ---: |
| 251 | anchor_guard | `0.000000` |
| 251 | operator_guard | `0.00245098` |
| 263 | anchor_guard | `0.000000` |
| 263 | operator_guard | `0.000000` |

At matched `+0.10`, only seed 251 had coverage for both methods. There the operator guard was again worse:

```text
seen loss difference   = +0.00103199
unseen loss difference = +0.00490196
```

The same differences appear against Gate-10 `qspan_scanned` at those matched checkpoints.

There was no matched `+0.20` or `+0.40` coverage because the operator guard never reached those checkpoints.

## It was also expensive

The operator constraints rapidly became active on seeds 251 and 263.

Seed 251 reached:

- `+0.05` after 4,339 calls;
- `+0.10` after 20,579 calls;
- final progress `0.12609` at the 45,000-call budget.

Seed 263 reached:

- `+0.05` after 4,339 calls;
- final progress `0.07683` at the 45,000-call budget.

Seed 277 reached no preregistered checkpoint under any protected method; the operator fingerprint was not binding there, so that seed reflects a different bottleneck.

The anchor-only guard progressed farther with fewer calls on the first two seeds:

| seed | anchor final progress / calls | operator final progress / calls |
| ---: | ---: | ---: |
| 251 | `0.26348 / 17,997` | `0.12609 / 45,000` |
| 263 | `0.11193 / 16,974` | `0.07683 / 45,000` |
| 277 | `0.02926 / 18,382` | `0.02491 / 45,000` |

So the static operator sketch did not buy preservation cheaply. It mostly consumed degrees of freedom and measurement budget.

## This was not another identical-path null

Gate 10's strongest diagnostic was that different replay banks often produced effectively identical parameter trajectories.

Gate 11 is different.

On seed 251, at `+0.05`, the parameter distance between `operator_guard` and the three ordinary protected controls (`anchor_guard`, `qspan_scanned`, `all_pool`) was approximately

```text
0.02897
```

and at `+0.10` it was approximately

```text
0.10740.
```

The new constraint therefore changed the write substantially. It just did not improve the capability metric.

On seed 263 at `+0.05`, the paths were still identical because the mode constraints had not yet forced a distinct solution strongly enough. Later they became active and the operator run consumed the budget before `+0.10`.

## The important correction

Gate 11 exposes a conceptual mistake in the tempting reading of Gate 10.

The Gate-10 signature

```math
s_j[k]
=
m_j(\theta+\epsilon d_k)-m_j(\theta)
```

is a **parameter-sensitivity sketch of the old response**. It mixes several things: how a cue enters the model, the current downstream response, and how a particular parameter perturbation propagates.

CausalHorizon's exact linear statement is instead

```math
\Delta y(b)=H Q b.
```

If stored `Qb_j` span the relevant entry space, the theorem says to preserve the **actual action of the candidate change** on that basis:

```math
H Q B_{\mathrm{basis}} \approx 0.
```

It does **not** say that the old parameter Jacobian or its singular gains must remain fixed.

Gate 11 froze something closer to

```math
\frac{\partial m_b}{\partial \theta}
```

projected onto a measured parameter subspace. That can remain stable while the capability changes, or be forced stable even when harmless adaptation would naturally rotate the local Jacobian.

The result is therefore sharper than “Q failed.”

> **A static local parameter-response fingerprint is not the causal invariant we were looking for.**

## What follows

Before escalating all the way to a state-conditioned nonlinear `Q`, there is one cleaner test that more literally matches the CausalHorizon corollary.

Use the Gate-10 measured geometry only to identify a basis. Then protect the **actual old responses on that basis as two-sided near-equalities**, rather than:

- one-sided survival inequalities (Gate 10), or
- the parameter-sensitivity fingerprint itself (Gate 11).

That is:

```math
Q\text{-ish geometry}
\rightarrow
\text{basis cues}
\rightarrow
\boxed{\Delta m(B_{\mathrm{basis}})\approx0}
```

and test whether this predicts preservation outside the basis.

If even that fails while the chosen basis is well conditioned, the evidence for the next escalation becomes much stronger:

```text
static cue basis
    -> insufficient
static local parameter-response operator
    -> insufficient
therefore test state-/trajectory-conditioned entry geometry
```

That would be the point to build a genuinely dynamic `Q(theta, x, history, proposed_write)` rather than another replay selector.

## Claim boundary

This Gate 11 result is a negative result on one small deterministic classifier test bed. It does not establish that operator preservation is generally useless, nor that CausalHorizon applies to the nonlinear classifier.

What it does establish inside this experiment is narrower and clean:

1. the frozen measured mode gains were successfully preserved;
2. preserving them changed the learning trajectory and cost substantial measurements;
3. it did not improve held-out unseen cue access at matched progress;
4. therefore this particular static response fingerprint is not a useful sufficient invariant for the old capability.
