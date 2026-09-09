# Gate 14 findings — rolling anchors matter; random rotation beats the instantaneous Q-span by the long checkpoint

Gate 14 was preregistered before the result run and used independent seeds `389, 401, 419`.

It separated the confound left by Gate 13. Every method used the same **current pre-write response anchors** and paid for the same ordinary proposal plus fresh 60-cue / 12-probe signature acquisition. Only the protected cue identities differed.

## Frozen result

GitHub Actions run: `34374395669`

Source head: `5eeb97832c4eab0d7768e15eaa083014f5a1918e`

Artifact: `gate14-receipt`, artifact id `10113251183`

Raw `gate14.json` SHA-256:

```text
86b7273c594fd12ea7ec2bf6d2196a887f894e40ab8ea6b378b293a13928f1b7
```

Artifact ZIP SHA-256:

```text
e6645cb3b6caf0f3acf3ee1bd37d42d5f4be3348ccdd6d529e73484ce3a6ab56
```

Dedicated Gate-14 tests passed (`4 passed`). All experiment audits passed: bounded callback budget, 60-cue pool, measured checkpoints, 20-cue two-per-class basis, rolling pre-write anchor invariant, precommit basis selection, finite local validation, and genuinely frozen cue identities in the frozen control.

## Final runs

| seed | method | progress | calls | final unseen loss | distinct bases |
| ---: | --- | ---: | ---: | ---: | ---: |
| 389 | frozen Q-span + rolling anchor | 0.4000 | 23,466 | 0.03347 | 1 |
| 389 | random rolling + rolling anchor | 0.4000 | 19,601 | **0.02510** | 10 |
| 389 | Q-span rolling + rolling anchor | 0.4000 | 25,346 | 0.03347 | 13 |
| 401 | frozen Q-span + rolling anchor | 0.4000 | 36,815 | 0.04883 | 1 |
| 401 | random rolling + rolling anchor | 0.4000 | 29,085 | **0.02344** | 15 |
| 401 | Q-span rolling + rolling anchor | 0.4000 | 31,091 | 0.02734 | 16 |
| 419 | frozen Q-span + rolling anchor | 0.4000 | 42,581 | 0.03827 | 1 |
| 419 | random rolling + rolling anchor | 0.4000 | 32,866 | **0.02041** | 17 |
| 419 | Q-span rolling + rolling anchor | 0.3646 | 45,000 | 0.02041 | 23 |

The random rolling condition reached `+0.40` on all three seeds and had lower final unseen loss than the frozen basis on **all three**. The Q-span rolling condition reached `+0.40` on only two seeds.

## Mean unseen loss at matched checkpoints

| progress | frozen rolling-anchor | random rolling-anchor | Q-span rolling-anchor |
| ---: | ---: | ---: | ---: |
| +0.05 | 0.00564 | **0.00435** | 0.00494 |
| +0.10 | 0.01004 | 0.01050 | **0.00804** |
| +0.20 | 0.01809 | 0.01565 | **0.01498** |
| +0.40 | 0.04019 (3 seeds) | **0.02298 (3 seeds)** | 0.03041 (2 seeds) |

So instantaneous Q-span geometry had a modest early/middle advantage, but it did not become the best long-horizon strategy.

## Preregistered primary: Q-span rolling minus frozen rolling

Negative unseen-loss difference favors current-state Q-span selection.

```text
+0.05: -0.000697   coverage 3/3
+0.10: -0.001999   coverage 3/3
+0.20: -0.003109   coverage 3/3
+0.40: -0.010742   coverage 2/3
```

Thus dynamic Q-span selection does beat the frozen cue identity set at matched progress where paired. But that does **not** establish that Q-span geometry is the important mechanism, because the random rolling control is at least as revealing.

## Q-span rolling versus random rolling

```text
+0.05: +0.000591   Q-span slightly worse
+0.10: -0.002458   Q-span better
+0.20: -0.000672   Q-span slightly better
+0.40: +0.006137   Q-span worse, coverage 2/3
```

The differences are small early. By the long checkpoint, random rotation is better on the paired seeds, and unlike Q-span it reaches `+0.40` on seed 419.

## Random rolling versus frozen rolling

```text
+0.05: -0.001288
+0.10: +0.000459
+0.20: -0.002437
+0.40: -0.017205   coverage 3/3
```

At `+0.40`, random rotation reduces mean unseen loss by about **1.72 percentage points** versus keeping the same 20 cue identities, while also using about **7,103 fewer scalar calls** on average at that checkpoint.

## What Gate 14 changes

Gate 13 told us that abandoning a globally frozen `theta_0` contract helped greatly. Gate 14 now shows that two distinct ideas live inside that improvement:

1. **rolling local preservation is already powerful** — even the same fixed cue identities can be re-anchored locally and reach the full learning checkpoint on all three new seeds;
2. **changing which cues are watched over time helps further** — but the best long-horizon result here comes from random stratified rotation, not the instantaneous Q-span selector.

That makes the current strongest interpretation:

> **The bounded guard may need temporal coverage, not a single privileged basis.**

A basis can be excellent for the current local response geometry and still repeatedly inspect similar directions. A rotating bank can accumulate broader surveillance across a trajectory even if no one instant is geometrically optimal.

This is not a proof that randomness is intrinsically best. It says the present Q-like signature criterion has not earned that status. In this experiment, **when/which directions get revisited over time matters at least as much as instantaneous conditioning**.

## A more useful next object

The next selector should optimize a **history-dependent coverage ledger**, not only the current scan:

```text
current response geometry
+ directions insufficiently inspected recently
+ proposed-write damage / gain
+ measurement cost
-> next protected basis
```

For example, maintain a decaying information/coverage matrix over measured signatures and choose the next 20 cues for *incremental information gain relative to that history* rather than maximizing the determinant of the current batch alone.

That directly tests whether random rotation is winning because it accidentally supplies persistent excitation / temporal coverage.

## Claim boundary

This is still one small nonlinear classifier family with three independent resampled splits. The signature vectors remain only a black-box proxy for CausalHorizon's exact `Q`. Random rotation has not been shown generally optimal.

What Gate 14 supports is narrower and useful:

- rolling current pre-write preservation survives the Gate-13 confound;
- dynamic cue identity selection can beat a frozen cue set;
- the current instantaneous Q-span selector does not beat simple random rotation over the long checkpoint;
- a future protection architecture should treat **coverage across time** as a first-class variable rather than searching for one static or instantaneously optimal memory basis.
