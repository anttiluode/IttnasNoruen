# Gate 9 — selecting replay for the proposed change

Generated from [the final receipt](results/gate9_selected_replay.json) and [the initial receipt](results/gate9_initial_projection.json). See [protocol, resources and the recorded correction](GATE9_PROTOCOL.md).

Seeds: [151, 163, 179]; one dataset, resampled splits. These are descriptive comparisons, not independent datasets or a superiority significance test.

## Did each method actually reach the requested progress?

A checkpoint is an actual measured new-task log-probability improvement within 1e-6 of its target. Missing checkpoints stay missing. No interpolation.

| Method | +0.05 | +0.10 | +0.20 | +0.40 |
| --- | ---: | ---: | ---: | ---: |
| `frozen` | 0/3 | 0/3 | 0/3 | 0/3 |
| `unprotected` | 3/3 | 3/3 | 3/3 | 3/3 |
| `fixed_mixed` | 2/3 | 1/3 | 1/3 | 1/3 |
| `random_scanned` | 3/3 | 3/3 | 3/3 | 2/3 |
| `boundary_scanned` | 2/3 | 1/3 | 1/3 | 1/3 |
| `interference_scanned` | 3/3 | 1/3 | 1/3 | 1/3 |
| `all_pool` | 2/3 | 1/3 | 1/3 | 1/3 |

## Informed selection versus controls at matched progress

Each row uses only seeds on which **both** methods reached the checkpoint. Negative loss differences favor informed selection. Differences are percentage points in loss of initially correct cue responses. 'Unseen' means masks never offered to replay, on held-out images. A one-seed comparison cannot stand in for the full cohort.

| Progress | Comparator | Common seeds | Seen-cue loss difference | Unseen-mask loss difference | Extra scalar calls |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.05 | `unprotected` | 3/3 | +0.071 | +0.069 | +46 |
| 0.05 | `fixed_mixed` | 2/3 | +0.000 | +0.000 | +340 |
| 0.05 | `random_scanned` | 3/3 | +0.071 | +0.069 | -785 |
| 0.05 | `boundary_scanned` | 2/3 | +0.000 | +0.000 | +0 |
| 0.05 | `all_pool` | 2/3 | +0.000 | +0.000 | -3780 |
| 0.10 | `unprotected` | 1/3 | +0.000 | +0.000 | -519 |
| 0.10 | `fixed_mixed` | 1/3 | +0.000 | +0.000 | +510 |
| 0.10 | `random_scanned` | 1/3 | +0.000 | +0.000 | -1199 |
| 0.10 | `boundary_scanned` | 1/3 | +0.000 | +0.000 | +0 |
| 0.10 | `all_pool` | 1/3 | +0.000 | +0.000 | -5530 |
| 0.20 | `unprotected` | 1/3 | +0.000 | +0.000 | -1378 |
| 0.20 | `fixed_mixed` | 1/3 | +0.000 | +0.000 | +680 |
| 0.20 | `random_scanned` | 1/3 | +0.000 | +0.000 | -2398 |
| 0.20 | `boundary_scanned` | 1/3 | +0.000 | +0.000 | +0 |
| 0.20 | `all_pool` | 1/3 | +0.000 | +0.000 | -7400 |
| 0.40 | `unprotected` | 1/3 | +0.211 | +0.000 | -996 |
| 0.40 | `fixed_mixed` | 1/3 | +0.000 | -0.204 | +1041 |
| 0.40 | `random_scanned` | 1/3 | +0.211 | +0.000 | -2356 |
| 0.40 | `boundary_scanned` | 1/3 | +0.000 | +0.000 | +0 |
| 0.40 | `all_pool` | 1/3 | +0.000 | +0.000 | -11140 |

## Final states have unequal progress

This table exposes costs and failure, rather than ranking protection at equal learning. Pool losses refer to the 60 stored cues; selected constraints and protection of the entire pool are different.

| Method | Mean progress | Scalar calls | Example evaluations | Mean stored-cue losses | Mean held-out seen-cue losses | Mean unseen-mask losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen` | 0.000000 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| `unprotected` | 0.400000 | 11550 | 46200 | 3.33 | 18.67 | 16.67 |
| `fixed_mixed` | 0.170953 | 24738 | 98952 | 0.00 | 3.67 | 2.67 |
| `random_scanned` | 0.399984 | 19503 | 116733 | 2.33 | 15.67 | 18.33 |
| `boundary_scanned` | 0.170953 | 27805 | 158740 | 0.00 | 3.67 | 2.33 |
| `interference_scanned` | 0.177240 | 28575 | 161820 | 0.00 | 5.00 | 3.67 |
| `all_pool` | 0.165007 | 36181 | 73519 | 0.00 | 3.67 | 2.00 |

## Before and after measured nonlinear correction

The initial algorithm had no model-error corrections. The final one allows two per rejected candidate, charges those queries, and enforces the same actual response bounds. This change followed an audit of the first outcomes and is explicitly post hoc.

| Seed | Method | Initial progress | Corrected progress | Initial calls | Corrected calls | Final stop |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 151 | `fixed_mixed` | 0.021034 | 0.036523 | 6867 | 34734 | `attempt_limit` |
| 151 | `interference_scanned` | 0.021034 | 0.055384 | 7887 | 41124 | `attempt_limit` |
| 151 | `all_pool` | 0.021034 | 0.033199 | 19947 | 45000 | `measurement_budget_exhausted` |
| 163 | `fixed_mixed` | 0.023239 | 0.076336 | 6951 | 33117 | `attempt_limit` |
| 163 | `interference_scanned` | 0.023239 | 0.076336 | 7971 | 37197 | `attempt_limit` |
| 163 | `all_pool` | 0.023239 | 0.061821 | 20191 | 45000 | `measurement_budget_exhausted` |
| 179 | `fixed_mixed` | 0.082136 | 0.400000 | 7812 | 6363 | `all_checkpoints_reached` |
| 179 | `interference_scanned` | 0.082136 | 0.400000 | 9002 | 7404 | `all_checkpoints_reached` |
| 179 | `all_pool` | 0.082136 | 0.400000 | 22692 | 18544 | `all_checkpoints_reached` |

Engineering checks: `callback_budgets=True`, `bounded_pool=True`, `selected_contracts=True`, `measured_progress=True`, `active_capacity=True`.

These checks establish bookkeeping and observed constraint compliance. They do not require an accuracy advantage or certify unmeasured behavior.
