# Results and limits

These are three different experiments. Passing their engineering checks is not a claim of scientific superiority or a biological mechanism.

## Continuous four-branch integration

Means across seeds 7, 19 and 31; 640 five-tick windows, target 2→3 then 3→1.6, nonzero commands, noisy external returns, five-window delayed feedback, eight digital receipts and three retained questions. Each control gets the same actions, feedback opportunities and capacity ceilings.

| Method | Target RMSE before reversal | Target RMSE after reversal | Worst protected drift (mean of run maxima) | RMS protected drift |
| --- | ---: | ---: | ---: | ---: |
| proposal | 0.002208 | 0.001509 | 0.024947 | 0.003384 |
| naive | 0.002755 | 0.001551 | 0.528364 | 0.366851 |
| repair | 0.039796 | 0.035972 | 0.674748 | 0.169809 |
| frozen | 1.000000 | 0.400000 | 0.000000 | 0.000000 |

The query windows contain no self-only calibration and no zero-command replay. Material is constant within each four-tick balanced intervention; this controlled timescale separation is granted. The fifth tick supplies the noisy residual task output. The teacher's delayed error uses that emitted output, not evaluator-only clean activity.

All methods receive window IDs and task-version tokens. Receipts here are bounded, nondecaying digital records (six numerical values each plus bookkeeping), replacing the earlier decaying trace receipts. Three input/reference pairs cost 15 scalars; the latest measured sensitivity rows cost another 12. Working calibration buffers and correction messages are also counted in the JSON. Gate 6 still uses the gain-model sensitivity shortcut.

Hard gain bounds and a per-write trust radius are explicit controls, so bounded magnitude is not itself evidence of unconstrained stability.

### Attackers

| Condition | Final target RMSE | Worst protected drift | Final-window true echo error | Receipt overflows |
| --- | ---: | ---: | ---: | ---: |
| command_correlated_dither | 0.072315 | 0.138942 | 0.027430 | 0 |
| command_correlated_regression | 0.003586 | 0.022846 | 0.089820 | 0 |
| world_follows_dither | 0.000714 | 0.017038 | 0.094351 | 0 |
| receipt_overflow | 0.002124 | 0.011652 | 0.002135 | 384 |
| replay_overflow | 0.000934 | 0.500531 | 0.002118 | 0 |
| long_delay | 0.230350 | 0.028226 | 0.036719 | 0 |
| full_rank_protection | 0.400000 | 0.000000 | 0.002040 | 0 |
| stale_response_geometry | 0.000735 | 0.274105 | 0.002309 | 0 |

The confounded conditions expose a distinction between task performance and causal attribution. In this task, command-correlated external activity is a nuisance relative to the requested probe answer. A predictor that absorbs that activity can score well while misidentifying the self echo. Private dither gives cleaner causal identification only when the external process does not follow that dither; it is not automatically the best task denoiser. The exact fixed-material contamination is separately tested.

Overflow need not cause catastrophic failure in a redundant stream. Dropped receipts remain visible. Losing a protected replay and increasing delay have different consequences. The geometry attacker changes an input pathway externally, so its peak drift includes that exogenous change as well as the learner's subsequent behavior.

[Full continuous receipt](results/gate6_continuous.json)

## Transfer to a coupled nonlinear response

A 13-node recurrent tree has 12 adjustable couplings, leaky state, tanh nonlinearities and one soma readout. The guard receives a scalar callback and measures parameter sensitivities by finite differences. Four responses are protected to absolute tolerance 0.001; a fifth should increase by 0.08. Seeds 11, 23 and 47, at most 30 updates and 5,000 scalar calls per method.

| Method | Remaining target error | Worst retained drift | Worst unretained-probe drift | Scalar calls |
| --- | ---: | ---: | ---: | ---: |
| guard | 0.046546 | 0.000988 | 0.180455 | 3288.3 |
| tangent_only | 0.024177 | 0.032119 | 0.148650 | 955.0 |
| unprotected | -0.000000 | 0.087361 | 0.082094 | 400.0 |
| frozen | 0.080000 | 0.000000 | 0.000000 | 5.0 |

The guard respects the measured bank but does not solve every target within budget. A locally preserving tangent can violate the nonlinear contract after a finite step. Unretained questions can change substantially even when the retained bank stays within tolerance.

This is an external software test: each query starts from zero fast state and temporary parameter vectors can be evaluated without changing live material. The guard receives no analytic Jacobian, but this candidate-evaluation capability is stronger than the local continuous interface. This experiment and Gate 6 have not been combined into a continuous nonlinear arbor.

[Full nonlinear receipt](results/gate7_transfer.json)

## A trained classifier on handwritten digits

The same guard adjusts 24 hidden biases in a trained tanh classifier. Base training, calibration and test examples are disjoint. Adaptation uses 64 calibration images with contrast changed to 0.55*x+0.22. Ten correctly recognized calibration examples (one per class) supply protected probabilities. Test examples never select updates or stopping. Three seeds: 71, 83 and 97.

| Method | Original test accuracy before → after | Changed-contrast test accuracy before → after | Worst retained probability drift | Mean held-out probability drift |
| --- | ---: | ---: | ---: | ---: |
| guard | 95.83% → 95.65% | 92.41% → 92.41% | 0.000002 | 0.013829 |
| tangent_only | 95.83% → 95.65% | 92.41% → 92.41% | 0.000002 | 0.013829 |
| unprotected | 95.83% → 95.74% | 92.41% → 93.89% | 0.001011 | 0.014314 |
| frozen | 95.83% → 95.83% | 92.41% → 92.41% | 0.000000 | 0.000000 |

This example does not establish an advantage over ordinary adaptation. Exact local output preservation can be unnecessarily restrictive, and ten anchor responses do not characterize an entire recognition skill. The raw receipt includes worst held-out probability drift, which can be large despite small changes in average accuracy. The unprotected control may already satisfy the declared anchor tolerance.

All adaptive methods have the same 15,000-scalar-call ceiling and 24-update limit; actual costs differ and are reported. A scalar query on the adaptation batch averages 64 model evaluations, and the receipt counts those evaluations separately. Candidate model evaluation, retained input images, trained base-model weights, measured sensitivity/basis workspaces and labels are declared software resources.

[Full digits receipt](results/digits_transfer.json) · [Dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

## Reproduce

```bash
python -m pip install -r requirements-examples.txt
python -m pytest -q
python gate6_continuous_learning.py --output results/gate6_continuous.json
python gate7_measured_transfer.py --output results/gate7_transfer.json
python digits_transfer.py --output results/digits_transfer.json
python report_results.py
```

Full receipts preserve individual seeds, cost counts and failure cases. The result supports a reusable measured-update instrument and a continuous mechanism prototype. It does not establish a new continual-learning algorithm, protection of all unseen behavior, or a biological neuron implementing the observer.
