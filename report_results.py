"""Render committed numerical receipts without silently changing their experiments."""
import json
from pathlib import Path


def render():
    root=Path(__file__).resolve().parent
    g6=json.loads((root/'results/gate6_continuous.json').read_text())
    g7=json.loads((root/'results/gate7_transfer.json').read_text())
    digits=json.loads((root/'results/digits_transfer.json').read_text())
    lines=["# Results and limits", "",
           "These are three different experiments. Passing their engineering checks is not a claim of scientific superiority or a biological mechanism.", "",
           "## Continuous four-branch integration", "",
           "Means across seeds 7, 19 and 31; 640 five-tick windows, target 2→3 then 3→1.6, nonzero commands, noisy external returns, five-window delayed feedback, eight digital receipts and three retained questions. Each control gets the same actions, feedback opportunities and capacity ceilings.", "",
           "| Method | Target RMSE before reversal | Target RMSE after reversal | Worst protected drift (mean of run maxima) | RMS protected drift |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for name,s in g6['summaries'].items():
        lines.append(f"| {name} | {s['before_reversal_rmse']:.6f} | {s['final_window_rmse']:.6f} | {s['max_protected_drift']:.6f} | {s['rms_protected_drift']:.6f} |")
    lines += ["", "The query windows contain no self-only calibration and no zero-command replay. Material is constant within each four-tick balanced intervention; this controlled timescale separation is granted. The fifth tick supplies the noisy residual task output. The teacher's delayed error uses that emitted output, not evaluator-only clean activity.", "",
              "All methods receive window IDs and task-version tokens. Receipts here are bounded, nondecaying digital records (six numerical values each plus bookkeeping), replacing the earlier decaying trace receipts. Three input/reference pairs cost 15 scalars; the latest measured sensitivity rows cost another 12. Working calibration buffers and correction messages are also counted in the JSON. Gate 6 still uses the gain-model sensitivity shortcut.", "",
              "Hard gain bounds and a per-write trust radius are explicit controls, so bounded magnitude is not itself evidence of unconstrained stability.", "",
              "### Attackers", "",
              "| Condition | Final target RMSE | Worst protected drift | Final-window true echo error | Receipt overflows |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name,s in g6['attackers'].items():
        lines.append(f"| {name} | {s['final_window_rmse']:.6f} | {s['max_protected_drift']:.6f} | {s['final_window_echo_error']:.6f} | {s['counts'].get('receipt_overflows',0)} |")
    lines += ["", "The confounded conditions expose a distinction between task performance and causal attribution. In this task, command-correlated external activity is a nuisance relative to the requested probe answer. A predictor that absorbs that activity can score well while misidentifying the self echo. Private dither gives cleaner causal identification only when the external process does not follow that dither; it is not automatically the best task denoiser. The exact fixed-material contamination is separately tested.", "",
              "Overflow need not cause catastrophic failure in a redundant stream. Dropped receipts remain visible. Losing a protected replay and increasing delay have different consequences. The geometry attacker changes an input pathway externally, so its peak drift includes that exogenous change as well as the learner's subsequent behavior.", "",
              "[Full continuous receipt](results/gate6_continuous.json)", "",
              "## Transfer to a coupled nonlinear response", "",
              "A 13-node recurrent tree has 12 adjustable couplings, leaky state, tanh nonlinearities and one soma readout. The guard receives a scalar callback and measures parameter sensitivities by finite differences. Four responses are protected to absolute tolerance 0.001; a fifth should increase by 0.08. Seeds 11, 23 and 47, at most 30 updates and 5,000 scalar calls per method.", "",
              "| Method | Remaining target error | Worst retained drift | Worst unretained-probe drift | Scalar calls |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name,s in g7['summaries'].items():
        lines.append(f"| {name} | {s['remaining_error']:.6f} | {s['max_protected_drift']:.6f} | {s['max_unprotected_probe_drift']:.6f} | {s['scalar_calls']:.1f} |")
    lines += ["", "The guard respects the measured bank but does not solve every target within budget. A locally preserving tangent can violate the nonlinear contract after a finite step. Unretained questions can change substantially even when the retained bank stays within tolerance.", "",
              "This is an external software test: each query starts from zero fast state and temporary parameter vectors can be evaluated without changing live material. The guard receives no analytic Jacobian, but this candidate-evaluation capability is stronger than the local continuous interface. This experiment and Gate 6 have not been combined into a continuous nonlinear arbor.", "",
              "[Full nonlinear receipt](results/gate7_transfer.json)", "",
              "## A trained classifier on handwritten digits", "",
              "The same guard adjusts 24 hidden biases in a trained tanh classifier. Base training, calibration and test examples are disjoint. Adaptation uses 64 calibration images with contrast changed to 0.55*x+0.22. Ten correctly recognized calibration examples (one per class) supply protected probabilities. Test examples never select updates or stopping. Three seeds: 71, 83 and 97.", "",
              "| Method | Original test accuracy before → after | Changed-contrast test accuracy before → after | Worst retained probability drift | Mean held-out probability drift |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name,s in digits['summaries'].items():
        lines.append(f"| {name} | {100*s['old_test_accuracy_before']:.2f}% → {100*s['old_test_accuracy_after']:.2f}% | {100*s['changed_test_accuracy_before']:.2f}% → {100*s['changed_test_accuracy_after']:.2f}% | {s['max_retained_anchor_drift']:.6f} | {s['heldout_mean_probability_drift']:.6f} |")
    lines += ["", "This example does not establish an advantage over ordinary adaptation. Exact local output preservation can be unnecessarily restrictive, and ten anchor responses do not characterize an entire recognition skill. The raw receipt includes worst held-out probability drift, which can be large despite small changes in average accuracy. The unprotected control may already satisfy the declared anchor tolerance.", "",
              "All adaptive methods have the same 15,000-scalar-call ceiling and 24-update limit; actual costs differ and are reported. A scalar query on the adaptation batch averages 64 model evaluations, and the receipt counts those evaluations separately. Candidate model evaluation, retained input images, trained base-model weights, measured sensitivity/basis workspaces and labels are declared software resources.", "",
              "[Full digits receipt](results/digits_transfer.json) · [Dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)", "",
              "## Reproduce", "", "```bash", "python -m pip install -r requirements-examples.txt", "python -m pytest -q",
              "python gate6_continuous_learning.py --output results/gate6_continuous.json",
              "python gate7_measured_transfer.py --output results/gate7_transfer.json",
              "python digits_transfer.py --output results/digits_transfer.json",
              "python report_results.py", "```", "",
              "Full receipts preserve individual seeds, cost counts and failure cases. The result supports a reusable measured-update instrument and a continuous mechanism prototype. It does not establish a new continual-learning algorithm, protection of all unseen behavior, or a biological neuron implementing the observer.", ""]
    (root/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    render()
