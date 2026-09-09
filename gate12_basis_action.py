"""Gate 12: use measured geometry to choose a basis, then preserve basis action.

Gate 10 selected examples using a Q-like response signature but protected only one-sided
survival inequalities. Gate 11 preserved the signature/Jacobian-like geometry itself.
Gate 12 instead freezes a measured basis at theta_0 and keeps the actual baseline
responses on that basis approximately unchanged during every accepted write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from behavioral_guard import BehavioralUpdateGuard, MeasurementBudgetExceeded, ResponseMeter
from gate8_retrieval_access import CueRecognizer, margins, view
from gate9_selected_replay import (
    CHECKPOINTS,
    CHECKPOINT_TOLERANCE,
    build_evaluator,
    make_pool,
    reached,
)
from gate10_q_coverage import (
    DEFAULT_SIGNATURE_EPS,
    DEFAULT_SIGNATURE_PROBES,
    _qspan_indices,
    measured_signatures,
    probe_directions,
    run_method as run_gate10_method,
    signature_geometry,
)

DEFAULT_EQUAL_REL_TOL = 0.05
DEFAULT_EQUAL_ABS_TOL = 0.002
NEW_METHODS = ("boundary_equal", "qspan_equal")
CONTROL_METHODS = ("qspan_scanned", "all_pool", "unprotected")


def response_tolerance(value, rel=DEFAULT_EQUAL_REL_TOL, floor=DEFAULT_EQUAL_ABS_TOL):
    if rel < 0 or floor < 0:
        raise ValueError("response tolerances must be nonnegative")
    return float(max(floor, rel * abs(float(value))))


def boundary_indices(pool, capacity=20):
    if capacity != 20:
        raise ValueError("Gate 12 fixes the active basis at 20")
    selected = []
    for label in range(10):
        candidates = [i for i, r in enumerate(pool) if int(r["label"]) == label]
        selected.extend(sorted(candidates, key=lambda i: (pool[i]["initial_margin"], i))[:2])
    return selected


def select_frozen_basis(method, pool, signatures):
    if method == "qspan_equal":
        return _qspan_indices(pool, signatures, capacity=20)
    if method == "boundary_equal":
        return boundary_indices(pool, capacity=20)
    raise ValueError("unknown Gate-12 method")


def basis_audit(theta, world, encoded_pool, pool, selected, baseline_values, tolerances):
    values = np.array([
        float(margins(
            world.encoded_probabilities(theta, encoded_pool[i]), [pool[i]["label"]]
        )[0])
        for i in selected
    ], dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    tolerances = np.asarray(tolerances, dtype=float)
    drift = np.abs(values - baseline_values)
    ratios = drift / np.maximum(tolerances, 1e-30)
    return dict(
        values=values.tolist(),
        max_abs_drift=float(np.max(drift, initial=0.0)),
        max_tolerance_ratio=float(np.max(ratios, initial=0.0)),
        violations=int(np.sum(drift > tolerances + 1e-12)),
    )


def run_equal_method(world, pool, new_images, new_labels, evaluate, method, seed, *,
                     checkpoints=CHECKPOINTS, attempts=24, budget=45000,
                     model_corrections=2,
                     signature_probes=DEFAULT_SIGNATURE_PROBES,
                     signature_eps=DEFAULT_SIGNATURE_EPS,
                     equal_rel_tol=DEFAULT_EQUAL_REL_TOL,
                     equal_abs_tol=DEFAULT_EQUAL_ABS_TOL):
    if method not in NEW_METHODS:
        raise ValueError("unknown Gate-12 method")

    encoded_pool = [world.encode(r["image"][None, :]) for r in pool]
    encoded_new = world.encode(new_images)
    example_evaluations = 0

    def response(theta, query):
        nonlocal example_evaluations
        if query == "new":
            example_evaluations += len(new_labels)
            probabilities = world.encoded_probabilities(theta, encoded_new)
            return float(np.mean(np.log(
                probabilities[np.arange(len(new_labels)), new_labels] + 1e-12)))
        i = int(query)
        example_evaluations += 1
        return float(margins(
            world.encoded_probabilities(theta, encoded_pool[i]), [pool[i]["label"]]
        )[0])

    theta = np.zeros(24)
    initial = response(theta, "new")
    example_evaluations = 0
    meter = ResponseMeter(response, budget)
    baseline = evaluate(theta)
    directions = probe_directions(seed, theta.size, signature_probes)

    try:
        current_margins, signatures = measured_signatures(
            theta, pool, meter, directions, signature_eps)
    except MeasurementBudgetExceeded:
        return dict(
            method=method, seed=seed, initial_objective=initial,
            final_objective=initial, actual_progress=0.0,
            stop_reason="measurement_budget_exhausted_during_basis_scan",
            checkpoints=[], trajectory=[], baseline=baseline, final=baseline,
            calls=meter.calls, budget=budget,
            model_example_evaluations=example_evaluations,
            active_capacity=0, final_parameters=theta.tolist(), parameter_norm=0.0,
        )

    selected = select_frozen_basis(method, pool, signatures)
    baseline_values = current_margins[np.asarray(selected, dtype=int)]
    tolerances = np.array([
        response_tolerance(v, equal_rel_tol, equal_abs_tol) for v in baseline_values
    ], dtype=float)
    geometry = signature_geometry(signatures, selected)
    scan_calls = meter.calls

    targets = [(float(g), min(-.001, initial + g)) for g in checkpoints]
    index = 0
    objective = initial
    records = []
    trace = []
    max_violation = 0.0
    stop = "attempt_limit"

    for attempt in range(attempts):
        if index >= len(targets):
            stop = "all_checkpoints_reached"
            break
        requested, target = targets[index]
        start = meter.calls
        guard = BehavioralUpdateGuard(
            len(selected), trust_radius=.5, projection_mode="equalities",
            max_model_corrections=model_corrections)
        for i, value, tolerance in zip(selected, baseline_values, tolerances):
            guard.remember(i, float(value), float(tolerance))
        decision = guard.step(theta, "new", target, meter, protect=True, validate=True)
        theta = decision.parameters
        if decision.max_constraint_violation is not None:
            max_violation = max(max_violation, float(decision.max_constraint_violation))
        if decision.target_error_after is not None:
            objective = target - decision.target_error_after
        audit = basis_audit(
            theta, world, encoded_pool, pool, selected, baseline_values, tolerances)
        trace.append(dict(
            attempt=attempt,
            status=decision.status,
            requested=requested,
            calls=meter.calls - start,
            total_calls=meter.calls,
            rejected_candidates=decision.rejected_candidates,
            max_constraint_violation=decision.max_constraint_violation,
            basis_max_tolerance_ratio=audit["max_tolerance_ratio"],
            basis_violations=audit["violations"],
        ))

        if reached(objective, target):
            records.append(dict(
                requested_progress=requested,
                actual_progress=objective - initial,
                objective=objective,
                target=target,
                calls=meter.calls,
                example_evaluations=example_evaluations,
                access=evaluate(theta),
                basis=audit,
                parameters=theta.tolist(),
            ))
            index += 1
            if index == len(targets):
                stop = "all_checkpoints_reached"
                break
        elif decision.status != "accepted":
            stop = decision.status
            break

    final_margins = np.array([
        float(margins(world.encoded_probabilities(theta, encoded_pool[i]), [pool[i]["label"]])[0])
        for i in range(len(pool))
    ])
    initial_margins = np.array([r["initial_margin"] for r in pool])
    bounds = np.array([r["bound"] for r in pool])
    final_audit = basis_audit(
        theta, world, encoded_pool, pool, selected, baseline_values, tolerances)
    return dict(
        method=method,
        seed=seed,
        initial_objective=initial,
        final_objective=objective,
        actual_progress=objective - initial,
        stop_reason=stop,
        checkpoints=records,
        trajectory=trace,
        baseline=baseline,
        final=evaluate(theta),
        calls=meter.calls,
        budget=budget,
        model_example_evaluations=example_evaluations,
        frozen_basis_scan_calls=scan_calls,
        signature_probes=signature_probes,
        signature_eps=signature_eps,
        equal_relative_tolerance=equal_rel_tol,
        equal_absolute_tolerance=equal_abs_tol,
        active_capacity=len(selected),
        selected=selected,
        selected_labels=[int(pool[i]["label"]) for i in selected],
        selected_baseline_values=baseline_values.tolist(),
        selected_tolerances=tolerances.tolist(),
        signature_geometry=geometry,
        basis=final_audit,
        max_accepted_constraint_violation=float(max_violation),
        final_pool_bound_violations=int(np.sum(final_margins < bounds - 1e-12)),
        final_pool_recognition_losses=int(np.sum((initial_margins > 0) & (final_margins <= 0))),
        parameter_norm=float(np.linalg.norm(theta)),
        final_parameters=theta.tolist(),
    )


def run_specimen(seed, *, checkpoints=CHECKPOINTS, attempts=24, budget=45000,
                 model_corrections=2,
                 signature_probes=DEFAULT_SIGNATURE_PROBES,
                 signature_eps=DEFAULT_SIGNATURE_EPS,
                 equal_rel_tol=DEFAULT_EQUAL_REL_TOL,
                 equal_abs_tol=DEFAULT_EQUAL_ABS_TOL):
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    images, labels = load_digits(return_X_y=True)
    images = images / 16.0
    pool_ids, test = train_test_split(
        np.arange(len(labels)), test_size=.2, stratify=labels, random_state=seed)
    train, calibration = train_test_split(
        pool_ids, test_size=.25, stratify=labels[pool_ids], random_state=seed + 1)
    old_cal, new_cal = train_test_split(
        calibration, test_size=.5, stratify=labels[calibration], random_state=seed + 2)
    new_cal = new_cal[:64]
    assert not (set(test) & (set(train) | set(old_cal) | set(new_cal)))

    with warnings.catch_warnings(record=True) as notes, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model = MLPClassifier(
            hidden_layer_sizes=(24,), activation="tanh", solver="lbfgs",
            max_iter=200, random_state=seed,
        ).fit(
            np.concatenate([view(images[train], v) for v in ("full", "upper", "lower")]),
            np.tile(labels[train], 3),
        )
        world = CueRecognizer(model, seed)
        pool, calibration_calls = make_pool(world, images, labels, old_cal)
        evaluate = build_evaluator(world, images[test], labels[test])
        shifted_new = .55 * images[new_cal] + .22
        new_runs = [
            run_equal_method(
                world, pool, shifted_new, labels[new_cal], evaluate, method, seed,
                checkpoints=checkpoints, attempts=attempts, budget=budget,
                model_corrections=model_corrections,
                signature_probes=signature_probes, signature_eps=signature_eps,
                equal_rel_tol=equal_rel_tol, equal_abs_tol=equal_abs_tol,
            )
            for method in NEW_METHODS
        ]
        controls = [
            run_gate10_method(
                world, pool, shifted_new, labels[new_cal], evaluate, method, seed,
                checkpoints=checkpoints, attempts=attempts, budget=budget,
                model_corrections=model_corrections,
                signature_probes=signature_probes, signature_eps=signature_eps,
            )
            for method in CONTROL_METHODS
        ]

    return dict(
        seed=seed,
        runs=new_runs + controls,
        versions=dict(numpy=np.__version__, sklearn=sklearn.__version__),
        training_warnings=[str(w.message) for w in notes],
        training_iterations=int(model.n_iter_),
        split_sizes={name: len(ids) for name, ids in (
            ("train", train), ("old_cal", old_cal), ("new_cal", new_cal), ("test", test))},
        split_sha256={name: hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
                      for name, ids in (("train", train), ("old_cal", old_cal),
                                        ("new_cal", new_cal), ("test", test))},
        baseline_calibration_calls=calibration_calls,
        initial_objective_calibration_examples=64,
        stored_pool=[{k: v for k, v in r.items() if k != "image"} for r in pool],
    )


def paired_summary(specimens, checkpoints):
    output = []
    for requested in checkpoints:
        for comparator in ("boundary_equal", "qspan_scanned", "all_pool", "unprotected"):
            pairs = []
            for specimen in specimens:
                runs = {r["method"]: r for r in specimen["runs"]}
                a = next((p for p in runs["qspan_equal"]["checkpoints"]
                          if p["requested_progress"] == requested), None)
                b = next((p for p in runs[comparator]["checkpoints"]
                          if p["requested_progress"] == requested), None)
                if a is not None and b is not None:
                    pairs.append((specimen["seed"], a, b))
            row = dict(
                requested_progress=requested,
                comparator=comparator,
                paired_seeds=[p[0] for p in pairs],
                coverage=len(pairs),
                total_seeds=len(specimens),
            )
            for family in ("seen", "unseen"):
                row[family + "_loss_difference"] = (
                    float(np.mean([
                        a["access"][family]["loss_rate_among_initially_correct"] -
                        b["access"][family]["loss_rate_among_initially_correct"]
                        for _, a, b in pairs
                    ])) if pairs else None
                )
            row["call_difference"] = (
                float(np.mean([a["calls"] - b["calls"] for _, a, b in pairs]))
                if pairs else None
            )
            output.append(row)
    return output


def gate12(seeds=(293, 307, 331), *, checkpoints=CHECKPOINTS, attempts=24,
           budget=45000, model_corrections=2,
           signature_probes=DEFAULT_SIGNATURE_PROBES,
           signature_eps=DEFAULT_SIGNATURE_EPS,
           equal_rel_tol=DEFAULT_EQUAL_REL_TOL,
           equal_abs_tol=DEFAULT_EQUAL_ABS_TOL):
    specimens = [
        run_specimen(
            seed, checkpoints=checkpoints, attempts=attempts, budget=budget,
            model_corrections=model_corrections,
            signature_probes=signature_probes, signature_eps=signature_eps,
            equal_rel_tol=equal_rel_tol, equal_abs_tol=equal_abs_tol,
        )
        for seed in seeds
    ]
    runs = [r for s in specimens for r in s["runs"]]
    equal_runs = [r for r in runs if r["method"] in NEW_METHODS]
    qspan_runs = [r for r in runs if r["method"] == "qspan_equal"]
    checks = dict(
        callback_budgets=all(r["calls"] <= budget for r in runs),
        bounded_pool=all(len(s["stored_pool"]) == 60 for s in specimens),
        measured_progress=all(
            reached(p["objective"], p["target"])
            for r in runs for p in r["checkpoints"]),
        frozen_basis_capacity=all(r["active_capacity"] == 20 for r in equal_runs),
        qspan_stratified=all(
            all(r["selected_labels"].count(label) == 2 for label in range(10))
            for r in qspan_runs),
        accepted_basis_contracts_validated=all(
            r["max_accepted_constraint_violation"] <= 1e-12 for r in equal_runs),
        final_basis_within_tolerance=all(r["basis"]["violations"] == 0 for r in equal_runs),
    )
    return dict(
        gate=12,
        status="frozen_basis_action_test",
        protocol=dict(
            seeds=list(seeds), checkpoints=list(checkpoints), attempts=attempts,
            budget=budget, stored_records=60, active_basis=20,
            adjustable_parameters=24, signature_probes=signature_probes,
            signature_eps=signature_eps,
            equal_relative_tolerance=equal_rel_tol,
            equal_absolute_tolerance=equal_abs_tol,
            basis_definition="frozen Gate-10 signature geometry at theta_0",
            protected_quantity="actual selected-cue margin relative to theta_0",
            progress_tolerance=CHECKPOINT_TOLERANCE,
            measured_model_corrections=model_corrections,
        ),
        resource_account=dict(
            frozen_signature_pool_current=60,
            frozen_signature_probe_calls=60 * signature_probes,
            note=("boundary_equal and qspan_equal pay the same one-time frozen signature scan. "
                  "All subsequent selected-basis response measurements, rejected candidates and "
                  "validations are charged. Test access evaluation is evaluator-only."),
        ),
        checks=checks,
        **{"pass": all(checks.values())},
        specimens=specimens,
        paired=paired_summary(specimens, checkpoints),
        limitations=[
            "the measured signature geometry is still only a black-box proxy for the exact CausalHorizon Q entry space",
            "the basis is frozen at theta_0 and may become stale after nonlinear learning",
            "two-sided response tolerances are preregistered but not theoretically optimal",
            "different controls have different acquisition structures; actual calls are reported",
            "single small dataset with resampled splits; matched-progress outcomes are descriptive",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[293, 307, 331])
    parser.add_argument("--checkpoints", type=float, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--attempts", type=int, default=24)
    parser.add_argument("--budget", type=int, default=45000)
    parser.add_argument("--model-corrections", type=int, default=2)
    parser.add_argument("--signature-probes", type=int, default=DEFAULT_SIGNATURE_PROBES)
    parser.add_argument("--signature-eps", type=float, default=DEFAULT_SIGNATURE_EPS)
    parser.add_argument("--equal-rel-tol", type=float, default=DEFAULT_EQUAL_REL_TOL)
    parser.add_argument("--equal-abs-tol", type=float, default=DEFAULT_EQUAL_ABS_TOL)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (args.attempts < 1 or args.budget < 1 or args.model_corrections < 0 or
            args.signature_probes < 1 or args.signature_eps <= 0 or
            args.equal_rel_tol < 0 or args.equal_abs_tol < 0):
        parser.error("invalid positive experiment parameters")
    result = gate12(
        seeds=tuple(args.seeds), checkpoints=tuple(args.checkpoints),
        attempts=args.attempts, budget=args.budget,
        model_corrections=args.model_corrections,
        signature_probes=args.signature_probes, signature_eps=args.signature_eps,
        equal_rel_tol=args.equal_rel_tol, equal_abs_tol=args.equal_abs_tol,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    else:
        print(text)
