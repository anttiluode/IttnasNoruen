"""Gate 11: preserve a frozen measured response-operator fingerprint directly.

Gate 10 used measured cue-response geometry only to choose replay examples and then
reduced those examples back to ordinary pointwise margin inequalities. Gate 11 freezes
a compact SVD-derived response sketch at theta_0 and exposes its mode gains as actual
black-box guard queries.

This is deliberately not called the exact CausalHorizon Q or HQB map. It is a finite,
counted, local response-operator sketch in the same software test bed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from behavioral_guard import BehavioralUpdateGuard, MeasurementBudgetExceeded
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
    measured_signatures,
    probe_directions,
    run_method as run_gate10_method,
)

DEFAULT_FINGERPRINT_REL_TOL = 0.15
DEFAULT_FINGERPRINT_ABS_TOL = 5e-4
NEW_METHODS = ("anchor_guard", "operator_guard")
CONTROL_METHODS = ("qspan_scanned", "all_pool", "unprotected")


class CostedResponseMeter:
    """ResponseMeter-compatible counter with explicit composite-query cost.

    A mode-gain response evaluates one old cue twice, at theta and theta+eps*w,
    and is therefore charged two scalar response calls. Ordinary margin and new-task
    aggregate queries keep the same callback accounting convention as Gates 9-10.
    """

    def __init__(self, response, max_calls, cost):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self._response = response
        self._cost = cost
        self.max_calls = int(max_calls)
        self.calls = 0

    def __call__(self, parameters, query):
        charge = int(self._cost(query))
        if charge < 1:
            raise ValueError("query cost must be positive")
        if self.calls + charge > self.max_calls:
            raise MeasurementBudgetExceeded("scalar response budget exhausted")
        self.calls += charge
        value = float(self._response(np.array(parameters, copy=True), query))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite measured response")
        return value


def frozen_operator_modes(signatures, directions, *, max_modes=None):
    """Build deterministic SVD modes and distinct high-leverage pivot cues.

    signatures: pool x K Gate-10 finite response signature matrix.
    directions: K x P orthonormal temporary parameter probe directions.

    Returns dictionaries with a fixed P-dimensional mode direction, singular value,
    pivot cue and leverage. Only numerically supported singular modes are retained.
    """
    signatures = np.asarray(signatures, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if signatures.ndim != 2 or directions.ndim != 2:
        raise ValueError("signatures and directions must be matrices")
    if signatures.shape[1] != directions.shape[0]:
        raise ValueError("signature width must match direction count")
    u, singular, vh = np.linalg.svd(signatures, full_matrices=False)
    if not len(singular):
        return []
    tol = max(1e-12, float(singular[0]) * 1e-9)
    rank = int(np.sum(singular > tol))
    count = rank if max_modes is None else min(rank, int(max_modes))
    if count < 0:
        raise ValueError("max_modes must be nonnegative")

    used = set()
    modes = []
    for j in range(count):
        order = np.argsort(-np.abs(u[:, j]), kind="stable")
        pivot = next((int(i) for i in order if int(i) not in used), int(order[0]))
        used.add(pivot)
        direction = vh[j] @ directions
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            raise RuntimeError("SVD mode collapsed in parameter space")
        direction = direction / norm
        modes.append(dict(
            mode=j,
            pivot=pivot,
            singular=float(singular[j]),
            leverage=float(abs(u[pivot, j])),
            direction=direction,
            predicted_linear_gain=float(singular[j] * u[pivot, j]),
        ))
    return modes


def fragile_class_anchors(pool):
    """One frozen lowest-initial-margin old cue per digit class."""
    anchors = []
    for label in range(10):
        candidates = [i for i, r in enumerate(pool) if int(r["label"]) == label]
        if not candidates:
            raise ValueError(f"missing class {label} from old-cue pool")
        anchors.append(min(candidates, key=lambda i: (pool[i]["initial_margin"], i)))
    return anchors


def fingerprint_tolerance(value, rel=DEFAULT_FINGERPRINT_REL_TOL,
                          floor=DEFAULT_FINGERPRINT_ABS_TOL):
    if rel < 0 or floor < 0:
        raise ValueError("fingerprint tolerances must be nonnegative")
    return float(max(floor, rel * abs(float(value))))


def fingerprint_audit(theta, world, encoded_pool, modes, baseline_gains,
                      tolerances, eps):
    """Post-run diagnostic; evaluator-only and not used to choose a write."""
    values = []
    for mode in modes:
        i = mode["pivot"]
        direction = mode["direction"]
        label = mode["label"]
        base = float(margins(world.encoded_probabilities(theta, encoded_pool[i]), [label])[0])
        shifted = float(margins(
            world.encoded_probabilities(theta + eps * direction, encoded_pool[i]), [label]
        )[0])
        values.append(shifted - base)
    values = np.asarray(values, dtype=float)
    baseline = np.asarray(baseline_gains, dtype=float)
    tolerances = np.asarray(tolerances, dtype=float)
    drift = np.abs(values - baseline)
    ratio = drift / np.maximum(tolerances, 1e-30)
    return dict(
        values=values.tolist(),
        max_abs_drift=float(np.max(drift, initial=0.0)),
        max_tolerance_ratio=float(np.max(ratio, initial=0.0)),
        violations=int(np.sum(drift > tolerances + 1e-12)),
    )


def run_operator_method(world, pool, new_images, new_labels, evaluate, method, seed, *,
                        checkpoints=CHECKPOINTS, attempts=24, budget=45000,
                        model_corrections=2,
                        signature_probes=DEFAULT_SIGNATURE_PROBES,
                        signature_eps=DEFAULT_SIGNATURE_EPS,
                        fingerprint_rel_tol=DEFAULT_FINGERPRINT_REL_TOL,
                        fingerprint_abs_tol=DEFAULT_FINGERPRINT_ABS_TOL):
    if method not in NEW_METHODS:
        raise ValueError("unknown Gate-11 method")

    encoded_pool = [world.encode(r["image"][None, :]) for r in pool]
    encoded_new = world.encode(new_images)
    example_evaluations = 0
    modes = []

    def raw_margin(theta, i):
        r = pool[int(i)]
        return float(margins(
            world.encoded_probabilities(theta, encoded_pool[int(i)]),
            [r["label"]],
        )[0])

    def response(theta, query):
        nonlocal example_evaluations
        if query == "new":
            example_evaluations += len(new_labels)
            probabilities = world.encoded_probabilities(theta, encoded_new)
            return float(np.mean(np.log(
                probabilities[np.arange(len(new_labels)), new_labels] + 1e-12)))
        if isinstance(query, tuple) and len(query) == 2 and query[0] == "gain":
            mode = modes[int(query[1])]
            i = mode["pivot"]
            example_evaluations += 2
            return raw_margin(theta + signature_eps * mode["direction"], i) - raw_margin(theta, i)
        example_evaluations += 1
        return raw_margin(theta, int(query))

    def query_cost(query):
        return 2 if isinstance(query, tuple) and len(query) == 2 and query[0] == "gain" else 1

    theta = np.zeros(24)
    initial = response(theta, "new")
    example_evaluations = 0
    meter = CostedResponseMeter(response, budget, query_cost)
    baseline = evaluate(theta)
    directions = probe_directions(seed, theta.size, signature_probes)

    # Both new methods pay the same frozen-geometry acquisition. anchor_guard ignores
    # the gain contracts after measuring them, isolating use of the operator sketch.
    try:
        _, signatures = measured_signatures(
            theta, pool, meter, directions, signature_eps)
        modes = frozen_operator_modes(signatures, directions, max_modes=signature_probes)
        for mode in modes:
            mode["label"] = int(pool[mode["pivot"]]["label"])
        baseline_gains = np.array([
            meter(theta, ("gain", j)) for j in range(len(modes))
        ], dtype=float)
    except MeasurementBudgetExceeded:
        return dict(
            method=method, seed=seed, initial_objective=initial, final_objective=initial,
            actual_progress=0.0, stop_reason="measurement_budget_exhausted_during_fingerprint",
            checkpoints=[], trajectory=[], baseline=baseline, final=baseline,
            calls=meter.calls, budget=budget, model_example_evaluations=example_evaluations,
            mode_count=0, anchor_count=0, fingerprint=None,
            final_parameters=theta.tolist(), parameter_norm=0.0,
        )

    tolerances = np.array([
        fingerprint_tolerance(v, fingerprint_rel_tol, fingerprint_abs_tol)
        for v in baseline_gains
    ], dtype=float)
    anchors = fragile_class_anchors(pool)
    geometry_calls = meter.calls
    mode_table = [dict(
        mode=m["mode"], pivot=m["pivot"], label=m["label"], singular=m["singular"],
        leverage=m["leverage"], predicted_linear_gain=m["predicted_linear_gain"],
        measured_gain=float(baseline_gains[j]), tolerance=float(tolerances[j]),
        direction=np.asarray(m["direction"]).tolist(),
    ) for j, m in enumerate(modes)]

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
        refs = len(anchors) + (len(modes) if method == "operator_guard" else 0)
        guard = BehavioralUpdateGuard(
            refs, trust_radius=.5, max_model_corrections=model_corrections)
        for i in anchors:
            guard.remember_range(i, minimum=pool[i]["bound"])
        if method == "operator_guard":
            for j, (value, tolerance) in enumerate(zip(baseline_gains, tolerances)):
                guard.remember(("gain", j), float(value), float(tolerance))

        decision = guard.step(theta, "new", target, meter, protect=True, validate=True)
        theta = decision.parameters
        if decision.max_constraint_violation is not None:
            max_violation = max(max_violation, float(decision.max_constraint_violation))
        if decision.target_error_after is not None:
            objective = target - decision.target_error_after
        audit = fingerprint_audit(
            theta, world, encoded_pool, modes, baseline_gains, tolerances, signature_eps)
        trace.append(dict(
            attempt=attempt,
            status=decision.status,
            requested=requested,
            calls=meter.calls - start,
            total_calls=meter.calls,
            max_constraint_violation=decision.max_constraint_violation,
            rejected_candidates=decision.rejected_candidates,
            fingerprint_max_tolerance_ratio=audit["max_tolerance_ratio"],
            fingerprint_violations=audit["violations"],
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
                fingerprint=audit,
                parameters=theta.tolist(),
            ))
            index += 1
            if index == len(targets):
                stop = "all_checkpoints_reached"
                break
        elif decision.status != "accepted":
            stop = decision.status
            break

    final_margins = np.array([raw_margin(theta, i) for i in range(len(pool))])
    initial_margins = np.array([r["initial_margin"] for r in pool])
    bounds = np.array([r["bound"] for r in pool])
    final_audit = fingerprint_audit(
        theta, world, encoded_pool, modes, baseline_gains, tolerances, signature_eps)
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
        frozen_geometry_calls=geometry_calls,
        signature_probes=signature_probes,
        signature_eps=signature_eps,
        fingerprint_relative_tolerance=fingerprint_rel_tol,
        fingerprint_absolute_tolerance=fingerprint_abs_tol,
        mode_count=len(modes),
        anchor_count=len(anchors),
        active_references=len(anchors) + (len(modes) if method == "operator_guard" else 0),
        anchors=anchors,
        modes=mode_table,
        fingerprint=final_audit,
        max_accepted_constraint_violation=float(max_violation),
        final_pool_bound_violations=int(np.sum(final_margins < bounds - 1e-12)),
        final_pool_recognition_losses=int(np.sum((initial_margins > 0) & (final_margins <= 0))),
        parameter_norm=float(np.linalg.norm(theta)),
        final_parameters=theta.tolist(),
    )


def run_specimen(seed, *, checkpoints=CHECKPOINTS, attempts=24, budget=45000,
                 model_corrections=2, signature_probes=DEFAULT_SIGNATURE_PROBES,
                 signature_eps=DEFAULT_SIGNATURE_EPS,
                 fingerprint_rel_tol=DEFAULT_FINGERPRINT_REL_TOL,
                 fingerprint_abs_tol=DEFAULT_FINGERPRINT_ABS_TOL):
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
            run_operator_method(
                world, pool, shifted_new, labels[new_cal], evaluate, method, seed,
                checkpoints=checkpoints, attempts=attempts, budget=budget,
                model_corrections=model_corrections, signature_probes=signature_probes,
                signature_eps=signature_eps,
                fingerprint_rel_tol=fingerprint_rel_tol,
                fingerprint_abs_tol=fingerprint_abs_tol,
            )
            for method in NEW_METHODS
        ]
        controls = [
            run_gate10_method(
                world, pool, shifted_new, labels[new_cal], evaluate, method, seed,
                checkpoints=checkpoints, attempts=attempts, budget=budget,
                model_corrections=model_corrections, signature_probes=signature_probes,
                signature_eps=signature_eps,
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
        for comparator in ("anchor_guard", "qspan_scanned", "all_pool", "unprotected"):
            pairs = []
            for specimen in specimens:
                runs = {r["method"]: r for r in specimen["runs"]}
                a = next((p for p in runs["operator_guard"]["checkpoints"]
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


def gate11(seeds=(251, 263, 277), *, checkpoints=CHECKPOINTS, attempts=24,
           budget=45000, model_corrections=2,
           signature_probes=DEFAULT_SIGNATURE_PROBES,
           signature_eps=DEFAULT_SIGNATURE_EPS,
           fingerprint_rel_tol=DEFAULT_FINGERPRINT_REL_TOL,
           fingerprint_abs_tol=DEFAULT_FINGERPRINT_ABS_TOL):
    specimens = [
        run_specimen(
            seed, checkpoints=checkpoints, attempts=attempts, budget=budget,
            model_corrections=model_corrections, signature_probes=signature_probes,
            signature_eps=signature_eps,
            fingerprint_rel_tol=fingerprint_rel_tol,
            fingerprint_abs_tol=fingerprint_abs_tol,
        )
        for seed in seeds
    ]
    runs = [r for s in specimens for r in s["runs"]]
    operator_runs = [r for r in runs if r["method"] == "operator_guard"]
    checks = dict(
        callback_budgets=all(r["calls"] <= budget for r in runs),
        bounded_pool=all(len(s["stored_pool"]) == 60 for s in specimens),
        measured_progress=all(
            reached(p["objective"], p["target"])
            for r in runs for p in r["checkpoints"]),
        frozen_modes_present=all(r.get("mode_count", 0) > 0 for r in operator_runs),
        operator_contracts_validated=all(
            r["max_accepted_constraint_violation"] <= 1e-12
            for r in operator_runs),
        final_operator_fingerprint_within_tolerance=all(
            r["fingerprint"]["violations"] == 0 for r in operator_runs),
    )
    return dict(
        gate=11,
        status="frozen_operator_fingerprint_test",
        protocol=dict(
            seeds=list(seeds), checkpoints=list(checkpoints), attempts=attempts,
            budget=budget, stored_records=60, adjustable_parameters=24,
            signature_probes=signature_probes, signature_eps=signature_eps,
            fingerprint_relative_tolerance=fingerprint_rel_tol,
            fingerprint_absolute_tolerance=fingerprint_abs_tol,
            mode_definition="SVD of frozen Gate-10 signature matrix; one finite mode-gain query per supported singular mode",
            anchors="one lowest-initial-margin cue per digit class",
            progress_tolerance=CHECKPOINT_TOLERANCE,
            measured_model_corrections=model_corrections,
        ),
        resource_account=dict(
            frozen_signature_pool_current=60,
            frozen_signature_probe_calls=60 * signature_probes,
            mode_gain_reference_cost=2,
            note=("anchor_guard and operator_guard pay the same initial frozen geometry acquisition. "
                  "operator_guard then pays extra per update because each protected mode-gain query "
                  "requires two old-cue margin measurements. Test access evaluation is evaluator-only."),
        ),
        checks=checks,
        **{"pass": all(checks.values())},
        specimens=specimens,
        paired=paired_summary(specimens, checkpoints),
        limitations=[
            "the SVD fingerprint is a finite black-box response sketch, not the exact CausalHorizon Q or HQB map",
            "the fingerprint is frozen at theta_0 and can become a poor local coordinate system after large writes",
            "one leverage pivot per singular mode is a compact sketch and does not preserve the entire 60x12 signature matrix",
            "the mode-gain tolerance is preregistered but not theoretically optimal",
            "different methods have different per-update acquisition costs; comparisons therefore report actual calls",
            "single small dataset with resampled splits; matched-progress outcomes are descriptive",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[251, 263, 277])
    parser.add_argument("--checkpoints", type=float, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--attempts", type=int, default=24)
    parser.add_argument("--budget", type=int, default=45000)
    parser.add_argument("--model-corrections", type=int, default=2)
    parser.add_argument("--signature-probes", type=int, default=DEFAULT_SIGNATURE_PROBES)
    parser.add_argument("--signature-eps", type=float, default=DEFAULT_SIGNATURE_EPS)
    parser.add_argument("--fingerprint-rel-tol", type=float, default=DEFAULT_FINGERPRINT_REL_TOL)
    parser.add_argument("--fingerprint-abs-tol", type=float, default=DEFAULT_FINGERPRINT_ABS_TOL)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (args.attempts < 1 or args.budget < 1 or args.model_corrections < 0 or
            args.signature_probes < 1 or args.signature_eps <= 0 or
            args.fingerprint_rel_tol < 0 or args.fingerprint_abs_tol < 0):
        parser.error("invalid positive experiment parameters")
    result = gate11(
        seeds=tuple(args.seeds), checkpoints=tuple(args.checkpoints),
        attempts=args.attempts, budget=args.budget,
        model_corrections=args.model_corrections,
        signature_probes=args.signature_probes, signature_eps=args.signature_eps,
        fingerprint_rel_tol=args.fingerprint_rel_tol,
        fingerprint_abs_tol=args.fingerprint_abs_tol,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    else:
        print(text)
