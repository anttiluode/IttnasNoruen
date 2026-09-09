"""Gate 14: isolate rolling anchors from dynamic basis selection.

Gate 13 mixed two changes: a basis that could move with theta and response anchors that
were reset to the current pre-write values. Gate 14 gives every method the same rolling
anchor law and varies only which 20 cue identities are protected on each atomic write.
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
from gate9_selected_replay import CHECKPOINTS, CHECKPOINT_TOLERANCE, build_evaluator, make_pool, reached
from gate10_q_coverage import (
    DEFAULT_SIGNATURE_EPS,
    DEFAULT_SIGNATURE_PROBES,
    _qspan_indices,
    measured_signatures,
    probe_directions,
    signature_geometry,
)
from gate12_basis_action import basis_audit, response_tolerance
from gate13_dynamic_q import generic_directions, _jaccard, _two_per_class

METHODS = (
    "frozen_qspan_rolling_anchor",
    "random_rolling_anchor",
    "qspan_rolling_anchor",
)

DEFAULT_EQUAL_REL_TOL = 0.05
DEFAULT_EQUAL_ABS_TOL = 0.002


def random_stratified_basis(pool, seed: int, attempt: int, capacity: int = 20):
    """Deterministic fresh random two-per-class basis."""
    if capacity != 20:
        raise ValueError("Gate 14 fixes active basis capacity at 20")
    rng = np.random.default_rng(seed + 70001 + 104729 * attempt)
    selected = []
    for label in range(10):
        candidates = [i for i, row in enumerate(pool) if int(row["label"]) == label]
        selected.extend(int(i) for i in rng.choice(candidates, 2, replace=False))
    return selected


def select_basis(method, pool, signatures, frozen_selected, seed, attempt):
    if method == "frozen_qspan_rolling_anchor":
        return list(frozen_selected)
    if method == "random_rolling_anchor":
        return random_stratified_basis(pool, seed, attempt)
    if method == "qspan_rolling_anchor":
        return _qspan_indices(pool, signatures, capacity=20)
    raise ValueError("unknown Gate-14 method")


def run_method(
    world,
    pool,
    new_images,
    new_labels,
    evaluate,
    method,
    seed,
    *,
    checkpoints=CHECKPOINTS,
    attempts=24,
    budget=45000,
    signature_probes=DEFAULT_SIGNATURE_PROBES,
    signature_eps=DEFAULT_SIGNATURE_EPS,
    equal_rel_tol=DEFAULT_EQUAL_REL_TOL,
    equal_abs_tol=DEFAULT_EQUAL_ABS_TOL,
):
    if method not in METHODS:
        raise ValueError("unknown Gate-14 method")

    encoded_pool = [world.encode(row["image"][None, :]) for row in pool]
    encoded_new = world.encode(new_images)
    example_evaluations = 0

    def response(theta, query):
        nonlocal example_evaluations
        if query == "new":
            example_evaluations += len(new_labels)
            probabilities = world.encoded_probabilities(theta, encoded_new)
            return float(np.mean(np.log(
                probabilities[np.arange(len(new_labels)), new_labels] + 1e-12
            )))
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

    # Common paid theta_0 scan defines only the frozen Q-span control. Other methods
    # pay the same scan but do not receive a selection advantage from it.
    initial_panel = probe_directions(seed, theta.size, signature_probes)
    try:
        _, initial_signatures = measured_signatures(
            theta, pool, meter, initial_panel, signature_eps
        )
    except MeasurementBudgetExceeded:
        return dict(
            method=method, seed=seed, initial_objective=initial,
            final_objective=initial, actual_progress=0.0,
            stop_reason="measurement_budget_exhausted_during_initial_scan",
            checkpoints=[], trajectory=[], baseline=baseline, final=baseline,
            calls=meter.calls, budget=budget, model_example_evaluations=example_evaluations,
            active_capacity=0, final_parameters=theta.tolist(), parameter_norm=0.0,
        )

    frozen_selected = _qspan_indices(pool, initial_signatures, capacity=20)
    if not _two_per_class(pool, frozen_selected):
        raise RuntimeError("frozen basis violated stratification")
    initial_scan_calls = meter.calls

    targets = [(float(g), min(-0.001, initial + g)) for g in checkpoints]
    index = 0
    objective = initial
    records = []
    trace = []
    selected_history = []
    max_accepted_violation = 0.0
    max_local_ratio = 0.0
    stop = "attempt_limit"

    for attempt in range(attempts):
        if index >= len(targets):
            stop = "all_checkpoints_reached"
            break
        requested, target = targets[index]
        attempt_start = meter.calls
        scan_start = meter.calls

        try:
            # All methods pay for the same ordinary intended write, even though Gate 14
            # does not use its direction for basis selection.
            ordinary = BehavioralUpdateGuard(1, trust_radius=0.5)
            proposal = ordinary.step(theta, "new", target, meter, protect=False, validate=False)
            if proposal.status not in ("accepted", "target_met"):
                stop = "proposal_" + proposal.status
                break
            proposed_step = proposal.parameters - theta
            proposal_norm = float(np.linalg.norm(proposed_step))
            if proposal_norm < 1e-12:
                stop = "proposal_no_direction"
                break

            panel = generic_directions(seed, attempt, theta.size, signature_probes)
            current_margins, signatures = measured_signatures(
                theta, pool, meter, panel, signature_eps
            )
            scan_calls = meter.calls - scan_start

            selected = select_basis(
                method, pool, signatures, frozen_selected, seed, attempt
            )
            if not _two_per_class(pool, selected):
                raise RuntimeError("Gate-14 basis violated the two-per-class contract")

            # Crucial Gate-14 invariant: ALL methods use current pre-write anchors.
            anchor_values = current_margins[np.asarray(selected, dtype=int)]
            tolerances = np.array([
                response_tolerance(v, equal_rel_tol, equal_abs_tol)
                for v in anchor_values
            ], dtype=float)
            geometry = signature_geometry(signatures, selected)
            previous = selected_history[-1] if selected_history else frozen_selected
            overlap_previous = _jaccard(previous, selected)
            overlap_frozen = _jaccard(frozen_selected, selected)
            selected_history.append(list(selected))

            # Anti-cheating boundary: cue identities + current anchors freeze here.
            guard = BehavioralUpdateGuard(
                len(selected), trust_radius=0.5, projection_mode="equalities"
            )
            for i, value, tolerance in zip(selected, anchor_values, tolerances):
                guard.remember(i, float(value), float(tolerance))
            decision = guard.step(theta, "new", target, meter, protect=True, validate=True)
            theta = decision.parameters
            if decision.max_constraint_violation is not None:
                max_accepted_violation = max(
                    max_accepted_violation, float(decision.max_constraint_violation)
                )
            if decision.target_error_after is not None:
                objective = target - decision.target_error_after

            local_audit = basis_audit(
                theta, world, encoded_pool, pool, selected, anchor_values, tolerances
            )
            max_local_ratio = max(max_local_ratio, local_audit["max_tolerance_ratio"])
            trace.append(dict(
                attempt=attempt,
                status=decision.status,
                requested=requested,
                calls=meter.calls - attempt_start,
                total_calls=meter.calls,
                scan_calls=scan_calls,
                proposal_norm=proposal_norm,
                selected=list(selected),
                selected_labels=[int(pool[i]["label"]) for i in selected],
                overlap_previous=overlap_previous,
                overlap_frozen=overlap_frozen,
                signature_geometry=geometry,
                local_basis=local_audit,
                anchors_are_current_prewrite=True,
                basis_chosen_before_guard=True,
                rejected_candidates=decision.rejected_candidates,
                max_constraint_violation=decision.max_constraint_violation,
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
                    selected=list(selected),
                    overlap_frozen=overlap_frozen,
                    signature_geometry=geometry,
                    local_basis=local_audit,
                    parameters=theta.tolist(),
                ))
                index += 1
                if index == len(targets):
                    stop = "all_checkpoints_reached"
                    break
            elif decision.status != "accepted":
                stop = decision.status
                break

        except MeasurementBudgetExceeded:
            stop = "measurement_budget_exhausted"
            break

    final_margins = np.array([
        float(margins(
            world.encoded_probabilities(theta, encoded_pool[i]), [pool[i]["label"]]
        )[0])
        for i in range(len(pool))
    ])
    initial_pool_margins = np.array([row["initial_margin"] for row in pool])
    bounds = np.array([row["bound"] for row in pool])
    distinct_bases = len({tuple(s) for s in selected_history})
    mean_overlap_previous = (
        float(np.mean([t["overlap_previous"] for t in trace])) if trace else None
    )
    mean_overlap_frozen = (
        float(np.mean([t["overlap_frozen"] for t in trace])) if trace else None
    )

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
        initial_scan_calls=initial_scan_calls,
        signature_probes=signature_probes,
        signature_eps=signature_eps,
        equal_relative_tolerance=equal_rel_tol,
        equal_absolute_tolerance=equal_abs_tol,
        active_capacity=20,
        frozen_selected=list(frozen_selected),
        distinct_bases=distinct_bases,
        mean_overlap_previous=mean_overlap_previous,
        mean_overlap_frozen=mean_overlap_frozen,
        max_local_tolerance_ratio=float(max_local_ratio),
        max_accepted_constraint_violation=float(max_accepted_violation),
        final_pool_bound_violations=int(np.sum(final_margins < bounds - 1e-12)),
        final_pool_recognition_losses=int(np.sum(
            (initial_pool_margins > 0) & (final_margins <= 0)
        )),
        parameter_norm=float(np.linalg.norm(theta)),
        final_parameters=theta.tolist(),
    )


def run_specimen(
    seed,
    *,
    checkpoints=CHECKPOINTS,
    attempts=24,
    budget=45000,
    signature_probes=DEFAULT_SIGNATURE_PROBES,
    signature_eps=DEFAULT_SIGNATURE_EPS,
    equal_rel_tol=DEFAULT_EQUAL_REL_TOL,
    equal_abs_tol=DEFAULT_EQUAL_ABS_TOL,
):
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    images, labels = load_digits(return_X_y=True)
    images = images / 16.0
    pool_ids, test = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed
    )
    train, calibration = train_test_split(
        pool_ids, test_size=0.25, stratify=labels[pool_ids], random_state=seed + 1
    )
    old_cal, new_cal = train_test_split(
        calibration, test_size=0.5, stratify=labels[calibration], random_state=seed + 2
    )
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
        shifted_new = 0.55 * images[new_cal] + 0.22
        runs = [
            run_method(
                world, pool, shifted_new, labels[new_cal], evaluate, method, seed,
                checkpoints=checkpoints, attempts=attempts, budget=budget,
                signature_probes=signature_probes, signature_eps=signature_eps,
                equal_rel_tol=equal_rel_tol, equal_abs_tol=equal_abs_tol,
            )
            for method in METHODS
        ]

    return dict(
        seed=seed,
        runs=runs,
        versions=dict(numpy=np.__version__, sklearn=sklearn.__version__),
        training_warnings=[str(w.message) for w in notes],
        training_iterations=int(model.n_iter_),
        split_sizes={name: len(ids) for name, ids in (
            ("train", train), ("old_cal", old_cal), ("new_cal", new_cal), ("test", test)
        )},
        split_sha256={
            name: hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
            for name, ids in (
                ("train", train), ("old_cal", old_cal),
                ("new_cal", new_cal), ("test", test)
            )
        },
        baseline_calibration_calls=calibration_calls,
        initial_objective_calibration_examples=64,
        stored_pool=[{k: v for k, v in row.items() if k != "image"} for row in pool],
    )


def _paired(specimens, requested, a_name, b_name):
    pairs = []
    for specimen in specimens:
        runs = {r["method"]: r for r in specimen["runs"]}
        a = next((p for p in runs[a_name]["checkpoints"]
                  if p["requested_progress"] == requested), None)
        b = next((p for p in runs[b_name]["checkpoints"]
                  if p["requested_progress"] == requested), None)
        if a is not None and b is not None:
            pairs.append((specimen["seed"], a, b))
    row = dict(
        requested_progress=requested,
        method_a=a_name,
        method_b=b_name,
        paired_seeds=[x[0] for x in pairs],
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
    return row


def paired_summary(specimens, checkpoints):
    rows = []
    comparisons = (
        ("qspan_rolling_anchor", "frozen_qspan_rolling_anchor"),
        ("qspan_rolling_anchor", "random_rolling_anchor"),
        ("random_rolling_anchor", "frozen_qspan_rolling_anchor"),
    )
    for requested in checkpoints:
        for a, b in comparisons:
            rows.append(_paired(specimens, requested, a, b))
    return rows


def gate14(
    seeds=(389, 401, 419),
    *,
    checkpoints=CHECKPOINTS,
    attempts=24,
    budget=45000,
    signature_probes=DEFAULT_SIGNATURE_PROBES,
    signature_eps=DEFAULT_SIGNATURE_EPS,
    equal_rel_tol=DEFAULT_EQUAL_REL_TOL,
    equal_abs_tol=DEFAULT_EQUAL_ABS_TOL,
):
    specimens = [
        run_specimen(
            seed, checkpoints=checkpoints, attempts=attempts, budget=budget,
            signature_probes=signature_probes, signature_eps=signature_eps,
            equal_rel_tol=equal_rel_tol, equal_abs_tol=equal_abs_tol,
        )
        for seed in seeds
    ]
    runs = [r for specimen in specimens for r in specimen["runs"]]
    checks = dict(
        callback_budgets=all(r["calls"] <= budget for r in runs),
        bounded_pool=all(len(s["stored_pool"]) == 60 for s in specimens),
        measured_progress=all(
            reached(p["objective"], p["target"])
            for r in runs for p in r["checkpoints"]
        ),
        active_capacity=all(r["active_capacity"] == 20 for r in runs),
        rolling_anchor_invariant=all(
            t["anchors_are_current_prewrite"] for r in runs for t in r["trajectory"]
        ),
        anti_cheating_precommit=all(
            t["basis_chosen_before_guard"] for r in runs for t in r["trajectory"]
        ),
        stratified_basis=all(
            all(t["selected_labels"].count(label) == 2 for label in range(10))
            for r in runs for t in r["trajectory"]
        ),
        accepted_local_contracts_validated=all(
            r["max_accepted_constraint_violation"] <= 1e-12 for r in runs
        ),
        frozen_basis_really_frozen=all(
            r["distinct_bases"] <= 1
            for r in runs if r["method"] == "frozen_qspan_rolling_anchor"
        ),
    )
    return dict(
        gate=14,
        status="local_anchor_vs_dynamic_basis_test",
        protocol=dict(
            seeds=list(seeds), checkpoints=list(checkpoints), attempts=attempts,
            budget=budget, stored_records=60, active_basis=20,
            adjustable_parameters=24, signature_probes=signature_probes,
            signature_eps=signature_eps,
            equal_relative_tolerance=equal_rel_tol,
            equal_absolute_tolerance=equal_abs_tol,
            progress_tolerance=CHECKPOINT_TOLERANCE,
            methods=list(METHODS),
            primary="qspan_rolling_anchor - frozen_qspan_rolling_anchor unseen loss",
            secondary=[
                "qspan_rolling_anchor - random_rolling_anchor unseen loss",
                "random_rolling_anchor - frozen_qspan_rolling_anchor unseen loss",
            ],
        ),
        resource_account=dict(
            common_initial_signature_scan=True,
            common_per_attempt_ordinary_proposal=True,
            common_per_attempt_pool_current_scan=True,
            common_per_attempt_signature_probe_count=signature_probes,
            rolling_prewrite_anchor_for_all=True,
            note=("Frozen and random methods pay for the same fresh response geometry they ignore. "
                  "Only basis identity selection differs across methods."),
        ),
        checks=checks,
        **{"pass": all(checks.values())},
        specimens=specimens,
        paired=paired_summary(specimens, checkpoints),
        limitations=[
            "measured signatures remain a black-box proxy rather than exact CausalHorizon Q",
            "random rolling basis tests rotation/coverage only within the same stratified 60-cue pool",
            "rolling equality tolerances permit tiny local drift even though actual finite responses are validated",
            "single small deterministic classifier family with three independent resampled splits",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = gate14()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
