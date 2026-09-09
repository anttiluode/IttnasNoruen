"""Gate 13: rebuild a Q-like cue basis at each proposed write.

Gate 12 froze a well-conditioned basis at theta_0 and preserved its scalar responses
almost exactly, yet unseen access still drifted. Gate 13 asks whether the useful entry
geometry must move with the current state and, more specifically, with the proposed
write itself.

This remains a black-box software experiment. The measured signatures are not asserted
to equal CausalHorizon's exact Q.
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

METHODS = (
    "static_scanned_equal",
    "rolling_current_equal",
    "rolling_write_equal",
)

DEFAULT_EQUAL_REL_TOL = 0.05
DEFAULT_EQUAL_ABS_TOL = 0.002


def generic_directions(seed: int, attempt: int, width: int, count: int) -> np.ndarray:
    """Deterministic fresh orthonormal panel for one atomic write."""
    return probe_directions(seed + 4001 * (attempt + 1), width, count)


def conditioned_directions(
    proposed_step: np.ndarray,
    seed: int,
    attempt: int,
    count: int,
) -> np.ndarray:
    """Orthonormal panel whose first axis is the actual proposed-write direction."""
    step = np.asarray(proposed_step, dtype=float).reshape(-1)
    width = step.size
    if count < 1 or count > width:
        raise ValueError("direction count must be in [1,width]")
    norm = float(np.linalg.norm(step))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("proposed write must have a finite nonzero direction")
    u = step / norm
    if count == 1:
        return u[None, :]

    rng = np.random.default_rng(seed + 60013 + 7919 * attempt)
    raw = rng.normal(size=(width, count - 1))
    raw -= np.outer(u, u @ raw)
    q, _ = np.linalg.qr(raw)
    q = q[:, : count - 1]
    # QR can only fail here for a numerically rank-deficient random draw.
    if q.shape[1] != count - 1 or np.max(np.abs(u @ q), initial=0.0) > 1e-10:
        raise RuntimeError("failed to construct write-conditioned orthogonal panel")
    return np.vstack([u, q.T])


def _two_per_class(pool, selected) -> bool:
    return (
        len(selected) == 20
        and len(set(int(i) for i in selected)) == 20
        and all(sum(int(pool[i]["label"]) == label for i in selected) == 2 for label in range(10))
    )


def _jaccard(a, b) -> float:
    a, b = set(int(i) for i in a), set(int(i) for i in b)
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


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
        raise ValueError("unknown Gate-13 method")

    encoded_pool = [world.encode(r["image"][None, :]) for r in pool]
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

    # Every method pays this same initial acquisition. Only static_scanned_equal uses
    # it later as a permanently frozen basis/anchor contract.
    initial_panel = probe_directions(seed, theta.size, signature_probes)
    try:
        initial_margins, initial_signatures = measured_signatures(
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
    frozen_values = initial_margins[np.asarray(frozen_selected, dtype=int)]
    frozen_tolerances = np.array([
        response_tolerance(v, equal_rel_tol, equal_abs_tol) for v in frozen_values
    ], dtype=float)
    initial_geometry = signature_geometry(initial_signatures, frozen_selected)
    initial_scan_calls = meter.calls

    targets = [(float(g), min(-0.001, initial + g)) for g in checkpoints]
    index = 0
    objective = initial
    records = []
    trace = []
    selected_history = []
    max_accepted_violation = 0.0
    max_local_ratio = 0.0
    scan_calls_total = initial_scan_calls
    stop = "attempt_limit"
    last_selected = frozen_selected
    last_audit = None

    for attempt in range(attempts):
        if index >= len(targets):
            stop = "all_checkpoints_reached"
            break
        requested, target = targets[index]
        attempt_start = meter.calls
        scan_start = meter.calls

        try:
            # First expose the ordinary intended write. It is measured, never committed.
            ordinary = BehavioralUpdateGuard(1, trust_radius=0.5)
            proposal = ordinary.step(
                theta, "new", target, meter, protect=False, validate=False
            )
            if proposal.status not in ("accepted", "target_met"):
                stop = "proposal_" + proposal.status
                break
            proposed_step = proposal.parameters - theta
            proposal_norm = float(np.linalg.norm(proposed_step))
            if proposal_norm < 1e-12:
                stop = "proposal_no_direction"
                break

            if method == "rolling_write_equal":
                panel = conditioned_directions(
                    proposed_step, seed, attempt, signature_probes
                )
                write_axis_cosine = float(
                    abs(panel[0] @ proposed_step) / max(proposal_norm, 1e-30)
                )
            else:
                panel = generic_directions(seed, attempt, theta.size, signature_probes)
                write_axis_cosine = float(
                    abs(panel[0] @ proposed_step) / max(proposal_norm, 1e-30)
                )

            current_margins, signatures = measured_signatures(
                theta, pool, meter, panel, signature_eps
            )
            scan_calls = meter.calls - scan_start
            scan_calls_total += scan_calls

            if method == "static_scanned_equal":
                selected = list(frozen_selected)
                anchor_values = np.array(frozen_values, copy=True)
                tolerances = np.array(frozen_tolerances, copy=True)
            else:
                selected = _qspan_indices(pool, signatures, capacity=20)
                anchor_values = current_margins[np.asarray(selected, dtype=int)]
                tolerances = np.array([
                    response_tolerance(v, equal_rel_tol, equal_abs_tol)
                    for v in anchor_values
                ], dtype=float)

            if not _two_per_class(pool, selected):
                raise RuntimeError("Gate-13 basis violated the two-per-class contract")

            geometry = signature_geometry(signatures, selected)
            previous = selected_history[-1] if selected_history else frozen_selected
            overlap_previous = _jaccard(previous, selected)
            overlap_frozen = _jaccard(frozen_selected, selected)
            selected_history.append(list(selected))

            # Anti-cheating boundary: selected + anchor_values are fixed here, before
            # the protected candidate is generated or observed.
            guard = BehavioralUpdateGuard(
                len(selected), trust_radius=0.5, projection_mode="equalities"
            )
            for i, value, tolerance in zip(selected, anchor_values, tolerances):
                guard.remember(i, float(value), float(tolerance))

            decision = guard.step(
                theta, "new", target, meter, protect=True, validate=True
            )
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
            last_selected = selected
            last_audit = local_audit
            max_local_ratio = max(max_local_ratio, local_audit["max_tolerance_ratio"])

            selected_first_axis = float(np.mean(np.abs(signatures[selected, 0])))
            pool_first_axis = float(np.mean(np.abs(signatures[:, 0])))
            trace.append(dict(
                attempt=attempt,
                status=decision.status,
                requested=requested,
                calls=meter.calls - attempt_start,
                total_calls=meter.calls,
                scan_calls=scan_calls,
                proposal_norm=proposal_norm,
                write_axis_cosine=write_axis_cosine,
                selected=list(selected),
                selected_labels=[int(pool[i]["label"]) for i in selected],
                overlap_previous=overlap_previous,
                overlap_frozen=overlap_frozen,
                signature_geometry=geometry,
                selected_first_axis_mean_abs=selected_first_axis,
                pool_first_axis_mean_abs=pool_first_axis,
                local_basis=local_audit,
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
                    proposal_norm=proposal_norm,
                    write_axis_cosine=write_axis_cosine,
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
        except ValueError as exc:
            if "proposed write" in str(exc):
                stop = "proposal_no_direction"
                break
            raise

    final_margins = np.array([
        float(margins(
            world.encoded_probabilities(theta, encoded_pool[i]), [pool[i]["label"]]
        )[0])
        for i in range(len(pool))
    ])
    initial_pool_margins = np.array([r["initial_margin"] for r in pool])
    bounds = np.array([r["bound"] for r in pool])
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
        per_attempt_scan_calls=scan_calls_total - initial_scan_calls,
        signature_probes=signature_probes,
        signature_eps=signature_eps,
        equal_relative_tolerance=equal_rel_tol,
        equal_absolute_tolerance=equal_abs_tol,
        active_capacity=20,
        frozen_selected=list(frozen_selected),
        frozen_signature_geometry=initial_geometry,
        selected_final=list(last_selected),
        distinct_bases=distinct_bases,
        mean_overlap_previous=mean_overlap_previous,
        mean_overlap_frozen=mean_overlap_frozen,
        max_local_tolerance_ratio=float(max_local_ratio),
        last_local_basis=last_audit,
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
        stored_pool=[{k: v for k, v in r.items() if k != "image"} for r in pool],
    )


def paired_summary(specimens, checkpoints):
    rows = []
    for requested in checkpoints:
        for comparator in ("rolling_current_equal", "static_scanned_equal"):
            pairs = []
            for specimen in specimens:
                runs = {r["method"]: r for r in specimen["runs"]}
                a = next((p for p in runs["rolling_write_equal"]["checkpoints"]
                          if p["requested_progress"] == requested), None)
                b = next((p for p in runs[comparator]["checkpoints"]
                          if p["requested_progress"] == requested), None)
                if a is not None and b is not None:
                    pairs.append((specimen["seed"], a, b))
            row = dict(
                requested_progress=requested,
                comparator=comparator,
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
            rows.append(row)
    return rows


def gate13(
    seeds=(347, 359, 373),
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
    runs = [r for s in specimens for r in s["runs"]]
    rolling = [r for r in runs if r["method"] != "static_scanned_equal"]
    write_runs = [r for r in runs if r["method"] == "rolling_write_equal"]
    checks = dict(
        callback_budgets=all(r["calls"] <= budget for r in runs),
        bounded_pool=all(len(s["stored_pool"]) == 60 for s in specimens),
        measured_progress=all(
            reached(p["objective"], p["target"])
            for r in runs for p in r["checkpoints"]
        ),
        active_capacity=all(r["active_capacity"] == 20 for r in runs),
        anti_cheating_precommit=all(
            t["basis_chosen_before_guard"] for r in runs for t in r["trajectory"]
        ),
        accepted_local_contracts_validated=all(
            r["max_accepted_constraint_violation"] <= 1e-12 for r in runs
        ),
        write_axis_is_proposal=all(
            t["write_axis_cosine"] >= 1.0 - 1e-10
            for r in write_runs for t in r["trajectory"]
        ),
        rolling_bases_stratified=all(
            all(labels.count(label) == 2 for label in range(10))
            for r in rolling for t in r["trajectory"]
            for labels in [t["selected_labels"]]
        ),
    )
    return dict(
        gate=13,
        status="rolling_proposed_write_conditioned_basis_test",
        protocol=dict(
            seeds=list(seeds), checkpoints=list(checkpoints), attempts=attempts,
            budget=budget, stored_records=60, active_basis=20,
            adjustable_parameters=24, signature_probes=signature_probes,
            signature_eps=signature_eps,
            equal_relative_tolerance=equal_rel_tol,
            equal_absolute_tolerance=equal_abs_tol,
            progress_tolerance=CHECKPOINT_TOLERANCE,
            methods=list(METHODS),
            primary="rolling_write_equal - rolling_current_equal unseen loss at matched progress",
            anti_cheating="basis and anchors fixed before protected candidate is evaluated",
        ),
        resource_account=dict(
            common_initial_signature_scan=True,
            common_per_attempt_ordinary_proposal=True,
            common_per_attempt_pool_current_scan=True,
            common_per_attempt_signature_probe_count=signature_probes,
            note=("All methods pay the same acquisition skeleton. Static_scanned_equal deliberately "
                  "pays for fresh scans it ignores so acquisition cost is not its advantage."),
        ),
        checks=checks,
        **{"pass": all(checks.values())},
        specimens=specimens,
        paired=paired_summary(specimens, checkpoints),
        limitations=[
            "measured signatures are a response-sensitivity proxy, not the exact CausalHorizon Q",
            "rolling equality anchors preserve only the selected scalar cue responses for one atomic write",
            "the proposed direction comes from the same black-box local learner and may itself be a poor causal probe",
            "single small deterministic classifier family with three resampled splits",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = gate13()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
