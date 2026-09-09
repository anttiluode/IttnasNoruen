"""Gate 10: approximate CausalHorizon's Q geometry with measured cue signatures.

The classifier does not expose the event-entry map Q from CausalHorizon. This gate
therefore builds a black-box proxy: for every stored cue, measure its finite response
under the same deterministic panel of temporary parameter probes. The resulting
signature vector is used only for replay-bank selection.

The new selector asks for a well-conditioned spanning bank, not the cues most damaged
by the current proposed update. Random, boundary, interference and Q-span selectors
all pay for the same scan. Held-out test labels remain evaluator-only.
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

METHODS = (
    "frozen",
    "unprotected",
    "random_scanned",
    "boundary_scanned",
    "interference_scanned",
    "qspan_scanned",
    "all_pool",
)

DEFAULT_SIGNATURE_PROBES = 12
DEFAULT_SIGNATURE_EPS = 0.03


def probe_directions(seed: int, width: int, count: int) -> np.ndarray:
    """Deterministic orthonormal temporary-update directions."""
    if count < 1 or count > width:
        raise ValueError("signature probe count must be in [1,width]")
    rng = np.random.default_rng(seed + 10091)
    raw = rng.normal(size=(width, count))
    q, _ = np.linalg.qr(raw)
    return q[:, :count].T


def measured_signatures(theta, pool, meter, directions, eps):
    """Return current margins and finite temporary-update response signatures.

    s_j[k] = margin_j(theta + eps*d_k) - margin_j(theta)

    No derivative is supplied and no temporary parameter vector is committed.
    Every scalar response is charged to the method's measurement budget.
    """
    current = np.array([meter(theta, i) for i in range(len(pool))], dtype=float)
    signatures = np.empty((len(pool), len(directions)), dtype=float)
    for k, direction in enumerate(directions):
        trial = theta + eps * direction
        signatures[:, k] = np.array(
            [meter(trial, i) for i in range(len(pool))], dtype=float
        ) - current
    return current, signatures


def signature_geometry(signatures, selected):
    """Conditioning/coverage diagnostics in the measured signature space."""
    full = np.asarray(signatures, dtype=float)
    chosen = full[np.asarray(selected, dtype=int)]
    if chosen.size == 0:
        return dict(rank=0, ambient_rank=int(np.linalg.matrix_rank(full)), min_singular=0.0,
                    condition=None, residual_fraction=1.0, logdet=None)
    singular = np.linalg.svd(chosen, compute_uv=False)
    tol = max(1e-12, singular[0] * 1e-9) if singular.size else 1e-12
    rank = int(np.sum(singular > tol))
    ambient_rank = int(np.linalg.matrix_rank(full, tol=tol))
    if rank:
        _, _, vh = np.linalg.svd(chosen, full_matrices=False)
        basis = vh[:rank].T
        projector = basis @ basis.T
        residual = full @ (np.eye(full.shape[1]) - projector)
        residual_fraction = float(np.linalg.norm(residual) / max(np.linalg.norm(full), 1e-15))
    else:
        residual_fraction = 1.0
    min_singular = float(singular[-1]) if singular.size else 0.0
    condition = (float(singular[0] / singular[-1])
                 if singular.size and singular[-1] > tol and rank == full.shape[1]
                 else None)
    gram = chosen.T @ chosen
    ridge = max(1e-15, 1e-6 * float(np.trace(full.T @ full)) / max(1, full.shape[1]))
    sign, logdet = np.linalg.slogdet(gram + ridge * np.eye(gram.shape[0]))
    return dict(rank=rank, ambient_rank=ambient_rank, min_singular=min_singular,
                condition=condition, residual_fraction=residual_fraction,
                logdet=float(logdet) if sign > 0 else None, ridge=ridge)


def _qspan_indices(pool, signatures, capacity=20):
    """Greedy D-optimal bank with the Gate-9 two-per-class contract.

    The objective maximizes log det(ridge I + sum s_i s_i^T). Since capacity=20
    and each of ten labels has a cap of two, the final bank has exactly two cues per
    class while still choosing them from the global measured geometry.
    """
    signatures = np.asarray(signatures, dtype=float)
    if capacity != 20:
        raise ValueError("Gate 10 currently fixes the active bank at 20")
    k = signatures.shape[1]
    scale = float(np.trace(signatures.T @ signatures)) / max(1, k)
    ridge = max(1e-15, 1e-6 * scale)
    gram = ridge * np.eye(k)
    selected = []
    per_label = {label: 0 for label in range(10)}
    remaining = set(range(len(pool)))
    while len(selected) < capacity:
        best = None
        best_score = -np.inf
        for i in sorted(remaining):
            label = int(pool[i]["label"])
            if per_label[label] >= 2:
                continue
            v = signatures[i]
            candidate = gram + np.outer(v, v)
            sign, score = np.linalg.slogdet(candidate)
            if sign > 0 and (score > best_score + 1e-14 or
                             (abs(score - best_score) <= 1e-14 and (best is None or i < best))):
                best = i
                best_score = float(score)
        if best is None:
            raise RuntimeError("could not fill stratified Q-span bank")
        selected.append(best)
        remaining.remove(best)
        label = int(pool[best]["label"])
        per_label[label] += 1
        v = signatures[best]
        gram += np.outer(v, v)
    return selected


def select_indices(method, pool, rng, signatures=None, proposed=None):
    if method == "all_pool":
        return list(range(len(pool)))
    if method == "qspan_scanned":
        if signatures is None:
            raise ValueError("Q-span selection requires measured signatures")
        return _qspan_indices(pool, signatures)
    selected = []
    for label in range(10):
        candidates = [i for i, r in enumerate(pool) if int(r["label"]) == label]
        if method in ("frozen", "unprotected"):
            selected.extend(next(i for i in candidates if pool[i]["view"] == kind)
                            for kind in ("upper", "lower"))
        elif method == "random_scanned":
            selected.extend(int(i) for i in rng.choice(candidates, 2, replace=False))
        elif method == "boundary_scanned":
            selected.extend(sorted(candidates, key=lambda i: pool[i]["initial_margin"])[:2])
        elif method == "interference_scanned":
            if proposed is None:
                raise ValueError("interference selection requires proposed margins")
            def risk(i):
                r = pool[i]
                return (r["bound"] - proposed[i]) / max(r["initial_margin"] - r["bound"], .01)
            selected.extend(sorted(candidates, key=risk, reverse=True)[:2])
        else:
            raise ValueError("unknown method")
    return selected


def run_method(world, pool, new_images, new_labels, evaluate, method, seed, *,
               checkpoints=CHECKPOINTS, attempts=24, budget=45000,
               model_corrections=2, signature_probes=DEFAULT_SIGNATURE_PROBES,
               signature_eps=DEFAULT_SIGNATURE_EPS):
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
        example_evaluations += 1
        return float(margins(
            world.encoded_probabilities(theta, encoded_pool[query]),
            [pool[query]["label"]],
        )[0])

    theta = np.zeros(24)
    initial = response(theta, "new")
    example_evaluations = 0
    meter = ResponseMeter(response, budget)
    baseline = evaluate(theta)
    rng = np.random.default_rng(seed + 1901)
    directions = probe_directions(seed, theta.size, signature_probes)
    targets = [(float(g), min(-.001, initial + g)) for g in checkpoints]
    index = 0
    objective = initial
    max_violation = 0.0
    selected_ever = set()
    scan_calls = 0
    trace = []
    records = []
    stop = "frozen" if method == "frozen" else "attempt_limit"

    for attempt in range(attempts if method != "frozen" else 0):
        if index >= len(targets):
            stop = "all_checkpoints_reached"
            break
        requested, target = targets[index]
        start = meter.calls
        scanning = method.endswith("_scanned")
        scan_start = meter.calls
        signatures = None
        proposed_margins = None
        geometry = None
        current_violations = proposed_violations = None
        try:
            if scanning:
                # All scanned methods pay for the same ordinary proposal, pool
                # measurements, and signature panel. Only the ranking rule differs.
                ordinary = BehavioralUpdateGuard(1, trust_radius=.5)
                proposal = ordinary.step(theta, "new", target, meter,
                                         protect=False, validate=False)
                if proposal.status not in ("accepted", "target_met"):
                    scan_calls += meter.calls - scan_start
                    stop = "scan_" + proposal.status
                    break
                current_margins, signatures = measured_signatures(
                    theta, pool, meter, directions, signature_eps)
                proposed_margins = np.array(
                    [meter(proposal.parameters, i) for i in range(len(pool))], dtype=float)
                bounds = np.array([r["bound"] for r in pool])
                current_violations = int(np.sum(current_margins < bounds - 1e-12))
                proposed_violations = int(np.sum(proposed_margins < bounds - 1e-12))
                scan_calls += meter.calls - scan_start

            selected = select_indices(method, pool, rng, signatures, proposed_margins)
            selected_ever.update(selected)
            if signatures is not None:
                geometry = signature_geometry(signatures, selected)

            guard = BehavioralUpdateGuard(
                len(selected), trust_radius=.5,
                max_model_corrections=model_corrections)
            for i in selected:
                guard.remember_range(i, minimum=pool[i]["bound"])
            decision = guard.step(
                theta, "new", target, meter,
                protect=method != "unprotected",
                validate=method != "unprotected",
            )
            theta = decision.parameters
            if decision.max_constraint_violation is not None:
                max_violation = max(max_violation, decision.max_constraint_violation)
            if decision.target_error_after is not None:
                objective = target - decision.target_error_after
            trace.append(dict(
                attempt=attempt,
                status=decision.status,
                requested=requested,
                calls=meter.calls - start,
                total_calls=meter.calls,
                scan_calls=meter.calls - scan_start if scanning else 0,
                selected=selected,
                signature_geometry=geometry,
                current_pool_violations=current_violations,
                proposed_pool_violations=proposed_violations,
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
                    selection_geometry=geometry,
                    parameters=theta.tolist(),
                ))
                index += 1
                if index == len(targets):
                    stop = "all_checkpoints_reached"
                    break
            elif decision.status != "accepted":
                stop = decision.status
                if method != "random_scanned" or stop == "measurement_budget_exhausted":
                    break
        except MeasurementBudgetExceeded:
            if scanning:
                scan_calls += meter.calls - scan_start
            stop = "measurement_budget_exhausted_during_scan"
            break

    final_margins = np.array([
        float(margins(world.encoded_probabilities(theta, x), [r["label"]])[0])
        for x, r in zip(encoded_pool, pool)
    ])
    initial_margins = np.array([r["initial_margin"] for r in pool])
    bounds = np.array([r["bound"] for r in pool])
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
        selection_scan_calls=scan_calls,
        signature_probes=signature_probes,
        signature_eps=signature_eps,
        pool_capacity=60,
        active_capacity=60 if method == "all_pool" else 20,
        unique_selected=len(selected_ever),
        max_accepted_selected_violation=float(max_violation),
        final_pool_bound_violations=int(np.sum(final_margins < bounds - 1e-12)),
        final_pool_recognition_losses=int(np.sum((initial_margins > 0) & (final_margins <= 0))),
        parameter_norm=float(np.linalg.norm(theta)),
        final_parameters=theta.tolist(),
    )


def run_specimen(seed, *, checkpoints=CHECKPOINTS, attempts=24, budget=45000,
                 model_corrections=2, signature_probes=DEFAULT_SIGNATURE_PROBES,
                 signature_eps=DEFAULT_SIGNATURE_EPS):
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
        runs = [
            run_method(
                world, pool, .55 * images[new_cal] + .22, labels[new_cal], evaluate,
                method, seed, checkpoints=checkpoints, attempts=attempts, budget=budget,
                model_corrections=model_corrections, signature_probes=signature_probes,
                signature_eps=signature_eps,
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
        for comparator in (
            "random_scanned", "boundary_scanned", "interference_scanned",
            "all_pool", "unprotected",
        ):
            pairs = []
            for specimen in specimens:
                runs = {r["method"]: r for r in specimen["runs"]}
                a = next((p for p in runs["qspan_scanned"]["checkpoints"]
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


def gate10(seeds=(211, 223, 239), *, checkpoints=CHECKPOINTS, attempts=24,
           budget=45000, model_corrections=2,
           signature_probes=DEFAULT_SIGNATURE_PROBES,
           signature_eps=DEFAULT_SIGNATURE_EPS):
    specimens = [
        run_specimen(
            seed, checkpoints=checkpoints, attempts=attempts, budget=budget,
            model_corrections=model_corrections, signature_probes=signature_probes,
            signature_eps=signature_eps,
        )
        for seed in seeds
    ]
    runs = [r for s in specimens for r in s["runs"]]
    scanned = [r for r in runs if r["method"].endswith("_scanned")]
    checks = dict(
        callback_budgets=all(r["calls"] <= budget for r in runs),
        bounded_pool=all(len(s["stored_pool"]) == 60 for s in specimens),
        selected_contracts=all(
            r["max_accepted_selected_violation"] <= 1e-12
            for r in runs if r["method"] != "unprotected"),
        measured_progress=all(
            reached(p["objective"], p["target"])
            for r in runs for p in r["checkpoints"]),
        active_capacity=all(
            len(t["selected"]) == r["active_capacity"]
            for r in runs for t in r["trajectory"]),
        shared_signature_protocol=all(
            r["signature_probes"] == signature_probes and
            abs(r["signature_eps"] - signature_eps) < 1e-15
            for r in scanned),
        qspan_stratified=all(
            all(sum(1 for i in t["selected"] if int(s["stored_pool"][i]["label"]) == label) == 2
                for label in range(10))
            for s in specimens
            for r in s["runs"] if r["method"] == "qspan_scanned"
            for t in r["trajectory"]),
    )
    return dict(
        gate=10,
        status="measured_event_entry_proxy_coverage",
        protocol=dict(
            seeds=list(seeds), checkpoints=list(checkpoints), attempts=attempts,
            budget=budget, stored_records=60, active_references=20,
            adjustable_parameters=24, signature_probes=signature_probes,
            signature_eps=signature_eps,
            signature_definition="margin(theta+eps*d_k)-margin(theta)",
            selection_objective="stratified greedy logdet of measured signature Gram matrix",
            progress_tolerance=CHECKPOINT_TOLERANCE,
            measured_model_corrections=model_corrections,
        ),
        resource_account=dict(
            per_signature_scan_pool_current=60,
            per_signature_scan_probe_calls=60 * signature_probes,
            per_signature_scan_proposed_pool=60,
            note=("All *_scanned methods pay the same signature panel and proposed-pool scan. "
                  "Temporary new-task proposal calls are also charged. Test-set access evaluation "
                  "is evaluator-only and never enters selection."),
        ),
        checks=checks,
        **{"pass": all(checks.values())},
        specimens=specimens,
        paired=paired_summary(specimens, checkpoints),
        limitations=[
            "the measured signatures are a black-box proxy for Q, not the exact CausalHorizon event-entry map",
            "finite temporary probes mix entry geometry with the classifier's current downstream response",
            "the selector is D-optimal in the supplied probe panel and may overfit that panel",
            "only selected cues are constrained on each commit; unselected pool cues can fail",
            "single small dataset with resampled splits; matched-progress outcomes are descriptive",
            "digital temporary parameter evaluation and externally supplied cue labels remain granted",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[211, 223, 239])
    parser.add_argument("--checkpoints", type=float, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--attempts", type=int, default=24)
    parser.add_argument("--budget", type=int, default=45000)
    parser.add_argument("--model-corrections", type=int, default=2)
    parser.add_argument("--signature-probes", type=int, default=DEFAULT_SIGNATURE_PROBES)
    parser.add_argument("--signature-eps", type=float, default=DEFAULT_SIGNATURE_EPS)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (args.attempts < 1 or args.budget < 1 or args.model_corrections < 0 or
            args.signature_probes < 1 or args.signature_probes > 24 or
            args.signature_eps <= 0 or not args.checkpoints or
            any(x <= 0 for x in args.checkpoints)):
        parser.error("positive budget/attempts/checkpoints/epsilon and 1..24 probes required")
    if args.checkpoints != sorted(set(args.checkpoints)):
        parser.error("checkpoints must be increasing and unique")
    result = gate10(
        args.seeds, checkpoints=args.checkpoints, attempts=args.attempts,
        budget=args.budget, model_corrections=args.model_corrections,
        signature_probes=args.signature_probes, signature_eps=args.signature_eps,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": result["checks"], "paired": result["paired"]}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)
