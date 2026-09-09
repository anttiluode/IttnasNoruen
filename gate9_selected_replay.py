"""Measured replay selection, matched learning progress, and unseen cue access.

The learner receives a 60-record pool and scalar queries. Test-set measurements
are evaluator-only. No supplied Jacobian, hidden test labels or growth are used.
See GATE9_PROTOCOL.md for fixed choices and prior work.
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


METHODS = ("frozen", "unprotected", "fixed_mixed", "random_scanned",
           "boundary_scanned", "interference_scanned", "all_pool")
CHECKPOINTS = (.05,.10,.20,.40)
CHECKPOINT_TOLERANCE = 1e-6


def extra_view(images, kind):
    result = np.asarray(images,dtype=float).copy().reshape(-1,8,8)
    if kind == "upper3":
        result[:,3:,:] = 0.
    elif kind == "lower3":
        result[:,:5,:] = 0.
    elif kind == "left4":
        result[:,:,4:] = 0.
    elif kind == "right4":
        result[:,:,:4] = 0.
    else:
        raise ValueError("unknown held-out mask")
    return result.reshape(-1,64)


def make_pool(world, images, labels, indices):
    """Streaming shared storage: two fragile correct cues per class and view."""
    slots = {(c,v):[] for c in range(10) for v in ("full","upper","lower")}
    calls = 0
    for index in indices:
        label = int(labels[index])
        for kind in ("full","upper","lower"):
            cue = view(images[index:index+1],kind)
            margin = float(margins(world.probabilities(np.zeros(24),cue),[label])[0])
            calls += 1
            if margin <= .002:
                continue
            record = dict(image=cue[0].copy(), label=label, view=kind,
                          source_index=int(index), initial_margin=margin,
                          bound=min(.1,.5*margin))
            keep = slots[label,kind]
            keep.append(record)
            keep.sort(key=lambda x:x["initial_margin"])
            del keep[2:]
    if any(len(v)!=2 for v in slots.values()):
        raise RuntimeError("calibration stream did not supply all 60 intended records")
    return [item for records in slots.values() for item in records],calls


def select_indices(method, pool, rng, proposed=None):
    if method == "all_pool":
        return list(range(len(pool)))
    selected = []
    for label in range(10):
        candidates = [i for i,r in enumerate(pool) if r["label"]==label]
        if method in ("frozen","unprotected","fixed_mixed"):
            selected.extend(next(i for i in candidates if pool[i]["view"]==kind)
                            for kind in ("upper","lower"))
        elif method == "random_scanned":
            selected.extend(int(i) for i in rng.choice(candidates,2,replace=False))
        elif method == "boundary_scanned":
            selected.extend(sorted(candidates,key=lambda i:pool[i]["initial_margin"])[:2])
        elif method == "interference_scanned":
            if proposed is None:
                raise ValueError("selection requires measured proposed responses")
            def risk(i):
                r = pool[i]
                return (r["bound"]-proposed[i])/max(r["initial_margin"]-r["bound"],.01)
            selected.extend(sorted(candidates,key=risk,reverse=True)[:2])
        else:
            raise ValueError("unknown selection method")
    return selected


def reached(value, target):
    return abs(float(value-target)) <= CHECKPOINT_TOLERANCE


def build_evaluator(world, images, labels):
    # Only this closure reads test labels. Its results never return to the selector.
    seen = {k:view(images,k) for k in ("full","upper","lower")}
    unseen = {k:extra_view(images,k) for k in ("upper3","lower3","left4","right4")}
    old_correct = {k:world.probabilities(np.zeros(24),x).argmax(1)==labels
                   for k,x in {**seen,**unseen}.items()}
    def evaluate(theta):
        current = {k:world.probabilities(theta,x).argmax(1)==labels
                   for k,x in {**seen,**unseen}.items()}
        result = {}
        for family,views in (("seen",seen),("unseen",unseen)):
            baseline = np.concatenate([old_correct[k] for k in views])
            now = np.concatenate([current[k] for k in views])
            lost = baseline & ~now
            result[family] = dict(baseline_accuracy=float(baseline.mean()),
                accuracy=float(now.mean()), baseline_correct=int(baseline.sum()),
                presentations=int(baseline.size), losses=int(lost.sum()),
                loss_rate_among_initially_correct=float(lost.sum()/max(1,baseline.sum())))
        full_stays = old_correct["full"] & current["full"]
        result["full_accuracy"] = float(current["full"].mean())
        result["partial_losses_with_full_still_correct"] = int(sum(
            np.sum(full_stays & old_correct[k] & ~current[k]) for k in ("upper","lower")))
        result["new_contrast_accuracy"] = float(np.mean(
            world.probabilities(theta,.55*images+.22).argmax(1)==labels))
        return result
    return evaluate


def run_method(world, pool, new_images, new_labels, evaluate, method, seed,
               *, checkpoints=CHECKPOINTS, attempts=24, budget=45000, model_corrections=2):
    encoded_pool = [world.encode(r["image"][None,:]) for r in pool]
    encoded_new = world.encode(new_images)
    example_evaluations = 0
    def response(theta, query):
        nonlocal example_evaluations
        if query == "new":
            example_evaluations += len(new_labels)
            probabilities = world.encoded_probabilities(theta,encoded_new)
            return float(np.mean(np.log(probabilities[np.arange(len(new_labels)),new_labels]+1e-12)))
        example_evaluations += 1
        return float(margins(world.encoded_probabilities(theta,encoded_pool[query]),
                             [pool[query]["label"]])[0])
    theta = np.zeros(24)
    # This common calibration query is disclosed separately from the online cap.
    initial = response(theta,"new")
    example_evaluations = 0
    meter = ResponseMeter(response,budget)
    baseline = evaluate(theta)
    records,trace = [],[]
    rng = np.random.default_rng(seed+1901)
    targets = [(float(g),min(-.001,initial+g)) for g in checkpoints]
    index = 0
    objective = initial
    max_violation = 0.
    selected_ever = set()
    scan_calls = 0
    stop = "frozen" if method=="frozen" else "attempt_limit"
    for attempt in range(attempts if method!="frozen" else 0):
        if index >= len(targets):
            stop = "all_checkpoints_reached"
            break
        requested,target = targets[index]
        start = meter.calls
        try:
            proposed_margins = None
            scanning = method.endswith("_scanned")
            scan_start = meter.calls
            current_violations = proposed_violations = None
            if scanning:
                ordinary = BehavioralUpdateGuard(1,trust_radius=.5)
                proposal = ordinary.step(theta,"new",target,meter,protect=False,validate=False)
                if proposal.status not in ("accepted","target_met"):
                    scan_calls += meter.calls-scan_start
                    stop = "scan_"+proposal.status
                    break
                current_margins = np.array([meter(theta,i) for i in range(len(pool))])
                proposed_margins = np.array([meter(proposal.parameters,i) for i in range(len(pool))])
                bounds = np.array([r["bound"] for r in pool])
                current_violations = int(np.sum(current_margins < bounds-1e-12))
                proposed_violations = int(np.sum(proposed_margins < bounds-1e-12))
                scan_calls += meter.calls-scan_start
            selected = select_indices(method,pool,rng,proposed_margins)
            selected_ever.update(selected)
            guard = BehavioralUpdateGuard(len(selected),trust_radius=.5,
                                           max_model_corrections=model_corrections)
            for i in selected:
                guard.remember_range(i,minimum=pool[i]["bound"])
            decision = guard.step(theta,"new",target,meter,
                                  protect=method!="unprotected",validate=method!="unprotected")
            theta = decision.parameters
            if decision.max_constraint_violation is not None:
                max_violation = max(max_violation,decision.max_constraint_violation)
            if decision.target_error_after is not None:
                objective = target-decision.target_error_after
            trace.append(dict(attempt=attempt,status=decision.status,requested=requested,
                              calls=meter.calls-start,total_calls=meter.calls,selected=selected,
                              current_pool_violations=current_violations,
                              proposed_pool_violations=proposed_violations))
            if reached(objective,target):
                records.append(dict(requested_progress=requested,actual_progress=objective-initial,
                    objective=objective,target=target,calls=meter.calls,
                    example_evaluations=example_evaluations,access=evaluate(theta),
                    parameters=theta.tolist()))
                index += 1
                if index == len(targets):
                    stop = "all_checkpoints_reached"
                    break
            elif decision.status != "accepted":
                stop = decision.status
                if method!="random_scanned" or stop=="measurement_budget_exhausted":
                    break
        except MeasurementBudgetExceeded:
            if scanning:
                scan_calls += meter.calls-scan_start
            stop = "measurement_budget_exhausted_during_scan"
            break
    # Additional diagnostics below are experimenter-only, excluded from learning.
    final_margins = np.array([float(margins(world.encoded_probabilities(theta,x),[r["label"]])[0])
                              for x,r in zip(encoded_pool,pool)])
    initial_margins = np.array([r["initial_margin"] for r in pool])
    bounds = np.array([r["bound"] for r in pool])
    return dict(method=method,seed=seed,initial_objective=initial,final_objective=objective,
        actual_progress=objective-initial,stop_reason=stop,checkpoints=records,trajectory=trace,
        baseline=baseline,final=evaluate(theta),calls=meter.calls,budget=budget,
        model_example_evaluations=example_evaluations,selection_scan_calls=scan_calls,
        pool_capacity=60,active_capacity=60 if method=="all_pool" else 20,
        unique_selected=len(selected_ever),max_accepted_selected_violation=float(max_violation),
        final_pool_bound_violations=int(np.sum(final_margins<bounds-1e-12)),
        final_pool_recognition_losses=int(np.sum((initial_margins>0)&(final_margins<=0))),
        parameter_norm=float(np.linalg.norm(theta)),final_parameters=theta.tolist())


def run_specimen(seed, *, checkpoints=CHECKPOINTS, attempts=24, budget=45000, model_corrections=2):
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    images,labels = load_digits(return_X_y=True)
    images = images/16.
    pool_ids,test = train_test_split(np.arange(len(labels)),test_size=.2,
                                    stratify=labels,random_state=seed)
    train,calibration = train_test_split(pool_ids,test_size=.25,
                                       stratify=labels[pool_ids],random_state=seed+1)
    old_cal,new_cal = train_test_split(calibration,test_size=.5,
                                     stratify=labels[calibration],random_state=seed+2)
    new_cal = new_cal[:64]
    assert not (set(test)&(set(train)|set(old_cal)|set(new_cal)))
    with warnings.catch_warnings(record=True) as notes,threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model = MLPClassifier(hidden_layer_sizes=(24,),activation="tanh",solver="lbfgs",
                              max_iter=200,random_state=seed).fit(
            np.concatenate([view(images[train],v) for v in ("full","upper","lower")]),
            np.tile(labels[train],3))
        world = CueRecognizer(model,seed)
        pool,calibration_calls = make_pool(world,images,labels,old_cal)
        evaluate = build_evaluator(world,images[test],labels[test])
        runs = [run_method(world,pool,.55*images[new_cal]+.22,labels[new_cal],evaluate,
                           method,seed,checkpoints=checkpoints,attempts=attempts,budget=budget,
                           model_corrections=model_corrections)
                for method in METHODS]
    return dict(seed=seed,runs=runs,versions=dict(numpy=np.__version__,sklearn=sklearn.__version__),
        training_warnings=[str(w.message) for w in notes],training_iterations=int(model.n_iter_),
        split_sizes={name:len(ids) for name,ids in (("train",train),("old_cal",old_cal),("new_cal",new_cal),("test",test))},
        split_sha256={name:hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
                      for name,ids in (("train",train),("old_cal",old_cal),("new_cal",new_cal),("test",test))},
        baseline_calibration_calls=calibration_calls,initial_objective_calibration_examples=64,
        stored_pool=[{k:v for k,v in r.items() if k!="image"} for r in pool])


def paired_summary(specimens, checkpoints):
    """Pair only equal reached checkpoints. Missing progress remains missing."""
    output = []
    for requested in checkpoints:
        for comparator in ("unprotected","fixed_mixed","random_scanned","boundary_scanned","all_pool"):
            pairs = []
            for s in specimens:
                runs = {r["method"]:r for r in s["runs"]}
                a = next((p for p in runs["interference_scanned"]["checkpoints"]
                          if p["requested_progress"]==requested),None)
                b = next((p for p in runs[comparator]["checkpoints"]
                          if p["requested_progress"]==requested),None)
                if a is not None and b is not None:
                    pairs.append((s["seed"],a,b))
            row = dict(requested_progress=requested,comparator=comparator,
                       paired_seeds=[p[0] for p in pairs],coverage=len(pairs),total_seeds=len(specimens))
            for family in ("seen","unseen"):
                row[family+"_loss_difference"] = (float(np.mean([
                    a["access"][family]["loss_rate_among_initially_correct"]-
                    b["access"][family]["loss_rate_among_initially_correct"] for _,a,b in pairs]))
                    if pairs else None)
            row["call_difference"] = float(np.mean([a["calls"]-b["calls"] for _,a,b in pairs])) if pairs else None
            output.append(row)
    return output


def gate9(seeds=(151,163,179), *, checkpoints=CHECKPOINTS, attempts=24, budget=45000, model_corrections=2):
    specimens = [run_specimen(seed,checkpoints=checkpoints,attempts=attempts,budget=budget,
                             model_corrections=model_corrections) for seed in seeds]
    runs = [r for s in specimens for r in s["runs"]]
    checks = dict(
        callback_budgets=all(r["calls"]<=budget for r in runs),
        bounded_pool=all(len(s["stored_pool"])==60 for s in specimens),
        selected_contracts=all(r["max_accepted_selected_violation"]<=1e-12 for r in runs if r["method"]!="unprotected"),
        measured_progress=all(reached(p["objective"],p["target"]) for r in runs for p in r["checkpoints"]),
        active_capacity=all(len(t["selected"])==r["active_capacity"] for r in runs for t in r["trajectory"]))
    return dict(gate=9,status="measured_replay_selection_at_matched_progress",
        protocol=dict(seeds=list(seeds),checkpoints=list(checkpoints),attempts=attempts,budget=budget,
                      stored_records=60,ordinary_active_references=20,adjustable_parameters=24,
                      heldout_masks=["upper3","lower3","left4","right4"],progress_tolerance=CHECKPOINT_TOLERANCE,
                      measured_model_corrections=model_corrections),
        resource_account=dict(base_model_values=1810,stored_cue_values=60*64,stored_labels=60,
            stored_margin_and_bound_values=120,old_encoding_cache=60*24,
            new_task_input_values=64*64,new_task_labels=64,new_encoding_cache=64*24,
            ordinary_fd_rows=21*24,full_pool_fd_rows=61*24,
            note="Stored query IDs, temporary vectors and numerical projection workspace are additional. Full-pool guard has 62*24 Dykstra correction values; 20-reference guard has 22*24. Model evaluation and labeled calibration are explicit software capabilities."),
        checks=checks,**{"pass":all(checks.values())},specimens=specimens,
        paired=paired_summary(specimens,checkpoints),
        limitations=["single dataset with resampled splits; results are descriptive",
                     "stored candidate pool is biased toward fragile labeled cues by a supplied rule",
                     "only selected cues are constrained on each commit; unselected pool cues can fail",
                     "unseen masks are a distribution shift, not a guaranteed natural cue family",
                     "selection is related to MIR; no novel general algorithm is claimed",
                     "digital temporary model evaluation, supplied cue generators and clean training remain granted",
                     "this does not reconstruct missing cues, learn recall-time search, or test low-rank operator overlap"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,nargs="+",default=[151,163,179])
    parser.add_argument("--checkpoints",type=float,nargs="+",default=list(CHECKPOINTS))
    parser.add_argument("--attempts",type=int,default=24)
    parser.add_argument("--budget",type=int,default=45000)
    parser.add_argument("--model-corrections",type=int,default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.attempts<1 or args.budget<1 or args.model_corrections<0 or not args.checkpoints or any(x<=0 for x in args.checkpoints):
        parser.error("positive budget, attempts and checkpoints required")
    if args.checkpoints != sorted(set(args.checkpoints)):
        parser.error("checkpoints must be increasing and unique")
    result = gate9(args.seeds,checkpoints=args.checkpoints,attempts=args.attempts,budget=args.budget,
                   model_corrections=args.model_corrections)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"checks":result["checks"],"paired":result["paired"]},indent=2))
    if not result["pass"]:
        raise SystemExit(1)
