"""Protect access through incomplete cues, and test bounded added pathways.

This is a real-data recognition/access benchmark, not autobiographical recall or
a growing biological dendrite. All controls share a trained digit recognizer,
calibration stream, labels, available cue transformations and measurement budget.
Held-out labels and route-rescue opportunities are evaluator-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from behavioral_guard import BehavioralUpdateGuard, ResponseMeter


METHODS = ("frozen", "unprotected", "exact_full", "margin_full", "margin_mixed",
           "margin_boundary", "grow_mixed", "large_mixed")
VIEWS = ("full", "upper", "lower")


def view(images, kind):
    result = np.asarray(images, dtype=float).copy().reshape(-1, 8, 8)
    if kind == "upper":
        result[:, 4:, :] = 0.
    elif kind == "lower":
        result[:, :4, :] = 0.
    elif kind != "full":
        raise ValueError("unknown cue")
    return result.reshape(-1, 64)


def softmax(logits):
    shifted = logits-logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values/values.sum(axis=1, keepdims=True)


def margins(probabilities, labels):
    competitors = probabilities.copy()
    competitors[np.arange(len(labels)), labels] = -np.inf
    return probabilities[np.arange(len(labels)), labels]-competitors.max(axis=1)


def retrieval_policy(upper, lower, threshold=.2):
    """Try another cue only on observed uncertainty. No correct label is used."""
    top = np.sort(upper, axis=1)[:, -2:]
    ask = top[:, 1]-top[:, 0] < threshold
    combined = upper.copy()
    combined[ask] = (upper[ask]+lower[ask])/2
    return combined.argmax(axis=1), ask


class CueRecognizer:
    def __init__(self, model, seed):
        self.w1, self.w2 = [a.copy() for a in model.coefs_]
        self.b1, self.b2 = [a.copy() for a in model.intercepts_]
        self.width = len(self.b1)
        # The same potential new paths are used by late-growth and initially-large
        # controls. Initial output weights are zero, so allocation alone changes no answer.
        self.path_factory_seed = seed+701
        self.extra_w = self.extra_b = None

    def allocate_paths(self):
        rng = np.random.default_rng(self.path_factory_seed)
        self.extra_w = rng.normal(0., .35, (64, 4))
        self.extra_b = rng.normal(0., .1, 4)

    def encode(self, images):
        return (images@self.w1+self.b1, images)

    def encoded_probabilities(self, theta, encoded):
        core, images = encoded
        logits = np.tanh(core+theta[:self.width])@self.w2+self.b2
        if theta.size > self.width:
            if self.extra_w is None:
                raise ValueError("new pathways must be allocated before using their weights")
            extra = np.tanh((images-.5)@self.extra_w+self.extra_b)
            logits = logits+extra@theta[self.width:].reshape(4, 10)
        return softmax(logits)

    def probabilities(self, theta, images):
        return self.encoded_probabilities(theta, self.encode(images))


def select_banks(world, images, labels, indices):
    """Three streaming selectors, each holding at most two cues per class.

    Every candidate receives the same baseline measurement. The full calibration
    pool is discarded by the learner; only selected input/label records survive.
    'boundary' means smallest positive baseline margin, not an adaptive MIR policy.
    """
    banks = {kind: {label: [] for label in range(10)}
             for kind in ("full", "mixed", "boundary")}
    zero = np.zeros(world.width)
    measured = 0
    for index in indices:
        label = int(labels[index])
        for kind in VIEWS:
            cue = view(images[index:index+1], kind)
            margin = float(margins(world.probabilities(zero, cue), [label])[0])
            measured += 1
            if margin <= .002:
                continue
            record = dict(image=cue[0].copy(), label=label, view=kind,
                          baseline_margin=margin, source_index=int(index))
            if kind == "full" and len(banks["full"][label]) < 2:
                banks["full"][label].append(record)
            if kind in ("upper", "lower") and not any(
                    r["view"] == kind for r in banks["mixed"][label]):
                banks["mixed"][label].append(record)
            selected = banks["boundary"][label]
            selected.append(record)
            selected.sort(key=lambda r: r["baseline_margin"])
            del selected[2:]
    for kind, per_class in banks.items():
        if any(len(records) != 2 for records in per_class.values()):
            raise RuntimeError(f"insufficient correctly recognized calibration cues for {kind}")
    return {kind: [r for records in bank.values() for r in records]
            for kind, bank in banks.items()}, measured


def evaluate_access(world, theta, images, labels):
    full = world.probabilities(theta, view(images, "full"))
    upper = world.probabilities(theta, view(images, "upper"))
    lower = world.probabilities(theta, view(images, "lower"))
    policy, asked = retrieval_policy(upper, lower)
    first_wrong = upper.argmax(1) != labels
    lower_correct = lower.argmax(1) == labels
    # This is a cue-channel failure with ALL material frozen during evaluation.
    # The remaining lower cue is observed data, not an invented correct completion.
    blocked = world.probabilities(theta, np.zeros_like(images))
    blocked_policy, blocked_asked = retrieval_policy(blocked, lower)
    return dict(full_accuracy=float(np.mean(full.argmax(1) == labels)),
                upper_accuracy=float(np.mean(~first_wrong)),
                lower_accuracy=float(np.mean(lower_correct)),
                two_cue_accuracy=float(np.mean(((upper+lower)/2).argmax(1) == labels)),
                policy_accuracy=float(np.mean(policy == labels)),
                policy_mean_queries=float(1+asked.mean()),
                confident_wrong_stop_fraction=float(np.mean(first_wrong & ~asked)),
                alternate_cue_rescue_opportunity=float(np.mean(first_wrong & lower_correct)),
                blocked_upper_accuracy=float(np.mean(blocked.argmax(1) == labels)),
                blocked_upper_policy_accuracy=float(np.mean(blocked_policy == labels)),
                blocked_upper_policy_mean_queries=float(1+blocked_asked.mean()))


def paired_access_changes(world, theta, images, labels):
    before = [world.probabilities(np.zeros(world.width),view(images,v)).argmax(1)==labels
              for v in VIEWS]
    after = [world.probabilities(theta,view(images,v)).argmax(1)==labels for v in VIEWS]
    still_full = before[0] & after[0]
    upper_lost = before[1] & ~after[1]
    lower_lost = before[2] & ~after[2]
    return dict(upper_route_lost_while_full_correct=float(np.mean(still_full & upper_lost)),
                lower_route_lost_while_full_correct=float(np.mean(still_full & lower_lost)),
                alternate_still_correct_after_upper_lost=float(np.mean(still_full & upper_lost & after[2])))


def tolerance_counterexample():
    def response(theta, query):
        return float(theta[query]) if isinstance(query, int) else float(theta[0]+theta[2])
    outcomes = {}
    for mode in ("equalities", "bounds"):
        guard = BehavioralUpdateGuard(4, projection_mode=mode)
        for q in range(4):
            guard.remember(q, 1., .1)
        decision = guard.step(np.ones(4), "new", 2.1, ResponseMeter(response, 100))
        outcomes[mode] = dict(status=decision.status,
                              new_answer=response(decision.parameters, "new"),
                              largest_old_drift=float(np.max(np.abs(decision.parameters-1))))
    return outcomes


def run_specimen(seed, *, steps=12, budget=45000):
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    images, labels = load_digits(return_X_y=True)
    images = images/16.
    pool, test = train_test_split(np.arange(len(labels)), test_size=.2,
                                  stratify=labels, random_state=seed)
    train, calibration = train_test_split(pool, test_size=.25,
                                         stratify=labels[pool], random_state=seed+1)
    old_cal, new_cal = train_test_split(calibration, test_size=.5,
                                      stratify=labels[calibration], random_state=seed+2)
    new_cal = new_cal[:64]
    train_x = np.concatenate([view(images[train], k) for k in VIEWS])
    train_y = np.tile(labels[train], 3)
    with warnings.catch_warnings(record=True) as training_warnings, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        model = MLPClassifier(hidden_layer_sizes=(24,), activation="tanh", solver="lbfgs",
                              max_iter=200, random_state=seed).fit(train_x, train_y)
    world = CueRecognizer(model, seed)
    banks, baseline_calls = select_banks(world, images, labels, old_cal)
    changed = .55*images[new_cal]+.22
    new_encoded = world.encode(changed)
    new_labels = labels[new_cal]
    initial_p = world.encoded_probabilities(np.zeros(24), new_encoded)
    initial_objective = float(np.log(initial_p[np.arange(len(new_labels)), new_labels]+1e-12).mean())
    target = min(-.001, initial_objective+.2)
    # Test outputs never influence bank selection, growth, hyperparameters or stopping.
    before = evaluate_access(world, np.zeros(24), images[test], labels[test])
    before_changed = float(np.mean(world.probabilities(np.zeros(24), .55*images[test]+.22).argmax(1)==labels[test]))
    runs = []
    with threadpool_limits(limits=1):
        for method in METHODS:
            world.extra_w = world.extra_b = None
            if method == "large_mixed":
                world.allocate_paths()
            kind = "full" if method in ("exact_full", "margin_full") else (
                "boundary" if method == "margin_boundary" else "mixed")
            bank = banks[kind]
            encoded = [world.encode(r["image"][None, :]) for r in bank]
            model_evaluations = 0
            def response(theta, query):
                nonlocal model_evaluations
                if query == "new":
                    model_evaluations += len(new_labels)
                    p = world.encoded_probabilities(theta, new_encoded)
                    return float(np.log(p[np.arange(len(new_labels)),new_labels]+1e-12).mean())
                model_evaluations += 1
                p = world.encoded_probabilities(theta, encoded[query])
                return float(margins(p, [bank[query]["label"]])[0])
            guard = BehavioralUpdateGuard(20, trust_radius=.5,
                                           projection_mode="equalities" if method=="exact_full" else "bounds",
                                           max_model_corrections=0)  # frozen Gate 8 protocol
            for q, record in enumerate(bank):
                if method == "exact_full":
                    guard.remember(q, record["baseline_margin"], 1e-4)
                else:
                    guard.remember_range(q, minimum=min(.1, .5*record["baseline_margin"]))
            theta = np.zeros(64 if method=="large_mixed" else 24)
            meter = ResponseMeter(response, budget)
            trajectory = []
            growth = []
            max_violation = 0.
            parameter_exposure = 0
            if method != "frozen":
                for step in range(steps):
                    parameter_exposure += theta.size
                    decision = guard.step(theta, "new", target, meter,
                                          protect=method!="unprotected", validate=method!="unprotected")
                    theta = decision.parameters
                    if decision.max_constraint_violation is not None:
                        max_violation = max(max_violation, decision.max_constraint_violation)
                    trajectory.append(dict(step=step, status=decision.status, calls=meter.calls,
                                           parameters=theta.size, objective_error=decision.target_error_after))
                    if decision.status != "accepted":
                        if (method=="grow_mixed" and theta.size==24 and step+1 < steps and
                            decision.status in ("no_acceptable_step_observed", "no_feasible_step_observed") and
                            meter.max_calls-meter.calls >= 2*64*21+21):
                            theta = np.r_[theta, np.zeros(40)]
                            world.allocate_paths()
                            growth.append(dict(step=step, reason=decision.status, added_hidden_units=4,
                                               added_trainable_values=40, added_fixed_values=260))
                            continue
                        break
            after = evaluate_access(world, theta, images[test], labels[test])
            access_changes = paired_access_changes(world,theta,images[test],labels[test])
            after_changed = float(np.mean(world.probabilities(theta, .55*images[test]+.22).argmax(1)==labels[test]))
            final_p = world.encoded_probabilities(theta, new_encoded)
            final_objective = float(np.log(final_p[np.arange(len(new_labels)),new_labels]+1e-12).mean())
            observed_bank = [float(margins(world.encoded_probabilities(theta, code), [r["label"]])[0])
                             for code,r in zip(encoded,bank)]
            final_violation = max(max(0.,r.lower-value,value-r.upper)
                                  for r,value in zip(guard.references,observed_bank))
            runs.append(dict(seed=seed, method=method, bank=kind, before=before, after=after,
                             paired_access_changes=access_changes,
                             changed_accuracy_before=before_changed, changed_accuracy_after=after_changed,
                             objective_before=initial_objective, objective_after=final_objective, target=target,
                             scalar_calls=meter.calls, model_example_evaluations=model_evaluations,
                             call_budget=budget, baseline_calibration_calls=baseline_calls,
                             final_bank_violation=float(final_violation), max_accepted_violation=max_violation,
                             parameter_change_norm=float(np.linalg.norm(theta)),
                             initial_adaptive_parameters=64 if method=="large_mixed" else 24,
                             final_adaptive_parameters=theta.size, parameter_step_exposure=parameter_exposure,
                             growth=growth, trajectory=trajectory,
                             bank_view_counts={v:sum(r["view"]==v for r in bank) for v in VIEWS}))
    return dict(seed=seed, train_original_images=len(train), train_presentations=len(train_x),
                old_calibration_images=len(old_cal), new_calibration_images=len(new_cal), test_images=len(test),
                split_sha256={name:hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
                              for name,ids in (("train",train),("old_cal",old_cal),("new_cal",new_cal),("test",test))},
                training_iterations=int(model.n_iter_), training_warnings=[str(w.message) for w in training_warnings],
                versions=dict(numpy=np.__version__,sklearn=sklearn.__version__),
                resources=dict(base_model_values=int(sum(a.size for a in (*model.coefs_,*model.intercepts_))),
                               retained_cue_values=20*64, retained_labels=20, retained_limits_or_references=40,
                               retained_cues=20, initial_adaptive_values=24, maximum_adaptive_values=64,
                               extra_path_fixed_values=260, extra_path_trainable_values=40,
                               replay_encoding_cache_max_values=20*24,
                               adaptation_image_values=64*64,adaptation_labels=64,
                               adaptation_encoding_cache_max_values=64*24,
                               extra_activation_workspace_max_values=64*4,
                               fd_and_projection_workspace_max_values=(21+22)*64,
                               selection_peak_cues_per_policy=21,
                               note="Each policy receives the same calibration stream. Only its selected bank persists. Backend, labels, probe generators, candidate evaluation and array caches are explicit software resources. The larger control has extra paths from adaptation start, not from base pretraining."),
                runs=runs)


def gate8(seeds=(113,127,139), *, steps=12, budget=45000):
    specimens = [run_specimen(seed, steps=steps, budget=budget) for seed in seeds]
    summaries = {}
    for method in METHODS:
        group = [r for s in specimens for r in s["runs"] if r["method"]==method]
        summaries[method] = {key:float(np.mean([r[key] for r in group])) for key in (
            "changed_accuracy_before", "changed_accuracy_after", "objective_before", "objective_after",
            "scalar_calls", "model_example_evaluations", "final_bank_violation", "final_adaptive_parameters")}
        for key in group[0]["after"]:
            summaries[method][key+"_before"] = float(np.mean([r["before"][key] for r in group]))
            summaries[method][key+"_after"] = float(np.mean([r["after"][key] for r in group]))
        summaries[method]["growth_events"] = sum(len(r["growth"]) for r in group)
        for key in group[0]["paired_access_changes"]:
            summaries[method][key] = float(np.mean([r["paired_access_changes"][key] for r in group]))
    counterexample = tolerance_counterexample()
    runs = [r for s in specimens for r in s["runs"]]
    checks = dict(tolerances_allow_full_rank_progress=counterexample["bounds"]["new_answer"]>2.099,
                  protected_banks_respected=all(r["final_bank_violation"]<=1e-12 for r in runs if r["method"]!="unprotected"),
                  measurement_caps=all(r["scalar_calls"]<=r["call_budget"] for r in runs),
                  structural_cap=all(r["final_adaptive_parameters"]<=64 and len(r["growth"])<=1 for r in runs))
    return dict(gate=8, status="recognition_access_and_bounded_capacity_experiment_not_biological_recall",
                dataset="scikit-learn bundled UCI optical handwritten digits",
                source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
                protocol=dict(seeds=list(seeds),steps=steps,scalar_budget_per_method=budget,
                              bank_capacity=20,views=list(VIEWS),base_hidden_units=24,extra_hidden_units=4,
                              new_contrast="0.55*x+0.22",new_objective="mean correct-label log probability",
                              growth_trigger="first blocked proposal, once, if another measured step fits the remaining budget",
                              heldout_policy="upper first; ask lower only when top-two probability gap <0.2; average if asked",
                              fixed_before_heldout_evaluation=True,
                              metric_history="paired route-loss diagnostics added after the first seed pilot; learner settings unchanged"),
                checks=checks,**{"pass":all(checks.values())},counterexample=counterexample,
                summaries=summaries,specimens=specimens,
                limitations=["recognition of classes from image fragments, not face/name autobiographical recall",
                             "partial cue types are supplied; the learner does not invent semantic cue routes",
                             "growth adds random nonlinear features and trainable output paths, not biological morphology",
                             "same final capacity from adaptation start is a control; full pretraining with a larger body is not tested",
                             "preservation only applies to the retained calibration cues",
                             "boundary selection is static and uses labels; it is not autonomous task-value discovery",
                             "clean base training and digital candidate evaluation remain granted",
                             "growth need not activate or outperform a fixed model; no superiority check is a CI gate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,nargs="+",default=[113,127,139])
    parser.add_argument("--steps",type=int,default=12)
    parser.add_argument("--budget",type=int,default=45000)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = gate8(args.seeds,steps=args.steps,budget=args.budget)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"checks":result["checks"],"summaries":result["summaries"]},indent=2))
    if not result["pass"]:
        raise SystemExit(1)
