"""A real-data use of the same scalar-response update guard.

Adapt 24 hidden biases of a trained handwritten-digit classifier to changed image
contrast. Preserve ten retained correct-class probabilities; evaluate ALL held-out
images separately. Keeping ten anchors is explicitly not a generalization guarantee.
No test response influences learning, reference selection, hyperparameters or stopping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np

from behavioral_guard import BehavioralUpdateGuard, ResponseMeter


def run_digits(seed: int, steps: int = 24) -> dict:
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    x,y=load_digits(return_X_y=True)
    x=x/16.
    pool,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
    train,cal=train_test_split(pool,test_size=.2,stratify=y[pool],random_state=seed+1)
    with warnings.catch_warnings(record=True) as messages, threadpool_limits(limits=1):
        warnings.simplefilter("always",ConvergenceWarning)
        model=MLPClassifier(hidden_layer_sizes=(24,),activation="tanh",solver="lbfgs",
                            max_iter=200,random_state=seed).fit(x[train],y[train])
    anchors=[]
    predictions=model.predict(x[cal])
    for label in range(10):
        eligible=[i for i,p in zip(cal,predictions) if y[i]==label and p==label]
        if not eligible:
            raise RuntimeError("no correctly recognized calibration anchor for a class")
        anchors.append(int(eligible[0]))
    batch=np.array([i for i in cal if i not in anchors][:64])
    changed=lambda data: .55*data+.22
    w1,w2=model.coefs_
    b1,b2=model.intercepts_

    def probabilities(theta,features):
        logits=np.tanh(features@w1+b1+theta)@w2+b2
        logits-=logits.max(axis=1,keepdims=True)
        exp=np.exp(logits)
        return exp/exp.sum(axis=1,keepdims=True)

    zero=np.zeros(24)
    initial_old=probabilities(zero,x[test])
    initial_changed=probabilities(zero,changed(x[test]))
    initial_batch=probabilities(zero,changed(x[batch]))[np.arange(len(batch)),y[batch]].mean()
    target=min(.995,float(initial_batch)+.05)
    runs=[]
    with threadpool_limits(limits=1):
        for method in ("guard","tangent_only","unprotected","frozen"):
            model_examples=0
            def response(theta,query):
                nonlocal model_examples
                if query=="new":
                    model_examples+=len(batch)
                    p=probabilities(theta,changed(x[batch]))
                    return float(p[np.arange(len(batch)),y[batch]].mean())
                index=anchors[int(query)]
                model_examples+=1
                return float(probabilities(theta,x[index:index+1])[0,y[index]])
            meter=ResponseMeter(response,max_calls=15000)
            # Historical exact-tangent comparator; new access tests use bounds.
            guard=BehavioralUpdateGuard(max_references=10,trust_radius=.2,
                                         projection_mode="equalities")
            theta=zero.copy()
            for q in range(10):
                guard.remember(q,meter(theta,q),tolerance=.01)
            trajectory=[]
            max_anchor_drift=0.
            rejected=0
            if method!="frozen":
                for step in range(steps):
                    decision=guard.step(theta,"new",target,meter,
                                        protect=method!="unprotected",validate=method=="guard")
                    theta=decision.parameters
                    rejected+=decision.rejected_candidates
                    # Pure evaluation: bypass accounting callback; never feed back.
                    values=np.array([probabilities(theta,x[i:i+1])[0,y[i]] for i in anchors])
                    drift=float(np.max(np.abs(values-np.array([r.response for r in guard.references]))))
                    max_anchor_drift=max(max_anchor_drift,drift)
                    trajectory.append(dict(step=step,status=decision.status,anchor_drift=drift,
                                           scalar_calls=meter.calls))
                    if decision.status!="accepted":
                        break
            final_old=probabilities(theta,x[test])
            final_changed=probabilities(theta,changed(x[test]))
            final_batch=probabilities(theta,changed(x[batch]))[np.arange(len(batch)),y[batch]].mean()
            old_true_drift=np.abs(final_old[np.arange(len(test)),y[test]]-
                                  initial_old[np.arange(len(test)),y[test]])
            runs.append(dict(method=method,seed=seed,
                             old_test_accuracy_before=float(np.mean(initial_old.argmax(1)==y[test])),
                             old_test_accuracy_after=float(np.mean(final_old.argmax(1)==y[test])),
                             changed_test_accuracy_before=float(np.mean(initial_changed.argmax(1)==y[test])),
                             changed_test_accuracy_after=float(np.mean(final_changed.argmax(1)==y[test])),
                             adaptation_probability_before=float(initial_batch),
                             adaptation_probability_after=float(final_batch),target=target,
                             max_retained_anchor_drift=max_anchor_drift,
                             heldout_mean_probability_drift=float(np.mean(old_true_drift)),
                             heldout_max_probability_drift=float(np.max(old_true_drift)),
                             scalar_calls=meter.calls,model_example_evaluations=model_examples,
                             parameter_change_norm=float(np.linalg.norm(theta)),
                             rejected_candidates=rejected,trajectory=trajectory))
    return dict(seed=seed,training_examples=len(train),calibration_examples=len(cal),
                heldout_examples=len(test),adaptation_examples=len(batch),
                model_training_iterations=int(model.n_iter_),
                training_warnings=[str(m.message) for m in messages],
                sklearn_version=sklearn.__version__,numpy_version=np.__version__,
                resources=dict(adaptive_parameters=24,anchors=10,
                               anchor_image_values=640,anchor_response_tolerances=20,
                               anchor_labels=10,measured_sensitivity_workspace_values=264,
                               orthogonal_basis_workspace_max_values=240,
                               base_model_parameter_values=int(sum(a.size for a in (*model.coefs_,*model.intercepts_))),
                               note="adaptation images/labels and temporary model evaluations also required; test data are evaluator-only"),
                runs=runs)


def digits_transfer(seeds=(71,83,97),steps=24):
    specimens=[run_digits(seed,steps) for seed in seeds]
    all_runs=[r for s in specimens for r in s["runs"]]
    summaries={}
    for method in ("guard","tangent_only","unprotected","frozen"):
        group=[r for r in all_runs if r["method"]==method]
        summaries[method]={key:float(np.mean([r[key] for r in group])) for key in (
            "old_test_accuracy_before","old_test_accuracy_after",
            "changed_test_accuracy_before","changed_test_accuracy_after",
            "adaptation_probability_before","adaptation_probability_after",
            "max_retained_anchor_drift","heldout_mean_probability_drift",
            "heldout_max_probability_drift","scalar_calls","model_example_evaluations")}
    checks=dict(anchor_contract=all(r["max_retained_anchor_drift"]<=.01+1e-12 for r in all_runs if r["method"]=="guard"),
                finite_measurement_budget=all(r["scalar_calls"]<=15000 for r in all_runs))
    return dict(status="real_data_software_adapter_demo_not_a_continual_learning_benchmark_victory",
                dataset="scikit-learn bundled UCI optical handwritten digits",
                source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
                protocol="disjoint base-train/calibration/test split; 24 hidden biases; changed contrast .55*x+.22; ten calibration anchors; fixed 24-step/15000-call ceiling",
                checks=checks,**{"pass":all(checks.values())},summaries=summaries,specimens=specimens,
                limitations=["old held-out responses are not constrained by ten retained anchors",
                             "the readout is a scalar aggregate for the 64-example adaptation batch; per-example computation is counted",
                             "candidate network evaluations are an explicit software privilege",
                             "small dataset and three seeds do not establish superiority to GEM/OGD or modern adapters"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--seeds",type=int,nargs="+",default=[71,83,97])
    parser.add_argument("--steps",type=int,default=24)
    args=parser.parse_args()
    result=digits_transfer(args.seeds,args.steps)
    payload=json.dumps(result,indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(payload+"\n")
        print(json.dumps({"summaries":result["summaries"],"checks":result["checks"]},indent=2))
    else:
        print(payload)
    if not result["pass"]:
        raise SystemExit(1)
