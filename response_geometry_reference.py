"""An exact reference: unchanged answers need not preserve nearby sensitivity.

This is a scalar counterexample, not a fluid model or empirical novelty claim.
The guard receives only measured responses. Derivatives are evaluator diagnostics.
"""
import json
from pathlib import Path

import numpy as np

from behavioral_guard import BehavioralUpdateGuard, ResponseMeter


def response(theta, query):
    return float(query+theta[0]*query*(query-1.))


def run_reference():
    outcomes = {}
    for method in ("anchors_only", "anchors_and_nearby"):
        initial = np.zeros(1)
        meter = ResponseMeter(response, 200)
        guard = BehavioralUpdateGuard(6, trust_radius=10.)
        for q in (0., 1.):
            guard.remember(q, meter(initial,q), 1e-8)
        if method == "anchors_and_nearby":
            for q in (-.02,.02,.98,1.02):
                guard.remember(q, meter(initial,q), .01)
        decision = guard.step(initial, .5, -1.5, meter)
        theta = decision.parameters
        # These diagnostics are not exposed to the learner or used to choose a step.
        q = np.array([-.02,.02,.98,1.02])
        drift = q+theta[0]*q*(q-1.)-q
        outcomes[method] = dict(status=decision.status, material=float(theta[0]),
            new_answer_before=.5, new_answer_after=response(theta,.5),
            max_anchor_drift=max(abs(response(theta,x)-x) for x in (0.,1.)),
            max_nearby_drift=float(np.max(np.abs(drift))),
            maximum_absolute_anchor_slope=float(max(abs(1.-theta[0]),abs(1.+theta[0]))),
            retained_queries=len(guard.references), scalar_calls=meter.calls, scalar_budget=200)
    checks = dict(
        exact_answers_can_hide_changed_sensitivity=outcomes["anchors_only"]["max_anchor_drift"]==0.
            and outcomes["anchors_only"]["maximum_absolute_anchor_slope"]>8.,
        nearby_contract_limits_the_change=outcomes["anchors_and_nearby"]["max_nearby_drift"]<=.01+1e-12,
        limited_new_progress=outcomes["anchors_and_nearby"]["new_answer_after"]<.5
            and outcomes["anchors_and_nearby"]["new_answer_after"]>-1.5)
    return dict(status="mathematical_counterexample_not_an_empirical_transfer_result",
        formula="f_k(q)=q+k*q*(q-1)", epsilon=.02, outcomes=outcomes,
        checks=checks, **{"pass":all(checks.values())},
        limitation="Nearby probes bound only their measured responses. This does not certify all input directions, nonlinear stability, or a compressed Jacobian. More protection permits less of this conflicting task.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_reference()
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not result["pass"]:
        raise SystemExit(1)
