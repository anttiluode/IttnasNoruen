"""Transfer the update guard to a coupled nonlinear propagating substrate.

The physical model is a controlled recurrent tree, not a biophysical neuron.
The learner sees only a scalar response callback at proposed parameter values.
This software evaluation capability is stronger than Gate 6's local interface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from behavioral_guard import BehavioralUpdateGuard, ResponseMeter


class NonlinearArbor:
    """Four three-edge arms, leaky state, saturating interactions, soma-only readout."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.edges = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
                      (0,7),(7,8),(8,9),(0,10),(10,11),(11,12)]
        self.couplings = rng.uniform(.18,.26,12)
        self.probes = []
        for _ in range(9):
            stimulus = np.zeros(13)
            stimulus[[3,6,9,12]] = rng.uniform(.3,1.8,4)
            self.probes.append(stimulus)

    def response(self, parameters: np.ndarray, query: int) -> float:
        adjacency = np.zeros((13,13))
        for (i,j), weight in zip(self.edges, self.couplings*np.exp(parameters)):
            adjacency[i,j] = adjacency[j,i] = weight
        x = np.zeros(13)
        total = 0.
        for t in range(36):
            drive = self.probes[query] if t < 12 else 0.
            x = .55*x + .35*np.tanh(adjacency@x + drive)
            total += x[0]
        return float(20*total)


def run_transfer(seed: int, method: str, *, steps: int = 30) -> dict:
    if method not in ("guard", "tangent_only", "unprotected", "frozen"):
        raise ValueError("unknown method")
    world = NonlinearArbor(seed)
    theta = np.zeros(12)
    meter = ResponseMeter(world.response, max_calls=5000)
    # Preserve the historical Gate-7 comparator and its published receipt.
    guard = BehavioralUpdateGuard(max_references=4, trust_radius=.35,
                                   projection_mode="equalities")
    for q in range(4):
        guard.remember(q,meter(theta,q),tolerance=.001)
    heldout_before = np.array([world.response(theta,q) for q in range(5,9)])
    new_before = meter(theta,4)
    target = new_before+.08
    max_drift = 0.
    max_heldout = 0.
    statuses = []
    trajectory = []
    rejected = 0
    normmax = 0.
    if method != "frozen":
        for step in range(steps):
            decision = guard.step(theta,4,target,meter,
                                  protect=method!="unprotected",validate=method=="guard")
            statuses.append(decision.status)
            theta = decision.parameters
            rejected += decision.rejected_candidates
            normmax = max(normmax,float(np.linalg.norm(theta)))
            # Evaluation only. These results cannot influence acceptance or stopping.
            old = np.array([world.response(theta,r.query)-r.response for r in guard.references])
            held = np.array([world.response(theta,q) for q in range(5,9)])-heldout_before
            max_drift = max(max_drift,float(np.max(np.abs(old))))
            max_heldout = max(max_heldout,float(np.max(np.abs(held))))
            trajectory.append(dict(step=step,new_response=world.response(theta,4),
                                   old_max_drift=float(np.max(np.abs(old))),
                                   calls=meter.calls,status=decision.status))
            if decision.status not in ("accepted",):
                break
    after = world.response(theta,4)
    return dict(seed=seed,method=method,new_before=new_before,target=target,new_after=after,
                remaining_error=target-after,max_protected_drift=max_drift,
                max_unprotected_probe_drift=max_heldout,max_parameter_norm=normmax,
                scalar_calls=meter.calls,call_budget=meter.max_calls,
                rejected_candidates=rejected,statuses=statuses,trajectory=trajectory,
                resources=dict(material_parameters=12,protected_references=4,
                               reference_target_tolerance_scalars=8,
                               measured_sensitivity_workspace_scalars=60,
                               stimulus_library_scalars=9*13,
                               note="query IDs, callback weights and candidate evaluation capability are additional declared software support"))


def gate7_measured_transfer(seeds=(11,23,47)) -> dict:
    runs=[run_transfer(seed,method) for seed in seeds
          for method in ("guard","tangent_only","unprotected","frozen")]
    protected=[r for r in runs if r["method"]=="guard"]
    checks=dict(
        measured_bank_is_respected=all(r["max_protected_drift"]<=.001+1e-12 for r in protected),
        finite_budget=all(r["scalar_calls"]<=r["call_budget"] for r in runs),
        some_target_progress=all(abs(r["remaining_error"])<.08 for r in protected),
    )
    summaries={}
    for method in ("guard","tangent_only","unprotected","frozen"):
        selected=[r for r in runs if r["method"]==method]
        summaries[method]={key:float(np.mean([r[key] for r in selected])) for key in (
            "remaining_error","max_protected_drift","max_unprotected_probe_drift",
            "scalar_calls","rejected_candidates","max_parameter_norm")}
    return dict(gate=7,status="software_measured_update_transfer_not_local_neuronal_plasticity",
                checks=checks,**{"pass":all(checks.values())},summaries=summaries,runs=runs,
                limitations=["deterministic replay with reset fast state for each measured response",
                             "candidate parameters can be evaluated without committing live material",
                             "only four retained responses are protected; held-out behavior is separately evaluated",
                             "finite differences measure the current response geometry, not a universal model",
                             "Gate 6 temporal integration and this coupled model are separate experiments"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output")
    args=parser.parse_args()
    result=gate7_measured_transfer()
    payload=json.dumps(result,indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(payload+"\n")
        print(json.dumps({"summaries":result["summaries"],"checks":result["checks"]},indent=2))
    else:
        print(payload)
    if not result["pass"]:
        raise SystemExit(1)
