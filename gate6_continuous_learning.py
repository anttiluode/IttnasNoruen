"""Continuous integration: no self-only calibration and no zero-command replay.

Each five-tick action window contains a four-tick balanced calibration intervention
and a live query with u=+/-1. External activity is present on every tick. Feedback
arrives later with a window ID. All controls get the same opportunities and budgets.
The balanced block assumes the material/external base drive is fixed for four ticks;
material may change between blocks. This assumption is explicit, not biological.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from behavioral_guard import replay_projection
from ittnas_noruen import IttnasNoruen


PROBES = np.array([[1.,1.,0.,0.], [0.,1.,1.,0.],
                   [0.,0.,1.,1.], [1.,0.,1.,0.]])


@dataclass
class Receipt:
    window: int
    query: int
    due: int
    sensitivity: np.ndarray
    measured_output: float
    target: float
    target_version: int


@dataclass(frozen=True)
class StreamConfig:
    blocks: int = 640
    delay: int = 5
    receipt_capacity: int = 8
    replay_capacity: int = 3
    noise: float = 0.001
    dither: float = 0.2
    command_correlation: float = 0.
    dither_correlation: float = 0.
    learning_rate: float = 0.5
    max_write_norm: float = 0.08
    reverse_at: int = 320
    full_rank_protection: bool = False
    geometry_change_at: int = -1


def run_stream(seed: int, method: str = "proposal", predictor: str = "dither",
               config: StreamConfig = StreamConfig()) -> dict[str, Any]:
    if method not in ("proposal", "naive", "repair", "frozen"):
        raise ValueError("unknown material rule")
    if predictor not in ("dither", "correlation", "frozen"):
        raise ValueError("unknown echo predictor")
    if config.receipt_capacity < 1 or config.replay_capacity < 1 or config.dither <= 0:
        raise ValueError("capacities and dither must be positive")
    rng = np.random.default_rng(seed)
    probes = np.vstack((np.eye(4), PROBES[3])) if config.full_rank_protection else PROBES
    n_old = len(probes)-1
    new_id = n_old
    old_targets = probes[:n_old]@np.ones(4)
    world_signature = np.array([0.9, 0.45, -0.35, 0.7])
    confound = np.array([0.3, -0.2, 0.15, 0.25])
    plant = IttnasNoruen(world_signature, echo_enabled=True, material_feedback=1.,
                         readout_weights=np.ones(4), max_receipts=config.receipt_capacity)
    # Separate bounded digital receipts store an observed gradient/output for feedback.
    # These do not accumulate across windows, and do not decay: 4+2 real scalars each,
    # plus window/query/due/version IDs. They replace, rather than secretly supplement,
    # the older decaying receipt traces for this online learner.
    pending: list[Receipt] = []
    rows: dict[int, tuple[np.ndarray, int]] = {}
    protected = tuple(range(min(n_old, config.replay_capacity)))
    counts: Counter = Counter()
    old_max = 0.
    squared_old = 0.
    phase_errors: list[float] = []
    echo_errors: list[float] = []
    gains_max = 1.
    peak_pending = 0
    oldest_row = 0
    total_probe_energy = 0.
    command_energy = 0.
    history: list[dict[str, Any]] = []
    # Allocated protected memory is explicit: probe + target + measured row + age.
    # Readout weights and the current material are locally available as in Gate 5.
    for block in range(config.blocks):
        query = (new_id, *range(n_old))[block % (n_old+1)]
        probe = probes[query]
        # Hidden input-path change. Only observed returns reveal it to the learner.
        mixing = np.eye(4)
        if config.geometry_change_at >= 0 and block >= config.geometry_change_at:
            mixing[0,2] = .15
            mixing[1,3] = .10
        effective_probe = mixing@probe
        version = int(block >= config.reverse_at)
        new_target = 3. if version == 0 else 1.6
        target = new_target if query == new_id else float(old_targets[query])

        # All four sign combinations; randomized order, external probe never absent.
        signs = np.array([[1.,1.], [1.,-1.], [-1.,1.], [-1.,-1.]])
        signs = signs[rng.permutation(4)]
        returns = []
        commands = []
        for a, d in signs:
            u = a + config.dither*d
            outside = (effective_probe + config.command_correlation*a*confound
                       + config.dither_correlation*d*confound)
            obs = plant.observe(u, outside, learn_echo=False)
            returns.append(obs.observed + rng.normal(0., config.noise, 4))
            commands.append(u)
            command_energy += u*u
            total_probe_energy += float(probe@probe)
            counts["physical_ticks"] += 1
        returns = np.asarray(returns)
        commands = np.asarray(commands)
        if predictor == "dither":
            estimate = signs[:,1]@returns/(4*config.dither)
        elif predictor == "correlation":
            estimate = commands@returns/float(commands@commands)
        else:
            estimate = plant.echo_weights.copy()
        innovation = float(np.linalg.norm(estimate-plant.echo_weights))
        if predictor != "frozen":
            plant.echo_weights += 0.35*(estimate-plant.echo_weights)

        # A fifth, live query: the command remains nonzero. Its noisy residual is
        # the task output delivered to the teacher, not a hidden clean-world readout.
        a = float(rng.choice([-1.,1.]))
        outside = effective_probe + config.command_correlation*a*confound
        obs = plant.observe(a, outside, learn_echo=False)
        residual = obs.residual + rng.normal(0., config.noise, 4)
        measured_output = float(residual.sum())
        sensitivity = residual / obs.branch_gain
        counts["physical_ticks"] += 1
        counts["nonzero_command_queries"] += 1
        command_energy += 1.
        total_probe_energy += float(probe@probe)

        if query < n_old:
            counts["protected_replay_opportunities"] += 1
            if query in protected:
                rows[query] = (sensitivity.copy(), block)
                counts["retained_protected_replays"] += 1
            else:
                counts["unretained_protected_replays"] += 1

        # Delayed feedback uses the response/target from the named original window.
        # A version token is given to EVERY method when the task instruction changes.
        # It prevents old-target feedback from silently undoing the new instruction.
        ready = [r for r in pending if r.due <= block]
        pending = [r for r in pending if r.due > block]
        for receipt in ready:
            counts["feedback_delivered"] += 1
            if receipt.query == new_id and receipt.target_version != version:
                counts["obsolete_target_feedback"] += 1
                continue
            error = receipt.target-receipt.measured_output
            if method == "frozen":
                continue
            # Confidence is based only on observed command/return estimates. A stable
            # confounded estimate can pass this check, which is deliberately attacked.
            if innovation > 0.025:
                counts["writes_deferred_echo_innovation"] += 1
                continue
            if receipt.query != new_id and method != "repair":
                continue
            row = receipt.sensitivity
            proposal = error*row/max(float(row@row), 1e-12)
            if method == "proposal":
                if len(rows) != len(protected):
                    counts["writes_deferred_missing_replay"] += 1
                    continue
                sketch = np.array([rows[q][0] for q in protected])
                free = replay_projection(row, sketch, cycles=30)
                if np.linalg.norm(free) < 1e-10:
                    counts["no_resolved_free_direction"] += 1
                    continue
                # Restore the retained targets, as well as protecting against further
                # drift. These are estimated responses, with noise/staleness risk.
                correction = old_targets[list(protected)]-sketch@plant.branch_gain()
                proposal = replay_projection(proposal, sketch, correction, cycles=30)
                counts["scalar_correction_broadcasts"] += 30*len(protected)
                oldest_row = max(oldest_row, max(block-rows[q][1] for q in protected))
            step = config.learning_rate*proposal
            size = float(np.linalg.norm(step))
            if size > config.max_write_norm:
                step *= config.max_write_norm/size
                counts["trust_radius_limited"] += 1
            # Uniformly shorten a step to respect gain bounds; coordinate clipping
            # would itself rotate a preserving direction into a harmful one.
            gain = plant.branch_gain()
            scale = 1.
            for gj, dj in zip(gain, step):
                if dj > 0:
                    scale = min(scale, (3.-gj)/dj)
                elif dj < 0:
                    scale = min(scale, (0.15-gj)/dj)
            step *= max(0., scale)
            plant.material += step
            counts["material_commits"] += 1

        if len(pending) >= config.receipt_capacity:
            counts["receipt_overflows"] += 1
        else:
            pending.append(Receipt(block, query, block+config.delay, sensitivity.copy(),
                                   measured_output, target, version))
        peak_pending = max(peak_pending, len(pending))

        # Evaluator-only clean probe bank and true echo: never fed back to the learner.
        physical_answers = (probes@mixing.T)@plant.branch_gain()
        drift = physical_answers[:n_old]-old_targets
        old_max = max(old_max, float(np.max(np.abs(drift))))
        squared_old += float(np.mean(drift**2))
        phase_errors.append(float(new_target-physical_answers[new_id]))
        echo_errors.append(float(np.linalg.norm(plant.echo_weights-
                                               plant.branch_gain()*world_signature)))
        gains_max = max(gains_max, float(np.max(np.abs(plant.branch_gain()))))
        if block % 16 == 0 or block == config.blocks-1:
            history.append(dict(block=block, target=new_target,
                                new_response=float(physical_answers[new_id]),
                                max_old_drift=float(np.max(np.abs(drift))),
                                echo_error=echo_errors[-1]))
    pre_end = max(0, min(config.reverse_at, config.blocks)-32)
    pre_stop = min(config.reverse_at, config.blocks)
    return dict(method=method, predictor=predictor, seed=seed,
                config=config.__dict__,
                before_reversal_rmse=float(np.sqrt(np.mean(np.square(phase_errors[pre_end:pre_stop])))),
                final_window_rmse=float(np.sqrt(np.mean(np.square(phase_errors[-32:])))),
                final_new_response=float(physical_answers[new_id]),
                final_protected_responses=physical_answers[:n_old].tolist(),
                max_protected_drift=old_max,
                rms_protected_drift=float(np.sqrt(squared_old/config.blocks)),
                final_echo_error=echo_errors[-1],
                final_window_echo_error=float(np.mean(echo_errors[-32:])),
                max_abs_gain=gains_max, material_norm=float(np.linalg.norm(plant.material)),
                pending_peak=peak_pending, pending_at_end=len(pending),
                oldest_used_sensitivity_blocks=oldest_row,
                probe_energy=total_probe_energy, command_energy=command_energy,
                counts=dict(counts), history=history,
                memory=dict(protected_probe_target_scalars=5*len(protected),
                            sensitivity_sketch_scalars=4*len(protected),
                            receipt_scalar_capacity=6*config.receipt_capacity,
                            temporary_proposal_scalars=4,
                            calibration_return_buffer_scalars=16,
                            calibration_command_and_sign_scalars=12,
                            echo_weights=4, material=4, legacy_aggregate_eligibility=4,
                            note="IDs/ages/version bookkeeping additional; simulator truth and history are evaluator-only"))


def gate6_continuous_learning(seeds=(7, 19, 31)) -> dict[str, Any]:
    runs = [run_stream(seed, method) for seed in seeds
            for method in ("proposal", "naive", "repair", "frozen")]
    attacks = {
        "command_correlated_dither": run_stream(43, predictor="dither", config=StreamConfig(command_correlation=.2)),
        "command_correlated_regression": run_stream(43, predictor="correlation", config=StreamConfig(command_correlation=.2)),
        "world_follows_dither": run_stream(43, predictor="dither", config=StreamConfig(command_correlation=.2, dither_correlation=.04)),
        "receipt_overflow": run_stream(43, config=StreamConfig(receipt_capacity=2)),
        "replay_overflow": run_stream(43, config=StreamConfig(replay_capacity=2)),
        "long_delay": run_stream(43, config=StreamConfig(delay=40, receipt_capacity=48)),
        "full_rank_protection": run_stream(43, config=StreamConfig(full_rank_protection=True,replay_capacity=4)),
        "stale_response_geometry": run_stream(43, config=StreamConfig(geometry_change_at=160)),
    }
    checks = {
        "continuous_nonzero_commands": all(r["counts"]["nonzero_command_queries"]==r["config"]["blocks"] for r in runs),
        "matched_action_budget": all(
            r["counts"]["physical_ticks"]==runs[0]["counts"]["physical_ticks"]
            and np.isclose(r["probe_energy"],runs[0]["probe_energy"],rtol=0,atol=1e-8)
            and np.isclose(r["command_energy"],runs[0]["command_energy"],rtol=0,atol=1e-8)
            for r in runs),
        "bounded_receipts": all(r["pending_peak"] <= r["config"]["receipt_capacity"] for r in runs+list(attacks.values())),
        "bounded_material": all(r["max_abs_gain"] <= 3.+1e-12 for r in runs+list(attacks.values())),
        "overflow_visible": attacks["receipt_overflow"]["counts"].get("receipt_overflows",0)>0,
        "unretained_replay_visible": attacks["replay_overflow"]["counts"].get("unretained_protected_replays",0)>0,
    }
    # Mechanism performance is reported, not made a prerequisite for a green test.
    summaries = {}
    for method in ("proposal","naive","repair","frozen"):
        group=[r for r in runs if r["method"]==method]
        summaries[method]={key:float(np.mean([r[key] for r in group])) for key in (
            "before_reversal_rmse","final_window_rmse","max_protected_drift",
            "rms_protected_drift","final_window_echo_error","max_abs_gain")}
    return dict(gate=6, status="continuous_gain_model_benchmark_not_a_biophysical_neuron",
                checks=checks, **{"pass":all(checks.values())},
                summaries=summaries, runs=runs, attackers=attacks,
                limitations=["gain-model local derivative and supplied scalar coordination remain",
                             "external base drive is fixed within each four-tick calibration intervention",
                             "window IDs and task-version tokens are supplied equally to controls",
                             "action blocks and numerical receipts are a digital architecture",
                             "delay test includes a trust radius and cannot use Gate 4's scalar threshold as a theorem"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output")
    args=parser.parse_args()
    result=gate6_continuous_learning()
    payload=json.dumps(result,indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(payload+"\n")
        print(json.dumps({"summaries":result["summaries"],"checks":result["checks"]},indent=2))
    else:
        print(payload)
    if not result["pass"]:
        raise SystemExit(1)
