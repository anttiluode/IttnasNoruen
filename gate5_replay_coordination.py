from __future__ import annotations

import json
from typing import Any

import numpy as np

from gate4_compatible_change import nullspace_projected_direction
from ittnas_noruen import IttnasNoruen
from replay_coordination import ReplayPlasticityCoordinator


PROTECTED_PROBES = np.array(
    [
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ],
    dtype=float,
)
NEW_PROBE = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
SELF_SIGNATURE = np.array([0.90, 0.45, -0.35, 0.70], dtype=float)


def _build_coordinator(
    *,
    readout_weights: np.ndarray | None = None,
    protected_probes: np.ndarray = PROTECTED_PROBES,
) -> tuple[IttnasNoruen, ReplayPlasticityCoordinator]:
    weights = np.ones(4, dtype=float) if readout_weights is None else readout_weights
    learner = IttnasNoruen(
        SELF_SIGNATURE,
        echo_enabled=True,
        echo_lr=0.25,
        material_feedback=1.0,
        readout_weights=weights,
        max_receipts=4,
    )
    coordinator = ReplayPlasticityCoordinator(
        learner,
        max_protected=int(protected_probes.shape[0]),
    )
    for i, probe in enumerate(protected_probes):
        coordinator.add_protected(
            f"old-{i}",
            probe,
            learner.soma_response(0.0, probe),
        )
    return learner, coordinator


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _acquire_proposal(
    learner: IttnasNoruen,
    coordinator: ReplayPlasticityCoordinator,
    *,
    cycles: int = 20,
    new_probe: np.ndarray = NEW_PROBE,
) -> tuple[np.ndarray, dict[str, Any]]:
    initial, new_before = coordinator.proposal_from_new_question(new_probe)
    shaped, history = coordinator.shape_proposal(initial, cycles=cycles)
    normalized, raw_effect = coordinator.normalize_for_question(shaped, new_probe)
    return normalized, {
        "new_response_before": new_before,
        "initial_proposal": initial.tolist(),
        "proposal_before_normalization": shaped.tolist(),
        "proposal_effect_before_normalization": raw_effect,
        "normalized_proposal": normalized.tolist(),
        "protected_replays": cycles * len(coordinator.bank.items),
        "total_probe_measurements": coordinator.replay_count,
        "probe_energy": coordinator.probe_energy,
        "last_projection_effect": (
            history[-1].predicted_proposal_effect if history else None
        ),
    }


def _privileged_reference(
    learner: IttnasNoruen,
    protected_probes: np.ndarray,
    new_probe: np.ndarray,
) -> np.ndarray:
    # Evaluator only. The replay learner never receives this matrix.
    pmat = protected_probes * learner.readout_weights[None, :]
    new_sensitivity = new_probe * learner.readout_weights
    reference = nullspace_projected_direction(pmat, new_sensitivity)
    effect = float(new_sensitivity @ reference)
    if abs(effect) < 1e-12:
        return np.zeros_like(reference)
    return reference / effect


def _direct_write_then_repair_transient() -> dict[str, Any]:
    """Attacker: expose real material to corrections instead of shaping a temp proposal."""

    g = np.ones(4, dtype=float)
    old_targets = PROTECTED_PROBES @ g
    new_target = 3.0
    max_old_drift = 0.0
    trajectory: list[float] = []

    # First solve the new question directly. This immediately damages all old responses.
    new_sens = NEW_PROBE.copy()
    new_error = new_target - float(NEW_PROBE @ g)
    g += new_error * new_sens / float(new_sens @ new_sens)
    max_old_drift = max(max_old_drift, float(np.max(np.abs(PROTECTED_PROBES @ g - old_targets))))
    trajectory.append(max_old_drift)

    # Then repeatedly repair old questions and the new question in the actual material.
    for _ in range(50):
        for probe, target in zip(PROTECTED_PROBES, old_targets):
            error = float(target - probe @ g)
            g += error * probe / float(probe @ probe)
            max_old_drift = max(
                max_old_drift,
                float(np.max(np.abs(PROTECTED_PROBES @ g - old_targets))),
            )
        new_error = new_target - float(NEW_PROBE @ g)
        g += new_error * NEW_PROBE / float(NEW_PROBE @ NEW_PROBE)
        trajectory.append(float(np.max(np.abs(PROTECTED_PROBES @ g - old_targets))))

    return {
        "max_transient_protected_drift": max_old_drift,
        "final_gains": g.tolist(),
        "final_protected_drift": (PROTECTED_PROBES @ g - old_targets).tolist(),
        "final_new_response": float(NEW_PROBE @ g),
        "drift_trajectory_tail": trajectory[-5:],
    }


def gate5_replay_driven_coordination() -> dict[str, Any]:
    """Discover a compatible write by replaying old questions one row at a time.

    This gate removes Gate 4's main privilege: the learner is not handed P or its null
    space. It keeps explicit support that must be counted: replayable protected probes,
    their reference answers, known fixed readout couplings, local branch gains, a
    temporary proposal scalar per branch, and one scalar aggregation/broadcast pathway.

    It still does not solve the full continuous task with confounded self/world activity,
    delayed receipt feedback and noisy/coupled nonlinear propagation. It establishes the
    missing compatibility operation under the current gain model.
    """

    # ------------------------------------------------------------------
    # A. Standard overlapping reference: recover the compatible direction from replay.
    # ------------------------------------------------------------------
    learner, coordinator = _build_coordinator()
    protected_before = coordinator.protected_response_bank()
    proposal, acquisition = _acquire_proposal(learner, coordinator, cycles=20)
    reference = _privileged_reference(learner, PROTECTED_PROBES, NEW_PROBE)

    target = 3.0
    new_before = learner.soma_response(0.0, NEW_PROBE)
    task_error = target - new_before
    gain_delta = coordinator.commit_gain_proposal(
        proposal,
        task_error=task_error,
        learning_rate=1.0,
    )
    protected_after = coordinator.protected_response_bank()
    new_after = learner.soma_response(0.0, NEW_PROBE)

    learned = {
        **acquisition,
        "privileged_reference_for_evaluation_only": reference.tolist(),
        "proposal_reference_cosine": _cosine(proposal, reference),
        "gain_delta": gain_delta.tolist(),
        "new_response_after_commit": new_after,
        "protected_before": protected_before.tolist(),
        "protected_after": protected_after.tolist(),
        "max_protected_drift": float(np.max(np.abs(protected_after - protected_before))),
        "replay_memory_scalar_values": coordinator.bank.stored_scalar_values,
        "temporary_proposal_scalar_values": learner.n_branches,
    }

    # ------------------------------------------------------------------
    # B. Error-driven repeated commits stop when the task is solved, while the echo
    # predictor is allowed to keep tracking the material-shaped self response.
    # ------------------------------------------------------------------
    online, online_coord = _build_coordinator()
    online_protected_before = online_coord.protected_response_bank()
    online_proposal, _online_acq = _acquire_proposal(online, online_coord, cycles=20)
    error_trajectory: list[float] = []
    echo_error_trajectory: list[float] = []
    max_gain_trajectory: list[float] = []

    for _ in range(20):
        # Ongoing self action/prediction; there is no separate post-write relearning phase.
        online.observe(1.0, external=None, learn_echo=True)
        current = online.soma_response(0.0, NEW_PROBE)
        error = target - current
        error_trajectory.append(error)
        online_coord.commit_gain_proposal(
            online_proposal,
            task_error=error,
            learning_rate=0.5,
        )
        echo_error_trajectory.append(online.self_prediction_error())
        max_gain_trajectory.append(float(np.max(np.abs(online.branch_gain()))))

    online_protected_after = online_coord.protected_response_bank()
    online_new_after = online.soma_response(0.0, NEW_PROBE)
    online_result = {
        "new_response_after_20_error_driven_commits": online_new_after,
        "final_task_error": target - online_new_after,
        "max_protected_drift": float(
            np.max(np.abs(online_protected_after - online_protected_before))
        ),
        "final_echo_prediction_error": online.self_prediction_error(),
        "max_abs_branch_gain": float(np.max(np.abs(online.branch_gain()))),
        "task_error_trajectory": error_trajectory,
        "echo_error_trajectory_tail": echo_error_trajectory[-5:],
        "max_gain_trajectory_tail": max_gain_trajectory[-5:],
    }

    # ------------------------------------------------------------------
    # C. Different branch/readout contributions: same replay operation should discover a
    # different compatible direction and still agree with the evaluator's reference.
    # ------------------------------------------------------------------
    weights = np.array([0.7, 1.1, 0.9, 1.3], dtype=float)
    shifted, shifted_coord = _build_coordinator(readout_weights=weights)
    shifted_before = shifted_coord.protected_response_bank()
    shifted_proposal, _shifted_acq = _acquire_proposal(shifted, shifted_coord, cycles=30)
    shifted_reference = _privileged_reference(shifted, PROTECTED_PROBES, NEW_PROBE)
    shifted_target = shifted.soma_response(0.0, NEW_PROBE) + 1.0
    shifted_error = shifted_target - shifted.soma_response(0.0, NEW_PROBE)
    shifted_coord.commit_gain_proposal(
        shifted_proposal,
        task_error=shifted_error,
        learning_rate=1.0,
    )
    shifted_after = shifted_coord.protected_response_bank()
    shifted_result = {
        "readout_weights": weights.tolist(),
        "learned_proposal": shifted_proposal.tolist(),
        "privileged_reference_for_evaluation_only": shifted_reference.tolist(),
        "proposal_reference_cosine": _cosine(shifted_proposal, shifted_reference),
        "max_protected_drift": float(np.max(np.abs(shifted_after - shifted_before))),
        "new_response_error": float(
            shifted_target - shifted.soma_response(0.0, NEW_PROBE)
        ),
    }

    # ------------------------------------------------------------------
    # D. Full-rank protection: replay should erase the entire proposal and expose conflict.
    # ------------------------------------------------------------------
    full_rank_probes = np.eye(4, dtype=float)
    impossible, impossible_coord = _build_coordinator(protected_probes=full_rank_probes)
    impossible_initial, _ = impossible_coord.proposal_from_new_question(NEW_PROBE)
    impossible_shaped, _ = impossible_coord.shape_proposal(impossible_initial, cycles=3)
    impossible_result = {
        "proposal_norm_after_full_rank_replay": float(np.linalg.norm(impossible_shaped)),
        "compatible_change_detected": bool(np.linalg.norm(impossible_shaped) > 1e-10),
    }

    # ------------------------------------------------------------------
    # E. Omit one protected replay: the untested response is allowed to drift.
    # ------------------------------------------------------------------
    omitted_probes = PROTECTED_PROBES[:2]
    omitted, omitted_coord = _build_coordinator(protected_probes=omitted_probes)
    omitted_third_before = omitted.soma_response(0.0, PROTECTED_PROBES[2])
    omitted_proposal, _ = _acquire_proposal(omitted, omitted_coord, cycles=20)
    omitted_error = 3.0 - omitted.soma_response(0.0, NEW_PROBE)
    omitted_coord.commit_gain_proposal(
        omitted_proposal,
        task_error=omitted_error,
        learning_rate=1.0,
    )
    omitted_third_after = omitted.soma_response(0.0, PROTECTED_PROBES[2])
    omitted_result = {
        "untested_response_before": omitted_third_before,
        "untested_response_after": omitted_third_after,
        "untested_response_drift": omitted_third_after - omitted_third_before,
    }

    # ------------------------------------------------------------------
    # F. Compare against changing real material first and repairing afterward.
    # ------------------------------------------------------------------
    direct_repair = _direct_write_then_repair_transient()

    checks = {
        "replay_recovers_reference_direction": bool(learned["proposal_reference_cosine"] > 0.999999),
        "one_commit_solves_new_response": bool(abs(new_after - target) < 1e-6),
        "overlapping_old_responses_are_preserved": bool(learned["max_protected_drift"] < 1e-6),
        "error_driven_commits_converge_without_gain_runaway": bool(
            abs(online_result["final_task_error"]) < 2e-6
            and online_result["max_abs_branch_gain"] < 2.0
        ),
        "online_echo_tracks_changed_material": bool(
            online_result["final_echo_prediction_error"] < 0.01
        ),
        "different_contributions_produce_matching_new_direction": bool(
            shifted_result["proposal_reference_cosine"] > 0.999999
            and shifted_result["max_protected_drift"] < 1e-6
            and abs(shifted_result["new_response_error"]) < 1e-6
        ),
        "full_rank_replay_reports_incompatibility": bool(
            impossible_result["proposal_norm_after_full_rank_replay"] < 1e-10
        ),
        "unreplayed_behavior_is_not_magically_protected": bool(
            abs(omitted_result["untested_response_drift"]) > 0.4
        ),
        "temporary_proposal_avoids_large_transient_damage": bool(
            direct_repair["max_transient_protected_drift"] > 0.49
            and learned["max_protected_drift"] < 1e-6
        ),
    }

    return {
        "gate": 5,
        "name": "replay_driven_candidate_plasticity",
        "status": "mechanism_prototype_not_yet_full_continuous_task",
        "learned_compatible_change": learned,
        "error_driven_online_tracking": online_result,
        "changed_branch_contributions": shifted_result,
        "full_rank_incompatibility": impossible_result,
        "omitted_replay_attacker": omitted_result,
        "direct_write_then_repair_attacker": direct_repair,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "declared_resources": {
            "local_state": "one temporary proposal scalar z_j per branch",
            "sensitivity_source": "zero-command replay residual, local gain, fixed readout coupling",
            "coordination": "sum a_j*z_j, sum a_j^2, one broadcast correction scalar",
            "protected_memory": "dense replay probe plus reference answer per protected question",
            "delayed_credit": "existing receipt system is separate and not integrated into this gate",
        },
        "declared_limit": (
            "Gate 5 establishes replay-driven compatibility in the present diagonal gain model. "
            "It does not yet demonstrate the same mechanism under simultaneous command-confounded "
            "external activity, delayed receipt feedback, noise, finite replay bandwidth, or a "
            "coupled nonlinear arbor."
        ),
    }


if __name__ == "__main__":
    result = gate5_replay_driven_coordination()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)
