from __future__ import annotations

import math
from typing import Any

import numpy as np


def nullspace_projected_direction(
    protected_sensitivities: np.ndarray,
    proposed_direction: np.ndarray,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """Project a proposed material change into the local preservation subspace.

    A safe infinitesimal direction d must satisfy P d = 0, where rows of P are
    sensitivities of responses that should remain unchanged. This function returns
    the component of ``proposed_direction`` in null(P).

    This is a mathematical reference, not a claim that the neuron-shaped learner can
    derive P from its present local signals. Discovering or embodying this information
    is the next architectural problem.
    """

    pmat = np.asarray(protected_sensitivities, dtype=float)
    proposal = np.asarray(proposed_direction, dtype=float)
    if pmat.ndim != 2:
        raise ValueError("protected_sensitivities must be a 2D matrix")
    if proposal.ndim != 1 or proposal.size != pmat.shape[1]:
        raise ValueError("proposed_direction must match the material dimension")

    # SVD gives an auditable orthogonal projector even if protected rows are
    # redundant. The right-singular vectors whose singular values are effectively
    # zero span null(P).
    _u, s, vh = np.linalg.svd(pmat, full_matrices=True)
    if s.size == 0:
        return proposal.copy()
    tol = max(atol, np.max(s) * max(pmat.shape) * np.finfo(float).eps)
    rank = int(np.sum(s > tol))
    null_basis = vh[rank:].T
    if null_basis.size == 0:
        return np.zeros_like(proposal)
    return null_basis @ (null_basis.T @ proposal)


def delayed_error_stability_limit(delay: int) -> float:
    """Exact boundary for delta[t+1] = delta[t] - a delta[t-delay]."""

    if delay < 0:
        raise ValueError("delay must be non-negative")
    return float(2.0 * math.sin(math.pi / (4 * delay + 2)))


def delayed_error_spectral_radius(delay: int, effective_gain: float) -> float:
    """Spectral radius of z^(D+1) - z^D + a = 0 for the reference recurrence."""

    if delay < 0:
        raise ValueError("delay must be non-negative")
    a = float(effective_gain)
    coeff = np.zeros(delay + 2, dtype=float)
    coeff[0] = 1.0
    coeff[1] = -1.0
    coeff[-1] += a
    roots = np.roots(coeff)
    return float(np.max(np.abs(roots)))


def gate4_compatible_change_reference() -> dict[str, Any]:
    """Reference answer to: what may change without damaging old responses?

    Four gains participate in three protected, overlapping responses. A fourth response
    must move from 2 to 3. The reference has privileged access to the response
    sensitivity matrix P and therefore computes the preservation direction explicitly.

    Gate 4 is deliberately a *target* for the later continuous learner, not evidence
    that local efference-copy + receipt signals already discover this direction.
    """

    protected = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    new_probe = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
    gains0 = np.ones(4, dtype=float)
    target = 3.0

    direction = nullspace_projected_direction(protected, new_probe)
    direction_effect = float(new_probe @ direction)
    if abs(direction_effect) < 1e-12:
        raise AssertionError("constructed compatible example unexpectedly has no safe direction")

    # Normalize so one unit of coordinate motion changes the new response by one unit.
    direction = direction / direction_effect

    protected_before = protected @ gains0
    new_before = float(new_probe @ gains0)

    eta = 0.5
    gains = gains0.copy()
    trajectory = [new_before]
    protected_max_drift = 0.0
    for _ in range(20):
        error = target - float(new_probe @ gains)
        gains = gains + eta * error * direction
        trajectory.append(float(new_probe @ gains))
        protected_max_drift = max(
            protected_max_drift,
            float(np.max(np.abs(protected @ gains - protected_before))),
        )

    protected_after = protected @ gains
    new_after = float(new_probe @ gains)

    # Naive active-only strengthening can solve the new response but damages the old bank.
    naive_direction = new_probe / float(new_probe @ new_probe)
    naive = gains0 + (target - new_before) * naive_direction
    naive_protected_drift = protected @ naive - protected_before

    # Exact incompatibility control: four independent protected responses span R^4.
    fully_protected = np.eye(4, dtype=float)
    impossible_direction = nullspace_projected_direction(fully_protected, new_probe)

    delay_checks: dict[str, Any] = {}
    for delay in (0, 1, 3):
        limit = delayed_error_stability_limit(delay)
        below = 0.99 * limit
        above = 1.01 * limit
        delay_checks[str(delay)] = {
            "stability_limit": limit,
            "spectral_radius_below": delayed_error_spectral_radius(delay, below),
            "spectral_radius_above": delayed_error_spectral_radius(delay, above),
        }

    checks = {
        "safe_direction_preserves_old_responses": bool(
            np.max(np.abs(protected @ direction)) < 1e-12
        ),
        "safe_direction_changes_new_response": bool(abs(new_probe @ direction) > 0.99),
        "error_driven_learning_converges": bool(abs(new_after - target) < 2e-6),
        "old_response_bank_stays_fixed": bool(protected_max_drift < 1e-12),
        "naive_local_change_damages_old_bank": bool(
            np.max(np.abs(naive_protected_drift)) > 0.4
        ),
        "full_rank_protection_makes_change_impossible": bool(
            np.linalg.norm(impossible_direction) < 1e-12
        ),
        "delay_boundary_matches_root_test": bool(
            all(
                item["spectral_radius_below"] < 1.0
                and item["spectral_radius_above"] > 1.0
                for item in delay_checks.values()
            )
        ),
    }

    return {
        "gate": 4,
        "name": "compatible_change_reference",
        "status": "mathematical_reference_not_discovered_by_local_learner",
        "protected_sensitivity_matrix": protected.tolist(),
        "new_probe": new_probe.tolist(),
        "initial_gains": gains0.tolist(),
        "safe_direction": direction.tolist(),
        "protected_before": protected_before.tolist(),
        "protected_after": protected_after.tolist(),
        "new_response_before": new_before,
        "new_response_after_20_updates": new_after,
        "new_response_trajectory": trajectory,
        "max_protected_response_drift": protected_max_drift,
        "naive_active_only_final_gains": naive.tolist(),
        "naive_protected_response_drift": naive_protected_drift.tolist(),
        "incompatible_control_safe_direction_norm": float(np.linalg.norm(impossible_direction)),
        "delay_stability": delay_checks,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "declared_missing_mechanism": (
            "This reference is given P, the sensitivities of responses that must be "
            "preserved. IttnasNoruen does not yet know how to obtain or embody that "
            "constraint information from its permitted local signals, receipts, replay, "
            "geometry or additional measurements. That is the substantive next target."
        ),
    }
