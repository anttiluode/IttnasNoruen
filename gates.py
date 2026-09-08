from __future__ import annotations

import json
from typing import Any

import numpy as np

from ittnas_noruen import IttnasNoruen


SELF_SIGNATURE = np.array([0.90, 0.45, -0.35, 0.70], dtype=float)


def _event(n: int, branch: int, amplitude: float = 1.2) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    x[branch] = amplitude
    return x


def gate0_efference_copy_credit() -> dict[str, Any]:
    """Does prediction prevent predictable self-echo from contaminating delayed credit?"""

    target = 2
    with_copy = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True)
    no_copy = IttnasNoruen(SELF_SIGNATURE, echo_enabled=False)

    # Declared self-only calibration. This is intentionally easy and will be removed later.
    with_copy.calibrate_self([1.0] * 40)
    no_copy.calibrate_self([1.0] * 40)

    predictor_error = with_copy.self_prediction_error()

    for learner in (with_copy, no_copy):
        learner.clear_eligibility()
        learner.observe(1.0, _event(learner.n_branches, target), learn_echo=False)
        learner.quiet(3)
        learner.apply_consequence(+1.0)

    result = {
        "gate": 0,
        "name": "efference_copy_cleans_delayed_credit",
        "target_branch": target,
        "delay_ticks": 3,
        "predictor_error_after_calibration": predictor_error,
        "with_copy": {
            "material": with_copy.material.tolist(),
            "target_fraction": with_copy.target_fraction(target),
            "off_target_norm": with_copy.off_target_norm(target),
        },
        "no_copy": {
            "material": no_copy.material.tolist(),
            "target_fraction": no_copy.target_fraction(target),
            "off_target_norm": no_copy.off_target_norm(target),
        },
    }

    result["pass"] = bool(
        result["with_copy"]["target_fraction"] > 0.98
        and result["no_copy"]["target_fraction"] < 0.60
        and result["with_copy"]["off_target_norm"]
        < 0.02 * result["no_copy"]["off_target_norm"]
    )
    return result


def gate1_surprise_is_not_memory() -> dict[str, Any]:
    """Can the fast predictor adapt while slow material remains untouched without consequence?"""

    learner = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True)
    learner.calibrate_self([1.0] * 40)

    changed = SELF_SIGNATURE + np.array([0.20, -0.10, 0.08, 0.12])
    learner.set_true_self_signature(changed)

    before_error = learner.self_prediction_error()
    before_material = learner.material.copy()

    for _ in range(25):
        learner.observe(1.0, external=None, learn_echo=True)
        # Crucially: no apply_consequence call.

    after_error = learner.self_prediction_error()
    material_drift = float(np.linalg.norm(learner.material - before_material))

    result = {
        "gate": 1,
        "name": "prediction_error_is_not_automatically_structural_memory",
        "prediction_error_before_adaptation": before_error,
        "prediction_error_after_adaptation": after_error,
        "material_drift_without_consequence": material_drift,
    }
    result["pass"] = bool(after_error < 0.02 * before_error and material_drift == 0.0)
    return result


def _credit_episode(
    learner: IttnasNoruen,
    *,
    branch: int,
    delta: float,
    amplitude: float = 1.2,
    delay: int = 2,
) -> None:
    learner.clear_eligibility()
    learner.observe(1.0, _event(learner.n_branches, branch, amplitude), learn_echo=False)
    learner.quiet(delay)
    learner.apply_consequence(delta)


def gate2_reversal_with_preservation() -> dict[str, Any]:
    """Can one addressed distinction reverse while another useful one is preserved?"""

    learner = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True)
    learner.calibrate_self([1.0] * 60)

    protected = 0
    reversible = 1

    for _ in range(3):
        _credit_episode(learner, branch=protected, delta=+1.0)
    for _ in range(3):
        _credit_episode(learner, branch=reversible, delta=+1.0)

    protected_before = float(learner.material[protected])
    reversible_before = float(learner.material[reversible])

    # More negative than positive episodes so the association truly reverses.
    for _ in range(7):
        _credit_episode(learner, branch=reversible, delta=-1.0)

    protected_after = float(learner.material[protected])
    reversible_after = float(learner.material[reversible])
    protected_drift = abs(protected_after - protected_before)

    result = {
        "gate": 2,
        "name": "reversal_with_addressed_preservation",
        "protected_branch": protected,
        "reversible_branch": reversible,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_drift": protected_drift,
        "reversible_before": reversible_before,
        "reversible_after": reversible_after,
        "material": learner.material.tolist(),
    }
    result["pass"] = bool(
        protected_before > 0.0
        and reversible_before > 0.0
        and reversible_after < 0.0
        and protected_drift < 1e-6
    )
    return result


def run_all() -> list[dict[str, Any]]:
    return [
        gate0_efference_copy_credit(),
        gate1_surprise_is_not_memory(),
        gate2_reversal_with_preservation(),
    ]


if __name__ == "__main__":
    receipts = run_all()
    print(json.dumps(receipts, indent=2, sort_keys=True))
    if not all(item["pass"] for item in receipts):
        raise SystemExit(1)
