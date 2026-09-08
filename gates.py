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
    """Historical control: does prediction clean predictable self-echo from delayed credit?"""

    target = 2
    # Gate 0 intentionally preserves the original decoupled-material control. Gate 3
    # closes the loop by making written material alter later physical responses.
    with_copy = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True, material_feedback=0.0)
    no_copy = IttnasNoruen(SELF_SIGNATURE, echo_enabled=False, material_feedback=0.0)

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
    """Historical control: fast prediction can adapt without an immediate structural write."""

    learner = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True, material_feedback=0.0)
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
    """Historical addressed control: reverse one coordinate without touching another."""

    learner = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True, material_feedback=0.0)
    learner.calibrate_self([1.0] * 60)

    protected = 0
    reversible = 1

    for _ in range(3):
        _credit_episode(learner, branch=protected, delta=+1.0)
    for _ in range(3):
        _credit_episode(learner, branch=reversible, delta=+1.0)

    protected_before = float(learner.material[protected])
    reversible_before = float(learner.material[reversible])

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


def _off_target_fraction(update: np.ndarray, target: int) -> float:
    total = float(np.sum(np.abs(update)))
    if total == 0.0:
        return 0.0
    return float((total - abs(float(update[target]))) / total)


def _receipt_episode(
    learner: IttnasNoruen,
    receipt_id: str,
    *,
    branch: int,
    delta: float,
    amplitude: float = 1.2,
    delay: int = 2,
) -> np.ndarray:
    learner.begin_receipt(receipt_id)
    learner.observe(0.0, _event(learner.n_branches, branch, amplitude), learn_echo=False)
    learner.seal_receipt(receipt_id)
    learner.quiet(delay)
    return learner.apply_receipt_consequence(receipt_id, delta)


def gate3_material_behavior_and_credit_receipts() -> dict[str, Any]:
    """Close the material->activity loop and attack aggregate eligibility under overlap.

    This gate does not solve arbitrary causal credit. It tests one explicitly counted
    resource: a temporal/action receipt that stores the residual pattern produced inside a
    declared window. The delayed teaching signal later names the receipt, not a branch.
    """

    # ------------------------------------------------------------------
    # A. Written material must change a future measured response and thereby make the
    # self predictor stale. The fast predictor may adapt again without another write.
    # ------------------------------------------------------------------
    coupled = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True, material_feedback=1.0)
    coupled.calibrate_self([1.0] * 60)
    target = 2
    held_out = _event(coupled.n_branches, target, amplitude=0.7)

    soma_before = coupled.soma_response(0.0, held_out)
    predictor_before = coupled.self_prediction_error()
    _receipt_episode(coupled, "write-target", branch=target, delta=+1.0, delay=2)
    soma_after = coupled.soma_response(0.0, held_out)
    predictor_after_write = coupled.self_prediction_error()

    material_after_write = coupled.material.copy()
    coupled.clear_eligibility()
    for _ in range(30):
        coupled.observe(1.0, external=None, learn_echo=True)
    predictor_after_relearning = coupled.self_prediction_error()
    material_drift_during_relearning = float(
        np.linalg.norm(coupled.material - material_after_write)
    )

    behavior = {
        "soma_before": soma_before,
        "soma_after": soma_after,
        "soma_change": soma_after - soma_before,
        "prediction_error_before_write": predictor_before,
        "prediction_error_after_write": predictor_after_write,
        "prediction_error_after_relearning": predictor_after_relearning,
        "material_drift_during_relearning": material_drift_during_relearning,
    }

    # ------------------------------------------------------------------
    # B. Astra attacker: a stale self-model error survives in aggregate eligibility and a
    # later valid consequence for a different event writes it. A new receipt opened only
    # for the external-event window excludes that older residual without receiving a branch
    # label. This is a timing/action-boundary solution, not a universal causal oracle.
    # ------------------------------------------------------------------
    changed = SELF_SIGNATURE + np.array([0.20, -0.10, 0.08, 0.12])

    legacy = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True, material_feedback=0.0)
    legacy.calibrate_self([1.0] * 40)
    legacy.set_true_self_signature(changed)
    legacy.observe(1.0, external=None, learn_echo=True)
    legacy.observe(0.0, _event(legacy.n_branches, target), learn_echo=False)
    legacy.quiet(3)
    legacy_update = legacy.apply_consequence(+1.0)

    receipted = IttnasNoruen(SELF_SIGNATURE, echo_enabled=True, material_feedback=0.0)
    receipted.calibrate_self([1.0] * 40)
    receipted.set_true_self_signature(changed)
    receipted.observe(1.0, external=None, learn_echo=True)
    receipted.begin_receipt("external-2")
    receipted.observe(0.0, _event(receipted.n_branches, target), learn_echo=False)
    receipted.seal_receipt("external-2")
    receipted.quiet(3)
    receipt_update = receipted.apply_receipt_consequence("external-2", +1.0)

    stale_credit = {
        "legacy_update": legacy_update.tolist(),
        "legacy_off_target_fraction": _off_target_fraction(legacy_update, target),
        "receipt_update": receipt_update.tolist(),
        "receipt_off_target_fraction": _off_target_fraction(receipt_update, target),
    }

    # ------------------------------------------------------------------
    # C. Two events are outstanding before either consequence arrives. Aggregate credit
    # assigns the first consequence to both and clears the trace, so the second gets zero.
    # Separate temporal receipts preserve both pending traces.
    # ------------------------------------------------------------------
    legacy_overlap = IttnasNoruen([0.0] * 4, echo_enabled=True, material_feedback=0.0)
    legacy_overlap.observe(0.0, _event(4, 0), learn_echo=False)
    legacy_overlap.observe(0.0, _event(4, 1), learn_echo=False)
    legacy_overlap.quiet(2)
    legacy_first = legacy_overlap.apply_consequence(+1.0)
    legacy_second = legacy_overlap.apply_consequence(+1.0)

    ledger = IttnasNoruen([0.0] * 4, echo_enabled=True, material_feedback=0.0)
    ledger.begin_receipt("A")
    ledger.observe(0.0, _event(4, 0), learn_echo=False)
    ledger.seal_receipt("A")
    ledger.begin_receipt("B")
    ledger.observe(0.0, _event(4, 1), learn_echo=False)
    ledger.seal_receipt("B")
    ledger.quiet(2)
    receipt_a = ledger.apply_receipt_consequence("A", +1.0)
    receipt_b = ledger.apply_receipt_consequence("B", +1.0)

    overlap = {
        "legacy_first_update": legacy_first.tolist(),
        "legacy_second_update": legacy_second.tolist(),
        "receipt_A_update": receipt_a.tolist(),
        "receipt_B_update": receipt_b.tolist(),
    }

    # ------------------------------------------------------------------
    # D. Preservation is now measured through a fixed bounded soma readout. We write two
    # useful addressed relations, then revise branch 1 only. Branch 0's held-out response
    # should remain stable while branch 1's response changes materially.
    # ------------------------------------------------------------------
    preserve = IttnasNoruen([0.0] * 4, echo_enabled=True, material_feedback=1.0)
    _receipt_episode(preserve, "protect+", branch=0, delta=+1.0)
    _receipt_episode(preserve, "rev+", branch=1, delta=+1.0)

    probe0 = _event(4, 0, amplitude=0.7)
    probe1 = _event(4, 1, amplitude=0.7)
    bank_before = np.array(
        [preserve.soma_response(0.0, probe0), preserve.soma_response(0.0, probe1)]
    )

    for i in range(3):
        _receipt_episode(preserve, f"rev-{i}", branch=1, delta=-1.0)

    bank_after = np.array(
        [preserve.soma_response(0.0, probe0), preserve.soma_response(0.0, probe1)]
    )
    preservation = {
        "response_bank_before": bank_before.tolist(),
        "response_bank_after": bank_after.tolist(),
        "protected_response_drift": float(abs(bank_after[0] - bank_before[0])),
        "revised_response_change": float(bank_after[1] - bank_before[1]),
    }

    checks = {
        "material_changes_future_behavior": bool(abs(behavior["soma_change"]) > 0.005),
        "write_makes_predictor_stale": bool(
            predictor_after_write > max(1e-4, 100.0 * predictor_before)
        ),
        "predictor_relearns_without_material_write": bool(
            predictor_after_relearning < 0.01 * predictor_after_write
            and material_drift_during_relearning == 0.0
        ),
        "aggregate_trace_reproduces_stale_error_contamination": bool(
            stale_credit["legacy_off_target_fraction"] > 0.20
        ),
        "receipt_excludes_stale_error": bool(
            stale_credit["receipt_off_target_fraction"] < 1e-12
        ),
        "aggregate_overlap_is_misattributed": bool(
            abs(legacy_first[1]) > abs(legacy_first[0]) > 0.0
        ),
        "aggregate_clear_loses_second_credit": bool(np.linalg.norm(legacy_second) == 0.0),
        "receipts_keep_pending_causes_separate": bool(
            receipt_a[0] > 0.0
            and np.count_nonzero(np.abs(receipt_a) > 1e-12) == 1
            and receipt_b[1] > 0.0
            and np.count_nonzero(np.abs(receipt_b) > 1e-12) == 1
        ),
        "protected_future_response_survives_revision": bool(
            preservation["protected_response_drift"] < 1e-12
            and abs(preservation["revised_response_change"]) > 0.02
        ),
    }

    return {
        "gate": 3,
        "name": "material_behavior_and_credit_receipts",
        "behavior_coupling": behavior,
        "stale_prediction_credit": stale_credit,
        "overlapping_pending_consequences": overlap,
        "future_response_preservation": preservation,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "declared_limit": (
            "Receipts require a declared temporal/action boundary and feedback that names "
            "that receipt. They do not solve arbitrarily overlapping unlabeled causes."
        ),
    }


def run_all() -> list[dict[str, Any]]:
    return [
        gate0_efference_copy_credit(),
        gate1_surprise_is_not_memory(),
        gate2_reversal_with_preservation(),
        gate3_material_behavior_and_credit_receipts(),
    ]


if __name__ == "__main__":
    receipts = run_all()
    print(json.dumps(receipts, indent=2, sort_keys=True))
    if not all(item["pass"] for item in receipts):
        raise SystemExit(1)
