import math

import numpy as np
import pytest

from gate4_compatible_change import (
    delayed_error_stability_limit,
    gate4_compatible_change_reference,
)
from gate5_replay_coordination import gate5_replay_driven_coordination
from gates import (
    gate0_efference_copy_credit,
    gate1_surprise_is_not_memory,
    gate2_reversal_with_preservation,
    gate3_material_behavior_and_credit_receipts,
)
from ittnas_noruen import IttnasNoruen
from replay_coordination import ReplayPlasticityCoordinator


def test_gate0_efference_copy_cleans_credit():
    result = gate0_efference_copy_credit()
    assert result["pass"]
    assert result["with_copy"]["target_fraction"] > 0.98
    assert result["no_copy"]["target_fraction"] < 0.60


def test_gate1_prediction_can_learn_without_structural_write():
    result = gate1_surprise_is_not_memory()
    assert result["pass"]
    assert result["material_drift_without_consequence"] == 0.0


def test_gate2_reversal_preserves_other_address():
    result = gate2_reversal_with_preservation()
    assert result["pass"]
    assert result["reversible_before"] > 0.0
    assert result["reversible_after"] < 0.0


def test_gate3_material_changes_behavior_and_receipts_fix_declared_overlap():
    result = gate3_material_behavior_and_credit_receipts()
    assert result["pass"]
    assert result["checks"]["material_changes_future_behavior"]
    assert result["checks"]["receipt_excludes_stale_error"]
    assert result["checks"]["receipts_keep_pending_causes_separate"]
    assert result["checks"]["protected_future_response_survives_revision"]


def test_gate4_reference_finds_overlapping_preserving_change():
    result = gate4_compatible_change_reference()
    assert result["pass"]
    assert np.allclose(result["safe_direction"], [0.5, -0.5, 0.5, -0.5])
    assert np.allclose(result["protected_after"], [2.0, 2.0, 2.0], atol=1e-12)
    assert abs(result["new_response_after_20_updates"] - 3.0) < 2e-6
    assert result["incompatible_control_safe_direction_norm"] < 1e-12


def test_gate5_replay_discovers_compatible_direction_without_matrix_P():
    result = gate5_replay_driven_coordination()
    assert result["pass"]
    learned = result["learned_compatible_change"]
    assert learned["protected_replays"] == 60
    assert learned["proposal_reference_cosine"] > 0.999999
    assert learned["max_protected_drift"] < 1e-6
    assert abs(learned["new_response_after_commit"] - 3.0) < 1e-6
    assert learned["replay_memory_scalar_values"] == 15
    assert learned["temporary_proposal_scalar_values"] == 4


def test_gate5_exposes_missing_replay_and_full_rank_conflict():
    result = gate5_replay_driven_coordination()
    assert result["full_rank_incompatibility"]["proposal_norm_after_full_rank_replay"] < 1e-10
    assert abs(result["omitted_replay_attacker"]["untested_response_drift"]) > 0.4
    assert result["direct_write_then_repair_attacker"]["max_transient_protected_drift"] > 0.49


def test_replay_bank_capacity_is_explicit():
    learner = IttnasNoruen(
        [0.0, 0.0],
        material_feedback=1.0,
        readout_weights=[1.0, 1.0],
    )
    coordinator = ReplayPlasticityCoordinator(learner, max_protected=1)
    coordinator.add_protected("old", [1.0, 0.0], 1.0)
    assert coordinator.bank.stored_scalar_values == 3
    with pytest.raises(OverflowError):
        coordinator.add_protected("too-many", [0.0, 1.0], 1.0)


def test_delay_stability_reference_values():
    assert math.isclose(delayed_error_stability_limit(0), 2.0, rel_tol=1e-12)
    assert math.isclose(delayed_error_stability_limit(1), 1.0, rel_tol=1e-12)
    assert math.isclose(
        delayed_error_stability_limit(3),
        2.0 * math.sin(math.pi / 14.0),
        rel_tol=1e-12,
    )


def test_no_consequence_means_no_material_update_even_with_residual():
    learner = IttnasNoruen([1.0, 0.5], echo_enabled=True)
    before = learner.material.copy()
    observation = learner.observe(1.0, external=[0.0, 1.0], learn_echo=False)
    assert np.linalg.norm(observation.residual) > 0.0
    assert np.array_equal(learner.material, before)


def test_consequence_uses_surviving_eligibility_not_event_label():
    learner = IttnasNoruen(
        [0.0, 0.0, 0.0],
        echo_enabled=True,
        eligibility_decay=0.5,
        material_feedback=0.0,
    )
    learner.observe(0.0, external=[0.0, 2.0, 0.0], learn_echo=False)
    learner.quiet(2)
    update = learner.apply_consequence(+1.0)

    # 2.0 event * 0.5^2 eligibility decay * 0.1 memory learning rate.
    assert np.allclose(update, [0.0, 0.05, 0.0])


def test_material_write_changes_future_physical_response():
    learner = IttnasNoruen([0.0, 0.0], material_feedback=1.0)
    probe = [1.0, 0.0]
    before = learner.soma_response(0.0, probe)

    learner.begin_receipt("trial")
    learner.observe(0.0, probe, learn_echo=False)
    learner.seal_receipt("trial")
    learner.apply_receipt_consequence("trial", +1.0)

    after = learner.soma_response(0.0, probe)
    assert after > before


def test_two_pending_receipts_survive_separate_consequences():
    learner = IttnasNoruen([0.0, 0.0], material_feedback=0.0)

    learner.begin_receipt("A")
    learner.observe(0.0, [1.0, 0.0], learn_echo=False)
    learner.seal_receipt("A")

    learner.begin_receipt("B")
    learner.observe(0.0, [0.0, 1.0], learn_echo=False)
    learner.seal_receipt("B")

    assert learner.pending_receipt_ids == ("A", "B")
    update_a = learner.apply_receipt_consequence("A", +1.0)
    assert learner.pending_receipt_ids == ("B",)
    update_b = learner.apply_receipt_consequence("B", +1.0)

    assert update_a[0] > 0.0 and update_a[1] == 0.0
    assert update_b[1] > 0.0 and update_b[0] == 0.0


def test_receipt_capacity_is_explicit_and_reusable():
    learner = IttnasNoruen(
        [0.0, 0.0, 0.0, 0.0],
        material_feedback=0.0,
        max_receipts=2,
    )
    assert learner.receipt_trace_value_capacity == 8

    learner.begin_receipt("A")
    learner.seal_receipt("A")
    learner.begin_receipt("B")
    learner.seal_receipt("B")
    assert learner.receipt_count == 2

    with pytest.raises(OverflowError):
        learner.begin_receipt("C")

    learner.apply_receipt_consequence("A", +1.0)
    assert learner.receipt_count == 1
    learner.begin_receipt("C")
    assert learner.receipt_count == 2
