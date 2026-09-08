import numpy as np

from gates import (
    gate0_efference_copy_credit,
    gate1_surprise_is_not_memory,
    gate2_reversal_with_preservation,
)
from ittnas_noruen import IttnasNoruen


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


def test_no_consequence_means_no_material_update_even_with_residual():
    learner = IttnasNoruen([1.0, 0.5], echo_enabled=True)
    before = learner.material.copy()
    observation = learner.observe(1.0, external=[0.0, 1.0], learn_echo=False)
    assert np.linalg.norm(observation.residual) > 0.0
    assert np.array_equal(learner.material, before)


def test_consequence_uses_surviving_eligibility_not_event_label():
    learner = IttnasNoruen([0.0, 0.0, 0.0], echo_enabled=True, eligibility_decay=0.5)
    learner.observe(0.0, external=[0.0, 2.0, 0.0], learn_echo=False)
    learner.quiet(2)
    update = learner.apply_consequence(+1.0)

    # 2.0 event * 0.5^2 eligibility decay * 0.1 memory learning rate.
    assert np.allclose(update, [0.0, 0.05, 0.0])
