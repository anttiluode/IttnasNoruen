import numpy as np
import pytest

from behavioral_guard import MeasurementBudgetExceeded
from gate11_operator_guard import (
    CostedResponseMeter,
    fingerprint_tolerance,
    fragile_class_anchors,
    frozen_operator_modes,
)


def _pool():
    records = []
    for label in range(10):
        for j in range(6):
            records.append({
                "label": label,
                "initial_margin": 0.3 + 0.01 * j,
                "bound": 0.1,
            })
    return records


def test_costed_meter_charges_composite_query_before_evaluation():
    seen = []

    def response(theta, query):
        seen.append(query)
        return float(theta[0])

    meter = CostedResponseMeter(
        response, 4,
        lambda q: 2 if isinstance(q, tuple) and q[0] == "gain" else 1,
    )
    assert meter(np.array([1.0]), 0) == 1.0
    assert meter.calls == 1
    assert meter(np.array([2.0]), ("gain", 0)) == 2.0
    assert meter.calls == 3
    with pytest.raises(MeasurementBudgetExceeded):
        meter(np.array([3.0]), ("gain", 1))
    assert meter.calls == 3
    assert seen == [0, ("gain", 0)]


def test_frozen_operator_modes_recover_rank_and_distinct_pivots():
    rng = np.random.default_rng(7)
    left = rng.normal(size=(20, 3))
    right = rng.normal(size=(3, 5))
    signatures = left @ right
    directions = np.eye(5)
    modes = frozen_operator_modes(signatures, directions)
    assert len(modes) == 3
    assert len({m["pivot"] for m in modes}) == 3
    for mode in modes:
        assert mode["direction"].shape == (5,)
        assert np.isclose(np.linalg.norm(mode["direction"]), 1.0)
        assert mode["singular"] > 0
        assert mode["leverage"] > 0


def test_frozen_operator_modes_are_deterministic():
    signatures = np.array([
        [3.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0],
        [0.5, -0.2],
    ])
    directions = np.eye(2)
    a = frozen_operator_modes(signatures, directions)
    b = frozen_operator_modes(signatures, directions)
    assert [m["pivot"] for m in a] == [m["pivot"] for m in b]
    for x, y in zip(a, b):
        assert np.allclose(x["direction"], y["direction"])
        assert x["singular"] == y["singular"]


def test_fragile_class_anchors_choose_lowest_margin_per_class():
    pool = _pool()
    # Make a non-first cue uniquely fragile in every class.
    for label in range(10):
        pool[label * 6 + 4]["initial_margin"] = 0.01 + label * 1e-4
    anchors = fragile_class_anchors(pool)
    assert anchors == [label * 6 + 4 for label in range(10)]


def test_fingerprint_tolerance_has_absolute_floor_and_relative_branch():
    assert fingerprint_tolerance(1e-4, rel=0.15, floor=5e-4) == 5e-4
    assert np.isclose(fingerprint_tolerance(0.02, rel=0.15, floor=5e-4), 0.003)
    with pytest.raises(ValueError):
        fingerprint_tolerance(0.1, rel=-0.1)
