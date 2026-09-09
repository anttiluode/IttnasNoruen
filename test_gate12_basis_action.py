import numpy as np
import pytest

from gate12_basis_action import (
    boundary_indices,
    response_tolerance,
    select_frozen_basis,
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


def test_boundary_basis_is_two_per_class_and_frozen_by_margin():
    pool = _pool()
    for label in range(10):
        pool[label * 6 + 4]["initial_margin"] = 0.01 + label * 1e-4
        pool[label * 6 + 5]["initial_margin"] = 0.02 + label * 1e-4
    selected = boundary_indices(pool)
    assert len(selected) == 20
    for label in range(10):
        assert selected[2 * label:2 * label + 2] == [label * 6 + 4, label * 6 + 5]


def test_qspan_basis_keeps_two_per_class():
    pool = _pool()
    rng = np.random.default_rng(12)
    signatures = rng.normal(size=(60, 5))
    selected = select_frozen_basis("qspan_equal", pool, signatures)
    assert len(selected) == 20
    assert len(set(selected)) == 20
    for label in range(10):
        assert sum(pool[i]["label"] == label for i in selected) == 2


def test_response_tolerance_floor_and_relative_branch():
    assert response_tolerance(0.01, rel=0.05, floor=0.002) == 0.002
    assert np.isclose(response_tolerance(0.2, rel=0.05, floor=0.002), 0.01)
    with pytest.raises(ValueError):
        response_tolerance(0.1, rel=-0.1)


def test_unknown_basis_method_rejected():
    with pytest.raises(ValueError):
        select_frozen_basis("nope", _pool(), np.ones((60, 3)))
