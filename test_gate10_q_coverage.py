import numpy as np

from gate10_q_coverage import (
    _qspan_indices,
    probe_directions,
    signature_geometry,
)


def _pool():
    records = []
    for label in range(10):
        for j in range(6):
            records.append({
                "label": label,
                "view": ("full", "upper", "lower")[j % 3],
                "initial_margin": 0.2 + 0.01 * j,
                "bound": 0.1,
            })
    return records


def test_probe_directions_are_orthonormal_and_deterministic():
    a = probe_directions(211, 24, 12)
    b = probe_directions(211, 24, 12)
    assert np.allclose(a, b)
    assert np.allclose(a @ a.T, np.eye(12), atol=1e-12)


def test_qspan_selector_keeps_two_per_class():
    pool = _pool()
    rng = np.random.default_rng(5)
    signatures = rng.normal(size=(60, 8))
    selected = _qspan_indices(pool, signatures)
    assert len(selected) == 20
    assert len(set(selected)) == 20
    for label in range(10):
        assert sum(pool[i]["label"] == label for i in selected) == 2


def test_qspan_recovers_known_signature_span():
    pool = _pool()
    # Every class offers several mixtures of four true directions. The global
    # selector should find a bank spanning all four rather than duplicate one.
    base = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1., 1., 0., 0.],
        [0., 0., 1., 1.],
    ])
    signatures = np.vstack([base + 1e-4 * label for label in range(10)])
    selected = _qspan_indices(pool, signatures)
    geometry = signature_geometry(signatures, selected)
    assert geometry["rank"] == 4
    assert geometry["ambient_rank"] == 4
    assert geometry["residual_fraction"] < 1e-10
