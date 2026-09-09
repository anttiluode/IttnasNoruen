import numpy as np

from gate13_dynamic_q import conditioned_directions, generic_directions, _jaccard, _two_per_class


def make_pool():
    return [{"label": label, "initial_margin": 0.2 + 0.01*j, "bound": 0.1}
            for label in range(10) for j in range(6)]


def test_conditioned_panel_is_orthonormal_and_aligned():
    step = np.arange(1.0, 25.0)
    panel = conditioned_directions(step, 347, 2, 12)
    unit = step / np.linalg.norm(step)
    assert panel.shape == (12, 24)
    assert np.allclose(panel[0], unit, atol=1e-12)
    assert np.allclose(panel @ panel.T, np.eye(12), atol=1e-10)


def test_conditioned_panel_is_deterministic():
    step = np.linspace(-1.0, 2.0, 24)
    assert np.allclose(conditioned_directions(step, 359, 4, 12),
                       conditioned_directions(step, 359, 4, 12))


def test_generic_panel_changes_by_attempt():
    a = generic_directions(347, 0, 24, 12)
    b = generic_directions(347, 1, 24, 12)
    assert np.allclose(a @ a.T, np.eye(12), atol=1e-12)
    assert np.allclose(b @ b.T, np.eye(12), atol=1e-12)
    assert not np.allclose(a, b)


def test_two_per_class_contract():
    pool = make_pool()
    selected = [6*label + j for label in range(10) for j in (0, 1)]
    assert _two_per_class(pool, selected)


def test_jaccard_overlap():
    assert _jaccard([1, 2], [1, 2]) == 1.0
    assert np.isclose(_jaccard([1, 2], [2, 3]), 1/3)
