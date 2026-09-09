import numpy as np

from gate14_local_law import random_stratified_basis, select_basis


def pool():
    return [{"label": label, "initial_margin": 0.2 + .01*j, "bound": .1}
            for label in range(10) for j in range(6)]


def test_random_basis_is_two_per_class_and_deterministic():
    p = pool()
    a = random_stratified_basis(p, 389, 3)
    b = random_stratified_basis(p, 389, 3)
    assert a == b
    assert len(a) == 20 and len(set(a)) == 20
    for label in range(10):
        assert sum(p[i]["label"] == label for i in a) == 2


def test_random_basis_rolls_by_attempt():
    p = pool()
    assert random_stratified_basis(p, 389, 0) != random_stratified_basis(p, 389, 1)


def test_frozen_selector_reuses_identity_basis():
    p = pool()
    frozen = [6*label + j for label in range(10) for j in (0, 1)]
    sig = np.arange(60*4, dtype=float).reshape(60, 4)
    assert select_basis("frozen_qspan_rolling_anchor", p, sig, frozen, 389, 8) == frozen


def test_qspan_selector_is_stratified():
    p = pool()
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(60, 6))
    frozen = [6*label + j for label in range(10) for j in (0, 1)]
    selected = select_basis("qspan_rolling_anchor", p, sig, frozen, 389, 2)
    assert len(selected) == 20
    for label in range(10):
        assert sum(p[i]["label"] == label for i in selected) == 2
