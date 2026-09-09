import numpy as np
import pytest

from behavioral_guard import BehavioralUpdateGuard, ResponseMeter, project_response_bounds
from gate8_retrieval_access import CueRecognizer, retrieval_policy, tolerance_counterexample, view


def test_full_rank_tolerances_are_usable_learning_capacity():
    result = tolerance_counterexample()
    assert result["equalities"]["status"] == "no_feasible_step_observed"
    assert result["bounds"]["status"] == "accepted"
    assert abs(result["bounds"]["new_answer"]-2.1) < 1e-8
    assert result["bounds"]["largest_old_drift"] <= .1


def test_improving_old_margin_does_not_require_unchanged_output():
    guard = BehavioralUpdateGuard(1, trust_radius=2.)
    guard.remember_range("old", minimum=.5)
    result = guard.step(np.ones(1), "new", 2.,
                        ResponseMeter(lambda theta,q:float(theta[0]), 100))
    assert result.status == "accepted"
    assert np.allclose(result.parameters, [2.])


def test_margin_boundary_and_trust_region_limit_damaging_update():
    rows = np.eye(2)
    result = project_response_bounds([-1.,1.], rows, [-.1,-.2], [.2,.2], .15)
    assert np.all(result >= [-.1-1e-10,-.2-1e-10])
    assert np.all(result <= [.2+1e-10,.2+1e-10])
    assert np.linalg.norm(result) <= .15+1e-12


def test_interval_memory_is_bounded_and_rejects_invalid_limits():
    guard = BehavioralUpdateGuard(1)
    with pytest.raises(ValueError):
        guard.remember_range("invalid", minimum=2., maximum=1.)
    guard.remember_range("q", minimum=0.)
    with pytest.raises(OverflowError):
        guard.remember_range("other", minimum=0.)


def test_growth_starts_without_changing_the_existing_function():
    class Model:
        coefs_ = [np.ones((64,3))*.01, np.arange(30).reshape(3,10)*.01]
        intercepts_ = [np.zeros(3), np.zeros(10)]
    world = CueRecognizer(Model(), seed=10)
    x = np.arange(128).reshape(2,64)/128
    before = world.probabilities(np.zeros(3), x)
    world.allocate_paths()
    after = world.probabilities(np.zeros(43), x)
    assert np.array_equal(before, after)
    changed = np.zeros(43)
    changed[3] = .5
    assert not np.allclose(world.probabilities(changed,x),before)


def test_second_cue_policy_uses_confidence_not_correct_labels():
    first = np.array([[.51,.49],[.95,.05]])
    second = np.array([[.01,.99],[.01,.99]])
    predictions, asked = retrieval_policy(first, second)
    assert np.array_equal(asked,[True,False])
    assert np.array_equal(predictions,[1,0])


def test_partial_cues_really_remove_the_other_pixels():
    x = np.arange(64)[None,:]
    assert np.array_equal(view(x,"upper")+view(x,"lower"),x)
    assert np.all(view(x,"upper")[0,32:]==0)
    assert np.all(view(x,"lower")[0,:32]==0)
