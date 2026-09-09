import numpy as np

from behavioral_guard import BehavioralUpdateGuard,ResponseMeter


def response(theta,q):
    a,b=theta
    return float(b-a*a) if q=="old" else float(a)


def test_curved_boundary_needs_measured_second_order_compensation():
    before=np.zeros(2)
    old=BehavioralUpdateGuard(1,trust_radius=.5,max_model_corrections=0)
    old.remember_range("old",minimum=0.)
    failed=old.step(before,"new",.1,ResponseMeter(response,100))
    assert failed.status=="no_acceptable_step_observed"
    guard=BehavioralUpdateGuard(1,trust_radius=.5)
    guard.remember_range("old",minimum=0.)
    meter=ResponseMeter(response,100)
    corrected=guard.step(before,"new",.1,meter)
    assert corrected.status=="accepted"
    assert abs(response(corrected.parameters,"new")-.1)<1e-8
    assert response(corrected.parameters,"old")>=-1e-12
    assert corrected.calls==meter.calls
    assert np.array_equal(before,np.zeros(2))


def test_budget_exhaustion_during_correction_cannot_commit_partial_state():
    guard=BehavioralUpdateGuard(1,trust_radius=.5)
    guard.remember_range("old",minimum=0.)
    # 2 current + 8 derivative + 2 initial candidate calls, then only one remains.
    meter=ResponseMeter(response,13)
    result=guard.step(np.zeros(2),"new",.1,meter)
    assert result.status=="measurement_budget_exhausted"
    assert np.array_equal(result.parameters,np.zeros(2))
    assert meter.calls==13
