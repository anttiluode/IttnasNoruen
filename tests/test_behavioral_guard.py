import numpy as np
import pytest

from behavioral_guard import BehavioralUpdateGuard, ResponseMeter, orthogonal_replay_projection
from gate6_continuous_learning import run_stream, StreamConfig


def test_only_measured_callback_is_needed_for_overlapping_change():
    # P exists inside the world/evaluator. It is never a guard argument.
    matrix=np.array([[1.,1.,0.,0.],[0.,1.,1.,0.],[0.,0.,1.,1.],[1.,0.,1.,0.]])
    meter=ResponseMeter(lambda g,q:float(matrix[q]@g),max_calls=400)
    guard=BehavioralUpdateGuard(3,trust_radius=2.)
    original=np.ones(4)
    for q in range(3):
        guard.remember(q,meter(original,q),1e-8)
    result=guard.step(original,3,3.,meter)
    assert result.status=="accepted"
    assert np.allclose(matrix[:3]@result.parameters,2.,atol=1e-8)
    assert abs(matrix[3]@result.parameters-3.)<1e-8
    assert np.array_equal(original,np.ones(4))


def test_curvature_can_make_zero_first_order_harm_unsafe():
    # At zero the old answer has zero gradient, but every nonzero change harms it.
    def response(theta,q):
        return float(theta@theta) if q=="old" else float(theta[0])
    guard=BehavioralUpdateGuard(1,trust_radius=.2,max_backtracks=5)
    guard.remember("old",0.,0.)
    original=np.zeros(2)
    result=guard.step(original,"new",1.,ResponseMeter(response,100))
    assert result.status=="no_acceptable_step_observed"
    assert result.rejected_candidates==6
    assert np.array_equal(result.parameters,original)


def test_budget_exhaustion_cannot_partially_commit():
    guard=BehavioralUpdateGuard(1)
    guard.remember("old",0.,.01)
    meter=ResponseMeter(lambda x,q:float(x.sum()),max_calls=3)
    original=np.zeros(4)
    result=guard.step(original,"new",1.,meter)
    assert result.status=="measurement_budget_exhausted"
    assert meter.calls==3
    assert np.array_equal(result.parameters,original)


def test_reference_memory_is_bounded():
    guard=BehavioralUpdateGuard(1)
    guard.remember(0,1.,.01)
    with pytest.raises(OverflowError):
        guard.remember(1,1.,.01)


def test_full_rank_measured_constraints_block_a_new_answer():
    def response(x,q):
        return float(x[int(q)]) if q!="new" else float(x.sum())
    guard=BehavioralUpdateGuard(3)
    for i in range(3):
        guard.remember(i,1.,0.)
    result=guard.step(np.ones(3),"new",4.,ResponseMeter(response,200))
    assert result.status in ("no_feasible_step_observed", "no_acceptable_step_observed")
    assert np.array_equal(result.parameters,np.ones(3))


def test_measured_sketch_handles_nearly_parallel_replays():
    rows=np.array([[1.,0.,0.],[1.,1e-5,0.]])
    proposal=orthogonal_replay_projection(np.ones(3),rows,np.zeros(2))
    assert np.allclose(proposal,[0.,0.,1.],atol=1e-10)


def test_continuous_controls_get_same_actions_and_bounded_history():
    config=StreamConfig(blocks=80,reverse_at=40,receipt_capacity=2)
    protected=run_stream(101,"proposal",config=config)
    naive=run_stream(101,"naive",config=config)
    assert protected["counts"]["physical_ticks"]==400
    assert protected["counts"]["nonzero_command_queries"]==80
    assert protected["counts"]["receipt_overflows"]>0
    assert protected["pending_peak"]<=2
    assert protected["probe_energy"]==naive["probe_energy"]
    assert protected["command_energy"]==naive["command_energy"]


def test_dither_identifiability_attacker_has_exact_contamination():
    # This measures causal identifiability with fixed material, separate from task score.
    signs=np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
    eps=.2
    self_signature=np.array([.9,.45,-.35,.7])
    outside=np.array([.3,-.2,.15,.25])
    base=np.ones(4)
    for kappa in (0.,.04):
        returns=np.array([(a+eps*d)*self_signature+base+.2*a*outside+kappa*d*outside for a,d in signs])
        estimated=signs[:,1]@returns/(4*eps)
        assert np.allclose(estimated,self_signature+(kappa/eps)*outside)
        if kappa:
            assert np.linalg.norm(estimated-self_signature)>.08
