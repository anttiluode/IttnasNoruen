import numpy as np

from gate9_selected_replay import paired_summary, reached, run_method, select_indices


def make_records():
    return [dict(image=np.zeros(64),label=label,view=view,
                 initial_margin=.02 if i==0 else .6,bound=.01 if i==0 else .1)
            for label in range(10) for view in ("full","upper","lower") for i in range(2)]


def test_proposed_damage_can_select_a_previously_confident_cue():
    pool = make_records()
    proposed = np.array([r["initial_margin"] for r in pool])
    proposed[5] = -.8
    chosen = select_indices("interference_scanned",pool,np.random.default_rng(1),proposed)
    boundary = select_indices("boundary_scanned",pool,np.random.default_rng(1),proposed)
    assert 5 in chosen and 5 not in boundary
    assert len(chosen)==20 and len(set(chosen))==20
    assert all(sum(pool[i]["label"]==label for i in chosen)==2 for label in range(10))


def test_missing_progress_does_not_become_a_preservation_success():
    access = {f:dict(loss_rate_among_initially_correct=.02) for f in ("seen","unseen")}
    checkpoint = dict(requested_progress=.1,access=access,calls=100)
    def specimen(seed,active):
        return dict(seed=seed,runs=[dict(method=m,checkpoints=[checkpoint] if
                    m!="interference_scanned" or active else []) for m in
                    ("interference_scanned","unprotected","fixed_mixed","random_scanned","boundary_scanned","all_pool")])
    rows = paired_summary([specimen(1,True),specimen(2,False)],[.1,.2])
    assert all(r["coverage"]==1 for r in rows if r["requested_progress"]==.1)
    assert all(r["unseen_loss_difference"] is None for r in rows if r["requested_progress"]==.2)
    assert not reached(.08,.1)


def test_exhausted_selection_scan_never_changes_material():
    class World:
        def encode(self,x):
            return x
        def encoded_probabilities(self,theta,x):
            return np.full((len(x),10),.1)
    result = run_method(World(),make_records(),np.zeros((2,64)),np.zeros(2,dtype=int),
                        lambda theta:{"norm":float(np.linalg.norm(theta))},
                        "interference_scanned",151,attempts=2,budget=1)
    assert result["calls"]==1
    assert result["parameter_norm"]==0.
    assert result["checkpoints"]==[]
    assert "budget" in result["stop_reason"]
