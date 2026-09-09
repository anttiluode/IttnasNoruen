from response_geometry_reference import run_reference


def test_anchor_agreement_does_not_protect_nearby_responses():
    result = run_reference()
    assert result["pass"]
    for outcome in result["outcomes"].values():
        assert outcome["scalar_calls"] <= outcome["scalar_budget"]
