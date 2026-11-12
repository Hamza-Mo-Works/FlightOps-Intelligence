"""Check request date bounds, entity allowlists and parameterized filters."""
import pytest
from _flightops_support import standard_plan


@pytest.mark.parametrize("changes", [
    {"start_date": "2024-12-31"}, {"end_date": "2026-01-01"},
    {"start_date": "2025-12-31", "end_date": "2025-01-01"},
    {"start_date": "not-a-date"}, {"airlines": ["AA'); DROP TABLE x;--"]},
    {"airlines": ["ZZ"]}, {"airports": ["ZZZ"]},
])
def test_invalid_filters_are_rejected(backend, changes):
    with pytest.raises(backend["AnalysisRejected"]):
        backend["validate_plan"](standard_plan(**changes), "OPS")


@pytest.mark.parametrize("domain,changes", [
    ("OPS", {"airlines": ["AA"]}), ("AIRPORT", {"airlines": ["AA"]}),
    ("CARRIER", {"airports": ["JFK"]}),
])
def test_filters_must_match_the_domain(backend, domain, changes):
    with pytest.raises(backend["AnalysisRejected"]):
        backend["validate_plan"](standard_plan(domain, **changes), domain)


def test_carrier_filter_is_bound_as_a_parameter(backend):
    calls = []
    def execute(sql, params, bucket):
        calls.append((sql, params))
        return [{"m009": 2.0, "m009_observations": 100}]
    backend["_cached_aggregate"] = execute
    backend["execute_analysis"](standard_plan("CARRIER", airlines=["AA"]), "CARRIER")
    statement, params = calls[0]
    assert "airline IN (?)" in statement
    assert "'AA'" not in statement
    assert params == ("2025-01-01", "2025-12-31", "AA")
