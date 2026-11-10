"""Validate metadata records declared by the actual semantic-layer notebook."""
from _flightops_support import literal_value, source_path, semantic_records


def test_metric_ids_rules_and_denominators():
    metrics = semantic_records()
    assert set(metrics) == {f"M{index:03}" for index in range(1, 11)}
    for metric in metrics.values():
        assert metric["aggregation_rule"] in {"SUM", "RATIO_OF_SUMS"}
        assert metric["unit"] in {"flights", "minutes", "percent"}
        assert bool(metric["denominator_column"]) == (metric["aggregation_rule"] == "RATIO_OF_SUMS")


def test_domain_mappings_have_no_orphans_or_duplicates():
    path = source_path("05_Business_Semantic_Layer.py")
    domains = {row[0] for row in literal_value(path, "business_domains")}
    mappings = literal_value(path, "domain_metric_mappings")
    assert domains == {"OPS", "AIRPORT", "CARRIER"}
    assert len(mappings) == len({(domain, metric) for domain, metric, _ in mappings})
    assert all(domain in domains and metric in semantic_records() and priority > 0
               for domain, metric, priority in mappings)


def test_ops_and_carrier_on_time_denominators_are_distinct():
    metrics = semantic_records()
    assert metrics["M003"]["denominator_column"] == "completed_flights"
    assert metrics["M010"]["denominator_column"] == "arrival_delay_observations"
