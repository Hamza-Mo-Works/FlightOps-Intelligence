"""Execute the notebook context resolver with declared semantic metadata."""
import pytest
from _flightops_support import declarations, source_path, literal_value, semantic_records


def test_context_contains_only_the_selected_domain():
    ns = declarations(source_path("07_FlightOps_AI_Context_Layer.py"))
    semantic = source_path("05_Business_Semantic_Layer.py")
    ns["domains"] = {row[0]: {"domain_name": row[1]} for row in literal_value(semantic, "business_domains")}
    ns["metrics"] = [{**semantic_records()[metric], "domain_id": domain,
                     "supported_dimensions": ["flight_date"]}
                    for domain, metric, _ in literal_value(semantic, "domain_metric_mappings")]
    ns["questions"] = [{"domain_id": "OPS", "example_question": "How many flights?"}]
    result = ns["get_business_context"]("OPS")
    assert result["domain_id"] == "OPS"
    assert {row["metric_id"] for row in result["approved_metrics"]} == {"M001", "M002", "M003", "M004"}
    assert result["example_questions"] == ["How many flights?"]
    with pytest.raises(ValueError):
        ns["get_business_context"]("UNKNOWN")
