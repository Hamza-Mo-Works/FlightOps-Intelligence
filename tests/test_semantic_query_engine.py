"""Execute generated metric SQL and exercise metadata identifier validation."""
import sqlite3
import pytest
from _flightops_support import app_path, declarations, semantic_records


@pytest.mark.parametrize("metric,rows,expected", [
    ("M001", [(10, 0), (30, 0)], 40),
    ("M002", [(1, 10), (2, 90)], 3.0),
    ("M007", [(90, 9), (100, 1)], 19.0),
])
def test_sql_uses_sums_and_correct_units(backend, metric, rows, expected):
    spec = semantic_records()[metric]
    numerator = spec["numerator_column"]
    denominator = spec["denominator_column"] or "unused_observations"
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"CREATE TABLE metrics ({numerator} REAL, {denominator} REAL)")
        connection.executemany("INSERT INTO metrics VALUES (?, ?)", rows)
        expression, _ = backend["metric_expr"](spec)
        assert connection.execute(f"SELECT {expression} FROM metrics").fetchone()[0] == expected
    finally:
        connection.close()


def test_zero_denominator_returns_null(backend):
    expression, _ = backend["metric_expr"](semantic_records()["M007"])
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE metrics (total_taxi_out_minutes REAL, taxi_out_observations REAL)")
        connection.execute("INSERT INTO metrics VALUES (0, 0)")
        assert connection.execute(f"SELECT {expression} FROM metrics").fetchone()[0] is None


def test_metadata_cannot_supply_sql_identifiers():
    ns = declarations(app_path("flightops_backend.py"))
    records = []
    for domain, identifiers in ns["ALLOWED_METRIC_IDS"].items():
        records.extend({**semantic_records()[identifier], "domain_id": domain} for identifier in identifiers)
    ns["run_sql"] = lambda *args: records
    assert len(ns["load_semantic_metrics"]()) == 10
    records[0]["numerator_column"] = "total_flights); DROP TABLE flights; --"
    with pytest.raises(RuntimeError, match="numerator"):
        ns["load_semantic_metrics"]()
