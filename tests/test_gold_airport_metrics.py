"""Check airport grain, observation counts and weighted rollup components."""
import pytest
pytestmark = pytest.mark.spark


def test_airport_grain_and_null_observations(bronze, transform, aggregate):
    rows = aggregate(transform(bronze({}, {"Origin": "ATL", "TaxiOut": None})), "df_airport").collect()
    assert len(rows) == 2
    atl = next(row for row in rows if row.origin_airport == "ATL")
    assert atl.taxi_out_observations == 0 and atl.avg_taxi_out_minutes is None


def test_weighted_taxi_out_components(bronze, transform, aggregate):
    inputs = [{"Flight_Number_Reporting_Airline": str(n), "TaxiOut": "10"} for n in range(9)]
    inputs.append({"Flight_Number_Reporting_Airline": "99", "Origin": "ATL", "TaxiOut": "100"})
    rows = aggregate(transform(bronze(*inputs)), "df_airport").collect()
    weighted = sum(row.total_taxi_out_minutes for row in rows) / sum(row.taxi_out_observations for row in rows)
    assert weighted == 19.0
    assert sum(row.avg_taxi_out_minutes for row in rows) / len(rows) == 55.0
