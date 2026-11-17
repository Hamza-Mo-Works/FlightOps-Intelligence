"""Compare original Gold and backfill aggregation expressions on identical rows."""
import pytest
pytestmark = pytest.mark.spark


@pytest.mark.parametrize("variable", ["df_daily", "df_airport", "df_carrier"])
def test_original_and_backfill_metrics_match(bronze, transform, aggregate, variable):
    silver = transform(bronze({}, {"Origin": "ATL", "ArrDelay": "30"}, {"Reporting_Airline": "DL", "Cancelled": "1"}))
    original = aggregate(silver, variable).toJSON().collect()
    backfill = aggregate(silver, variable, backfill=True).toJSON().collect()
    assert sorted(original) == sorted(backfill)
