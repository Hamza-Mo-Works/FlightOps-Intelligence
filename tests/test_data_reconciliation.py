"""Read existing Silver/Gold tables and reconcile additive flight counts."""
import os
import pytest

pytestmark = pytest.mark.databricks


@pytest.fixture(scope="module")
def live_spark():
    if os.getenv("FLIGHTOPS_TEST_LIVE_TABLES") != "1":
        pytest.skip("Set FLIGHTOPS_TEST_LIVE_TABLES=1 to read Unity Catalog tables")
    from pyspark.sql import SparkSession
    active = SparkSession.getActiveSession()
    if active is None:
        pytest.fail("Live checks require an active Databricks/Connect Spark session")
    return active


@pytest.mark.parametrize("gold_table,total_column", [
    ("daily_flight_performance", "total_flights"),
    ("airport_congestion_metrics", "total_departures"),
    ("carrier_performance", "total_flights"),
])
def test_gold_totals_match_silver(live_spark, gold_table, total_column):
    from pyspark.sql import functions as F
    silver = live_spark.table("flightops_intelligence.silver.bts_flights_clean")
    gold = live_spark.table("flightops_intelligence.gold." + gold_table)
    assert silver.limit(1).count() == 1
    assert gold.agg(F.sum(total_column)).first()[0] == silver.count()


@pytest.mark.parametrize("table,keys", [
    ("daily_flight_performance", ["flight_date"]),
    ("airport_congestion_metrics", ["flight_date", "origin_airport"]),
    ("carrier_performance", ["flight_date", "airline"]),
])
def test_gold_grain_is_unique(live_spark, table, keys):
    from pyspark.sql import functions as F
    frame = live_spark.table("flightops_intelligence.gold." + table)
    assert frame.groupBy(*keys).count().filter(F.col("count") > 1).limit(1).count() == 0
