"""Provide isolated source namespaces and optional Spark sessions for the test suite."""
import os
from datetime import datetime
import pytest
from _flightops_support import (assignments, backend_namespace, predictor_namespace,
                                weather_namespace, source_path)


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name == "test_bronze_ingestion.py":
            item.add_marker(pytest.mark.databricks)


def pytest_runtest_setup(item):
    if item.get_closest_marker("databricks") and os.getenv("FLIGHTOPS_TEST_LIVE_TABLES") != "1":
        pytest.skip("Set FLIGHTOPS_TEST_LIVE_TABLES=1 in an active Databricks session")


@pytest.fixture
def backend():
    return backend_namespace()


@pytest.fixture
def predictor():
    return predictor_namespace()


@pytest.fixture
def weather():
    return weather_namespace()


@pytest.fixture(scope="session")
def spark():
    SparkSession = pytest.importorskip("pyspark.sql").SparkSession
    active = SparkSession.getActiveSession()
    created = False
    if active is None:
        if os.getenv("FLIGHTOPS_TEST_ALLOW_LOCAL_SPARK") != "1":
            pytest.skip("Set FLIGHTOPS_TEST_ALLOW_LOCAL_SPARK=1 or run inside an active Databricks session")
        active = (SparkSession.builder.master("local[2]").appName("FlightOpsSourceTests")
                  .config("spark.ui.enabled", "false").config("spark.driver.host", "127.0.0.1")
                  .config("spark.sql.shuffle.partitions", "2").getOrCreate())
        created = True
    isolated = active.newSession()
    isolated.conf.set("spark.sql.ansi.enabled", "false")
    isolated.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")
    isolated.conf.set("spark.sql.session.timeZone", "UTC")
    yield isolated
    if created:
        active.stop()


@pytest.fixture
def bronze(spark):
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType
    string_columns = ("FlightDate", "Year", "Month", "DayOfWeek", "Reporting_Airline",
        "Tail_Number", "Flight_Number_Reporting_Airline", "Origin", "Dest", "OriginAirportID",
        "DestAirportID", "CRSDepTime", "CRSArrTime", "DepTime", "ArrTime", "DepDelay", "ArrDelay",
        "TaxiOut", "TaxiIn", "AirTime", "Distance", "Cancelled", "Diverted", "CancellationCode",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
        "CRSElapsedTime", "_source_file", "_source_system")
    defaults = dict.fromkeys(string_columns)
    defaults.update(FlightDate="7/23/2025", Year="2025", Month="7", DayOfWeek="3",
        Reporting_Airline="AA", Tail_Number="N123AA", Flight_Number_Reporting_Airline="100",
        Origin="JFK", Dest="LAX", OriginAirportID="12478", DestAirportID="12892",
        CRSDepTime="0900", CRSArrTime="1200", DepTime="0910", ArrTime="1205",
        DepDelay="10", ArrDelay="5", TaxiOut="15", TaxiIn="5", AirTime="300",
        Distance="2475", Cancelled="0", Diverted="0", CRSElapsedTime="360",
        _source_file="bts_2025_07.csv", _source_system="BTS",
        _ingestion_timestamp=datetime(2026, 10, 20, 10))
    schema = StructType([StructField(name, StringType(), True) for name in string_columns]
                        + [StructField("_ingestion_timestamp", TimestampType(), True)])
    def make(*changes):
        return spark.createDataFrame([{**defaults, **change} for change in changes or ({},)], schema)
    return make


@pytest.fixture
def transform():
    def run(frame, backfill=False, deduplicate=False):
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        path = source_path("10_FlightOps_2025_Silver_Backfill.py" if backfill else "03_BTS_Silver_Transformation.py")
        names = ["df_typed", "df_validated"]
        ns = {"df_bronze": frame, "F": F, "Window": Window}
        assignments(path, names, ns)
        if deduplicate:
            ns["df_valid"] = ns["df_validated"].filter(F.col("is_valid"))
            assignments(path, ["flight_key", "window_spec", "df_deduplicated"], ns)
            return ns["df_deduplicated"]
        return ns["df_validated"]
    return run


@pytest.fixture
def aggregate():
    def run(frame, variable="df_daily", backfill=False):
        from pyspark.sql import functions as F
        path = source_path("11_FlightOps_2025_Gold_Backfill.py" if backfill else "04_BTS_Gold_Analytics.py")
        ns = assignments(path, [variable], {"df_silver": frame, "F": F})
        return ns[variable]
    return run
