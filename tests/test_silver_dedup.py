"""Exercise the notebook's flight key and ingestion/source-file ordering."""
from datetime import datetime
import pytest

pytestmark = pytest.mark.spark


def test_latest_ingestion_wins(bronze, transform):
    frame = bronze({"ArrDelay": "5"}, {"ArrDelay": "30", "_ingestion_timestamp": datetime(2026, 10, 21, 10)})
    result = transform(frame, deduplicate=True).collect()
    assert len(result) == 1 and result[0].arrival_delay_minutes == 30


def test_source_filename_breaks_timestamp_ties(bronze, transform):
    frame = bronze({"ArrDelay": "5", "_source_file": "a.csv"}, {"ArrDelay": "20", "_source_file": "z.csv"})
    result = transform(frame, deduplicate=True).collect()
    assert len(result) == 1 and result[0]._source_file == "z.csv"


def test_different_scheduled_departures_are_distinct_flights(bronze, transform):
    frame = bronze({}, {"CRSDepTime": "1000"})
    assert transform(frame, deduplicate=True).count() == 2


def test_validation_column_is_removed_from_final_silver(bronze, transform):
    assert "is_valid" not in transform(bronze(), deduplicate=True).columns
