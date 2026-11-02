"""
In Databricks:
    %pip install pytest
    %sh python -m pytest -q tests/test_bronze_ingestion.py

Or from a repository configured with Databricks Connect:
    python -m pytest -q tests/test_bronze_ingestion.py

Prerequisites: a working SparkSession and SELECT access to the Bronze table.
No records are inserted, updated, or deleted.
"""

from __future__ import annotations

import os
import unittest


BRONZE_TABLE = os.getenv(
    "FLIGHTOPS_BRONZE_TABLE",
    "flightops_intelligence.bronze.bts_flights_raw",
)

# Based on the actual raw-column references in 03_BTS_Silver_Transformation.
REQUIRED_SOURCE_COLUMNS = {
    "FlightDate",
    "Year",
    "Month",
    "DayOfWeek",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "Cancelled",
    "Diverted",
    "Distance",
}
REQUIRED_METADATA_COLUMNS = {
    "_ingestion_timestamp",
    "_source_file",
    "_source_system",
}


class TestBTSBronzeIngestion(unittest.TestCase):
    """Validate source fidelity and ingestion metadata on the Bronze table."""

    @classmethod
    def setUpClass(cls):
        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise unittest.SkipTest("PySpark is unavailable; run with Databricks Spark") from exc

        # Avoid starting an unrelated local Spark session automatically.
        cls.spark = SparkSession.getActiveSession()
        if cls.spark is None:
            raise unittest.SkipTest("No active Spark session; run in Databricks or Databricks Connect")
        try:
            cls.bronze = cls.spark.table(BRONZE_TABLE)
            cls.columns = set(cls.bronze.columns)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot access {BRONZE_TABLE}; verify catalog and SELECT permission: {exc}"
            ) from exc

    def test_source_columns_are_preserved(self):
        missing = REQUIRED_SOURCE_COLUMNS - self.columns
        self.assertFalse(missing, f"Missing BTS source columns: {sorted(missing)}")

    def test_ingestion_metadata_columns_exist(self):
        missing = REQUIRED_METADATA_COLUMNS - self.columns
        self.assertFalse(missing, f"Missing ingestion metadata: {sorted(missing)}")

    def test_table_contains_flight_records(self):
        self.assertEqual(self.bronze.limit(1).count(), 1, "Bronze table is empty")

    def test_sample_has_source_attribution(self):
        # Bounded scan avoids collecting the 7M+ row table into driver memory.
        missing = REQUIRED_METADATA_COLUMNS - self.columns
        if missing:
            self.fail(f"Cannot inspect attribution; missing metadata: {sorted(missing)}")

        rows = self.bronze.select(
            "_ingestion_timestamp", "_source_file", "_source_system"
        ).limit(1000).collect()
        self.assertTrue(rows, "Bronze sample is empty")
        for index, row in enumerate(rows, start=1):
            self.assertIsNotNone(row["_ingestion_timestamp"],
                                 f"Row {index}: missing ingestion timestamp")
            self.assertTrue(str(row["_source_file"] or "").strip(),
                            f"Row {index}: missing source file")
            self.assertTrue(str(row["_source_system"] or "").strip(),
                            f"Row {index}: missing source system")


if __name__ == "__main__":
    unittest.main()
