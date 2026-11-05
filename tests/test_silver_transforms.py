"""Check the actual Silver projection, operational flags and backfill parser."""
from datetime import date
import pytest

pytestmark = pytest.mark.spark


def test_primary_month_day_year_parser(bronze, transform):
    row = transform(bronze()).first()
    assert row.flight_date == date(2025, 7, 23)
    assert row.year == 2025 and row.month == 7
    assert row.departure_delay_minutes == 10.0


@pytest.mark.parametrize("raw", ["7/23/2025", "2025-07-23"])
def test_backfill_accepts_both_source_date_formats(bronze, transform, raw):
    row = transform(bronze({"FlightDate": raw}), backfill=True).first()
    assert row.flight_date == date(2025, 7, 23)


@pytest.mark.parametrize("raw,expected", [("0.00", 0), ("1.00", 1), ("2", None), ("bad", None)])
def test_backfill_flag_conversion_is_guarded(bronze, transform, raw, expected):
    assert transform(bronze({"Cancelled": raw}), backfill=True).first().cancelled == expected


def test_nullable_delay_fields_remain_nullable(bronze, transform):
    row = transform(bronze({"ArrDelay": None, "DepDelay": None})).first()
    assert row.is_valid is True
    assert row.is_arrival_delayed_15 is None and row.is_departure_delayed_15 is None


@pytest.mark.parametrize("value,delayed", [("14.99", False), ("15", True), ("-3", False)])
def test_arrival_delay_threshold(bronze, transform, value, delayed):
    assert transform(bronze({"ArrDelay": value})).first().is_arrival_delayed_15 is delayed


def test_invalid_distance_is_not_accepted(bronze, transform):
    assert transform(bronze({"Distance": "0"})).first().is_valid is False


def test_null_flag_uses_sql_three_valued_logic(bronze, transform):
    row = transform(bronze({"Cancelled": None})).first()
    assert row.is_valid is None
