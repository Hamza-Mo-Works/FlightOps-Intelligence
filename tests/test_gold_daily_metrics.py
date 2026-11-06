"""Execute the daily Gold expression on a bounded flight dataset."""
import pytest
pytestmark = pytest.mark.spark


def test_status_counts_and_rates(bronze, transform, aggregate):
    silver = transform(bronze({}, {"Flight_Number_Reporting_Airline": "101", "ArrDelay": "20"},
        {"Flight_Number_Reporting_Airline": "102", "Cancelled": "1", "ArrDelay": None},
        {"Flight_Number_Reporting_Airline": "103", "Diverted": "1", "ArrDelay": None}))
    row = aggregate(silver).first()
    assert (row.total_flights, row.completed_flights, row.cancelled_flights, row.diverted_flights) == (4, 2, 1, 1)
    assert row.cancellation_rate_pct == 25.0 and row.diversion_rate_pct == 25.0
    assert row.on_time_arrival_rate_pct == 50.0
    assert row.total_arrival_delay_minutes == 25.0 and row.arrival_delay_observations == 2


def test_missing_arrival_is_not_counted_as_on_time(bronze, transform, aggregate):
    row = aggregate(transform(bronze({"ArrDelay": None}))).first()
    assert row.completed_flights == 1 and row.on_time_arrivals == 0
    assert row.arrival_delay_observations == 0 and row.avg_arrival_delay_minutes is None


def test_all_cancelled_day_has_null_on_time_rate(bronze, transform, aggregate):
    row = aggregate(transform(bronze({"Cancelled": "1", "ArrDelay": None}))).first()
    assert row.completed_flights == 0 and row.on_time_arrival_rate_pct is None
