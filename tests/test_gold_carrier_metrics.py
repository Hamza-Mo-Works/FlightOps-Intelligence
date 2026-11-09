"""Check carrier/date grain and carrier-specific metric denominators."""
import pytest
pytestmark = pytest.mark.spark


def test_carrier_counts_and_observed_arrival_rate(bronze, transform, aggregate):
    frame = bronze({}, {"Flight_Number_Reporting_Airline": "101", "ArrDelay": None},
                   {"Reporting_Airline": "DL", "Cancelled": "1", "ArrDelay": None})
    rows = aggregate(transform(frame), "df_carrier").collect()
    aa = next(row for row in rows if row.airline == "AA")
    assert aa.total_flights == 2 and aa.arrival_delay_observations == 1
    assert aa.on_time_arrival_rate_pct == 100.0
    dl = next(row for row in rows if row.airline == "DL")
    assert dl.cancelled_flights == 1 and dl.cancellation_rate_pct == 100.0
    assert dl.on_time_arrival_rate_pct is None
