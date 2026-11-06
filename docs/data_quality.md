# Silver data quality

`03_BTS_Silver_Transformation.py` maps raw BTS fields to typed flight columns,
adds operational flags, filters valid rows, removes duplicate flight keys and
writes `flightops_intelligence.silver.bts_flights_clean`.

The initial projection parses `FlightDate` with `M/d/yyyy`. The backfill notebook
also parses `yyyy-MM-dd` through `TRY_TO_TIMESTAMP`. These are distinct source
paths and the tests exercise both. The backfill accepts cancelled/diverted values
only when their numeric representation is exactly zero or one.

A valid row needs a date, airline, flight number, origin, destination, zero/one
status flags and positive distance. Arrival/departure delays can remain null;
missing delay is not zero delay. Arrival delay of at least 15 minutes is delayed;
negative values represent early arrivals. Completion requires both status flags
to be zero.

SQL null logic matters: `is_valid` can be null for a missing status flag.
`filter(is_valid)` accepts only true and `filter(~is_valid)` selects only false;
null belongs to neither result. Count null validation outcomes separately when
reconciling rejected and accepted populations. The tests preserve this observed
behavior rather than asserting that the existing quarantine covers every null.

Deduplication uses flight date, airline, flight number, origin, destination and
scheduled departure time. Newer ingestion timestamps win, followed by descending
source filename. If conflicting rows tie on both ordering fields, that ordering
does not define a unique winner.

Run `python -m pytest -q tests/test_silver_transforms.py tests/test_silver_dedup.py`
with an enabled local Spark session or from the Databricks Python kernel.
