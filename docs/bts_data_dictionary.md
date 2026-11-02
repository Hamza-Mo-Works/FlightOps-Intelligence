# BTS Flight Data Dictionary — FlightOps Intelligence

## Purpose and scope

This reference documents the **US Bureau of Transportation Statistics (BTS)
Airline On-Time Performance** attributes used in the FlightOps ingestion and
Silver transformation pipeline. The source consists of twelve monthly files
covering January–December 2025. The source field names below are grounded in
the project's `03_BTS_Silver_Transformation` notebook.

Bronze retains the original source fields; Silver standardizes names and casts
values to consistent types. **Types shown below describe intended Silver
representations**, not an assertion about how every source CSV column was parsed
on ingestion. Actual source-file schemas should be inspected for each month.

## Flight identity and schedule

| BTS source field | Silver field | Intended representation | Meaning / quality note |
|---|---|---|---|
| `FlightDate` | `flight_date` | Date | Flight service date; source may use ISO or `M/d/yyyy` notation |
| `Year` | `year` | Integer | Reporting year |
| `Month` | `month` | Integer | Reporting month, 1–12 |
| `DayOfWeek` | `day_of_week` | Integer | Day-of-week code in BTS data |
| `Reporting_Airline` | `airline` | String | Operating carrier code |
| `Flight_Number_Reporting_Airline` | `flight_number` | String / identifier | Flight number within the reporting airline |
| `Tail_Number` | `tail_number` | String, nullable | Aircraft registration; may be missing |
| `Origin` | `origin_airport` | String | IATA departure airport code |
| `Dest` | `destination_airport` | String | IATA scheduled destination airport code |
| `OriginAirportID` | `origin_airport_id` | Identifier | BTS airport identifier |
| `DestAirportID` | `destination_airport_id` | Identifier | BTS airport identifier |
| `CRSDepTime` | `scheduled_departure_time` | HHMM representation | Published origin-local departure time |
| `CRSArrTime` | `scheduled_arrival_time` | HHMM representation | Published destination-local arrival time |
| `DepTime` | `actual_departure_time` | HHMM, nullable | Observed departure time, when applicable |
| `ArrTime` | `actual_arrival_time` | HHMM, nullable | Observed arrival time, when applicable |
| `CRSElapsedTime` | `scheduled_elapsed_minutes` | Numeric | Published elapsed flight time |
| `Distance` | `distance_miles` | Double | Flight distance in miles; required to be positive in Silver checks |
| `AirTime` | `air_time_minutes` | Double, nullable | Actual airborne duration |

## Delays and operational outcomes

| BTS source field | Silver field | Intended representation | Meaning / quality note |
|---|---|---|---|
| `DepDelay` | `departure_delay_minutes` | Double, nullable | Departure delay in minutes; negative indicates early departure |
| `ArrDelay` | `arrival_delay_minutes` | Double, nullable | Arrival delay in minutes; negative indicates early arrival |
| `TaxiOut` | `taxi_out_minutes` | Double, nullable | Gate departure to takeoff duration |
| `TaxiIn` | `taxi_in_minutes` | Double, nullable | Landing to arrival gate duration |
| `Cancelled` | `cancelled` | 0/1 integer | Flight cancellation flag |
| `Diverted` | `diverted` | 0/1 integer | Diversion flag |
| `CancellationCode` | `cancellation_code` | String, nullable | Source cancellation reason category |
| `CarrierDelay` | `carrier_delay_minutes` | Double, nullable | Reported carrier-attributed delay minutes |
| `WeatherDelay` | `weather_delay_minutes` | Double, nullable | Reported weather-attributed delay minutes |
| `NASDelay` | `nas_delay_minutes` | Double, nullable | Reported National Airspace System delay minutes |
| `SecurityDelay` | `security_delay_minutes` | Double, nullable | Reported security-attributed delay minutes |
| `LateAircraftDelay` | `late_aircraft_delay_minutes` | Double, nullable | Reported late-inbound-aircraft delay minutes |

## Bronze ingestion metadata

These fields are added by the ingestion process and carried forward to Silver:

| Column | Purpose |
|---|---|
| `_ingestion_timestamp` | Timestamp identifying when the record was loaded |
| `_source_file` | Source filename/path for lineage and issue investigation |
| `_source_system` | Identifier of the originating dataset/source |

## Derived Silver fields

- `is_completed`: flight was neither cancelled nor diverted.
- `is_departure_delayed_15`: observed departure delay is at least 15 minutes,
  when an applicable observation exists.
- `is_arrival_delayed_15`: completed flight arrived at least 15 minutes late,
  when an applicable arrival observation exists.

Cancelled and diverted records remain in Silver for reporting, but do not
receive ordinary completed-flight arrival-delay labels for binary ML training.

## Validation considerations

1. Accept and normalize actual source date representations before checking year/month coherence.
2. Validate mandatory flight identifiers and positive distance where required.
3. Keep null operational outcomes distinct from zero-minute delays.
4. Validate cancellation/diversion status flags before deriving completed-flight indicators.
5. Examine duplicate business keys before applying deterministic deduplication.
6. Reconcile record counts at source, Bronze, Silver and aggregated Gold stages.
7. Maintain source attribution to identify the monthly file associated with anomalies.

## Notes

- The raw schema may contain additional BTS columns beyond the ones projected
  into the current Silver pipeline.
- This is a field-definition reference, not a statement that every data-quality
  test has already been run or passed in a particular Databricks workspace.
- Reference: [BTS Airline On-Time Performance data](https://www.transtats.bts.gov/ONTIME/).
