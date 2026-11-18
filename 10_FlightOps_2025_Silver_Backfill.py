# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 12.2 — Silver backfill: 11 missing months (2025)
# MAGIC
# MAGIC This notebook is adapted directly from `03_BTS_Silver_Transformation` and **does not overwrite** `flightops_intelligence.silver.bts_flights_clean`.
# MAGIC
# MAGIC - July 2025 (631,428 existing rows) is explicitly excluded from the read and protected by a guard.
# MAGIC - Runs only when the Silver table contains **July only**; a complete 12-month load is a no-op; any unexpected intermediate state stops.
# MAGIC - Uses the same column mappings, validation, deduplication key/order and operational flags as the original Silver notebook.
# MAGIC - The sole transformation change is parsing both `yyyy-MM-dd` and `M/d/yyyy` FlightDate formats.
# MAGIC - Performs an atomic Delta append only after all checks succeed. **Run cells from top to bottom; do not rerun the original overwrite cell.**
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

source_table = "flightops_intelligence.bronze.bts_flights_raw"
target_table = "flightops_intelligence.silver.bts_flights_clean"
OTHER_MONTHS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

silver_before = spark.table(target_table)

existing_months = {
    int(r['month']): int(r['count'])
    for r in (
        silver_before.filter(F.col('year') == 2025)
        .groupBy('month').count().collect()
    )
}

# The original validated July output must stay untouched.
assert existing_months.get(7) == 631428, (
    f"July Silver count changed: {existing_months.get(7)}; expected 631428"
)

# Make sure no other years are present in this single-year MVP table.
other_year_rows = silver_before.filter(F.col('year') != 2025).limit(1).count()
assert other_year_rows == 0, "Silver contains a different year; stop and review"

if set(existing_months) == set(range(1, 13)):
    print('Already contains all 12 months. No write needed.')
    RUN_BACKFILL = False
else:
    assert set(existing_months) == {7}, (
        f"Unexpected existing Silver months: {existing_months}. "
        "Stop rather than risk a partial-month or duplicate append."
    )
    RUN_BACKFILL = True
    print('Safe to backfill 11 missing months. July will not be read or written.')

print('Existing Silver records:', silver_before.count())
print('Existing month counts:', existing_months)

# COMMAND ----------

if RUN_BACKFILL:
    # Restrict the entire source to January–June and August–December.
    df_bronze = spark.table(source_table).filter(
        (F.col("Year").cast("int") == 2025) &
        (F.col("Month").cast("int").isin(OTHER_MONTHS))
    )
    bronze_count = df_bronze.count()
    assert bronze_count > 0, 'No records found for 11 additional months'
    print(f"Bronze rows selected (excluding July): {bronze_count:,}")

    df_typed = df_bronze.select(
        F.coalesce(
            F.expr("CAST(TRY_TO_TIMESTAMP(FlightDate, 'yyyy-MM-dd') AS DATE)"),
            F.expr("CAST(TRY_TO_TIMESTAMP(FlightDate, 'M/d/yyyy') AS DATE)")
        ).alias("flight_date"),

        F.col("Year").cast("int").alias("year"),
        F.col("Month").cast("int").alias("month"),
        F.col("DayOfWeek").cast("int").alias("day_of_week"),

        F.col("Reporting_Airline").alias("airline"),
        F.col("Tail_Number").alias("tail_number"),
        F.col("Flight_Number_Reporting_Airline").alias("flight_number"),

        F.col("Origin").alias("origin_airport"),
        F.col("Dest").alias("destination_airport"),

        F.col("OriginAirportID").alias("origin_airport_id"),
        F.col("DestAirportID").alias("destination_airport_id"),

        F.col("CRSDepTime").alias("scheduled_departure_time"),
        F.col("CRSArrTime").alias("scheduled_arrival_time"),

        F.col("DepTime").alias("actual_departure_time"),
        F.col("ArrTime").alias("actual_arrival_time"),

        F.col("DepDelay").cast("double").alias("departure_delay_minutes"),
        F.col("ArrDelay").cast("double").alias("arrival_delay_minutes"),

        F.col("TaxiOut").cast("double").alias("taxi_out_minutes"),
        F.col("TaxiIn").cast("double").alias("taxi_in_minutes"),

        F.col("AirTime").cast("double").alias("air_time_minutes"),
        F.col("Distance").cast("double").alias("distance_miles"),

        F.when(
            F.expr("try_cast(Cancelled AS DOUBLE)").isin(0.0, 1.0),
            F.expr("try_cast(Cancelled AS DOUBLE)").cast("int")
        ).alias("cancelled"),
        F.when(
            F.expr("try_cast(Diverted AS DOUBLE)").isin(0.0, 1.0),
            F.expr("try_cast(Diverted AS DOUBLE)").cast("int")
        ).alias("diverted"),
        F.col("CancellationCode").alias("cancellation_code"),

        F.col("CarrierDelay").cast("double").alias("carrier_delay_minutes"),
        F.col("WeatherDelay").cast("double").alias("weather_delay_minutes"),
        F.col("NASDelay").cast("double").alias("nas_delay_minutes"),
        F.col("SecurityDelay").cast("double").alias("security_delay_minutes"),
        F.col("LateAircraftDelay").cast("double").alias("late_aircraft_delay_minutes"),

        F.col("CRSElapsedTime")
            .cast("double")
            .alias("scheduled_elapsed_minutes"),
        
        F.col("_ingestion_timestamp"),
        F.col("_source_file"),
        F.col("_source_system")
    )

    df_validated = (
        df_typed
        .withColumn(
            "is_completed",
            (F.col("cancelled") == 0) &
            (F.col("diverted") == 0)
        )
        .withColumn(
            "is_departure_delayed_15",
            F.when(
                (F.col("cancelled") == 0) &
                F.col("departure_delay_minutes").isNotNull(),
                F.col("departure_delay_minutes") >= 15
            ).otherwise(F.lit(None).cast("boolean"))
        )
        .withColumn(
            "is_arrival_delayed_15",
            F.when(
                F.col("is_completed") &
                F.col("arrival_delay_minutes").isNotNull(),
                F.col("arrival_delay_minutes") >= 15
            ).otherwise(F.lit(None).cast("boolean"))
        )
        .withColumn(
            "is_valid",
            F.col("flight_date").isNotNull() &
            F.col("airline").isNotNull() &
            F.col("origin_airport").isNotNull() &
            F.col("destination_airport").isNotNull() &
            F.col("flight_number").isNotNull() &
            F.col("cancelled").isin(0, 1) &
            F.col("diverted").isin(0, 1) &
            (F.col("distance_miles") > 0)
        )
    )

    # Bronze already passed all-year date validation; recheck the selected Silver dates.
    wrong_dates = df_typed.filter(
        F.col('flight_date').isNull() |
        (F.col('year') != 2025) |
        (~F.col('month').isin(OTHER_MONTHS)) |
        (F.year('flight_date') != 2025) |
        (F.month('flight_date') != F.col('month'))
    ).count()
    assert wrong_dates == 0, f'Found {wrong_dates} invalid/mismatched transformed dates'

    # The original notebook quarantines failures. This backfill deliberately stops
    # before writing if there are rejects, so they can be reviewed/quarantined safely.
    rejected_count = df_validated.filter(
        ~F.coalesce(F.col('is_valid'), F.lit(False))
    ).count()
    print(f'Quality-rule rejected rows: {rejected_count:,}')
    assert rejected_count == 0, (
        'Quality rejects detected. Stop before append. Review the failed rows '
        'and handle quarantine before proceeding.'
    )
    df_valid = df_validated.filter(F.col('is_valid'))

    flight_key = [
        "flight_date",
        "airline",
        "flight_number",
        "origin_airport",
        "destination_airport",
        "scheduled_departure_time"
    ]

    duplicate_groups = (
        df_valid.groupBy(*flight_key)
        .count()
        .filter(F.col("count") > 1)
    )



    window_spec = (
        Window.partitionBy(*flight_key)
        .orderBy(
            F.col("_ingestion_timestamp").desc(),
            F.col("_source_file").desc()
        )
    )

    df_deduplicated = (
        df_valid
        .withColumn("_row_number", F.row_number().over(window_spec))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number", "is_valid")
    )

    new_silver_count = df_deduplicated.count()
    deduplicated_rows = bronze_count - rejected_count - new_silver_count
    print(f'Duplicate rows removed using original flight key: {deduplicated_rows:,}')
    print(f'Validated new Silver rows: {new_silver_count:,}')

    # Assert schema and type compatibility with existing July Silver table.
    existing_fields = [(f.name, f.dataType.simpleString())
                       for f in silver_before.schema.fields]
    new_fields = [(f.name, f.dataType.simpleString())
                  for f in df_deduplicated.schema.fields]
    assert existing_fields == new_fields, (
        f'Silver schema differs from original.\nExisting: {existing_fields}\nNew: {new_fields}'
    )

    # Recheck target immediately before writing (sequential execution only).
    current_months = {
        int(r['month']) for r in spark.table(target_table)
        .filter(F.col('year') == 2025).select('month').distinct().collect()
    }
    assert current_months == {7}, (
        f'Target months changed before append: {current_months}'
    )

    # Atomic Delta append. Does not rewrite existing July rows.
    (df_deduplicated.select(*silver_before.columns).write
        .format('delta')
        .mode('append')
        .saveAsTable(target_table))
    print('BACKFILL APPEND COMPLETE')
else:
    print('Backfill skipped. Existing Silver data was not modified.')


# COMMAND ----------

# Final reconciliation. July remains unchanged and 365 days must be present.
final_silver = spark.table(target_table)
monthly = (
    final_silver.filter(F.col('year') == 2025)
    .groupBy('month')
    .agg(F.count('*').alias('silver_flights'),
         F.countDistinct('flight_date').alias('reporting_days'),
         F.min('flight_date').alias('start_date'),
         F.max('flight_date').alias('end_date'))
    .orderBy('month')
)
display(monthly)

assert final_silver.filter((F.col('year') == 2025) &
                           (F.col('month') == 7)).count() == 631428
assert monthly.count() == 12, 'Expected exactly 12 months in Silver'
assert final_silver.filter(F.col('flight_date').isNull()).limit(1).count() == 0
assert final_silver.select('flight_date').distinct().count() == 365
print('SILVER RECONCILIATION PASSED')
print('Total Silver records:', final_silver.count())

# COMMAND ----------

