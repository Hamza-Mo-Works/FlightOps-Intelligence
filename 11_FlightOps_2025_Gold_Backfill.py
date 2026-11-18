# Databricks notebook source
# MAGIC %md
# MAGIC # 11_FlightOps_2025_Gold_Backfill — Phase 12.3
# MAGIC
# MAGIC **Purpose:** Extend the **three existing Gold Delta tables** from July 2025 to all of 2025, **without touching July records**.
# MAGIC
# MAGIC - Reuses KPI aggregation expressions from the uploaded `04_BTS_Gold_Analytics` notebook.
# MAGIC - Reads **only 2025 Silver flights outside July**; does not rebuild the July Gold metrics.
# MAGIC - Checks all output schemas, grain keys and source-to-Gold reconciliations **before** writing.
# MAGIC - Uses Delta **append**; **never overwrite**.
# MAGIC - Skips an already completed Gold table on rerun. Refuses unexpected or partial table states. No simultaneous runs.
# MAGIC
# MAGIC > Do **not** rerun the original `04_BTS_Gold_Analytics` write cells (they use `overwrite`). Run this new notebook instead. Existing dashboards/app remain in place; their July-only date restrictions will be addressed in Phase 12.4–12.5.
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import date, timedelta

SILVER = "flightops_intelligence.silver.bts_flights_clean"
PREFIX = "flightops_intelligence.gold."
TARGETS = {
    "daily_flight_performance": (["flight_date"], 31),
    "airport_congestion_metrics": (["flight_date", "origin_airport"], 10549),
    "carrier_performance": (["flight_date", "airline"], 434),
}

all_dates_2025 = {date(2025, 1, 1) + timedelta(days=i) for i in range(365)}
july_dates = {d for d in all_dates_2025 if d.month == 7}
new_dates_expected = all_dates_2025 - july_dates

full_silver = spark.table(SILVER)
df_silver = full_silver.filter((F.year("flight_date") == 2025) & (F.month("flight_date") != 7))

counts = full_silver.groupBy(F.year("flight_date").alias("year"), F.month("flight_date").alias("month")).count().collect()
source_counts = {(r["year"], r["month"]): r["count"] for r in counts}
assert set(source_counts) == {(2025, m) for m in range(1, 13)}, f"Unexpected Silver coverage: {source_counts}"
assert source_counts[(2025, 7)] == 631428, "July Silver count changed"
assert sum(source_counts[(2025, m)] for m in range(1, 13)) == 7001619, "Full-year Silver row count changed"
assert sum(source_counts[(2025, m)] for m in range(1, 13) if m != 7) == 6370191, "Non-July Silver count changed"

new_dates_actual = {r["flight_date"] for r in df_silver.select("flight_date").distinct().collect()}
assert new_dates_actual == new_dates_expected, f"Non-July dates mismatch; absent={new_dates_expected-new_dates_actual}, extra={new_dates_actual-new_dates_expected}"
print("Source Silver: 7,001,619 rows; 6,370,191 non-July rows; 334 non-July dates")
print("July is excluded from df_silver and will remain untouched in all Gold tables.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Original business-logic aggregations for the other 11 months
# MAGIC
# MAGIC These calculations are taken from the existing Gold notebook, with the dataframe `df_silver` intentionally **restricted to dates outside July**. Only output-ordering and display operations were omitted; KPI formulas are unchanged.

# COMMAND ----------

df_daily = (
    df_silver
    .groupBy("flight_date")
    .agg(
        # Total scheduled flights
        F.count("*").alias("total_flights"),

        # Completed flights
        F.sum(
            F.when(
                (F.col("cancelled") == 0) &
                (F.col("diverted") == 0),
                1
            ).otherwise(0)
        ).alias("completed_flights"),

        # Cancelled flights
        F.sum(
            F.when(F.col("cancelled") == 1, 1).otherwise(0)
        ).alias("cancelled_flights"),

        # Diverted flights
        F.sum(
            F.when(F.col("diverted") == 1, 1).otherwise(0)
        ).alias("diverted_flights"),

        # Average departure delay for flights with an observed departure delay
        F.round(
            F.avg("departure_delay_minutes"), 2
        ).alias("avg_departure_delay_minutes"),

        # Average arrival delay for completed flights
        F.round(
            F.avg(
                F.when(
                    (F.col("cancelled") == 0) &
                    (F.col("diverted") == 0),
                    F.col("arrival_delay_minutes")
                )
            ), 2
        ).alias("avg_arrival_delay_minutes"),

        # Completed flights arriving on time (< 15 min delay)
        F.sum(
            F.when(
                (F.col("cancelled") == 0) &
                (F.col("diverted") == 0) &
                (F.col("arrival_delay_minutes") < 15),
                1
            ).otherwise(0)
        ).alias("on_time_arrivals"),
    
        # Departure delay aggregation components
        F.sum(
            "departure_delay_minutes"
        ).alias("total_departure_delay_minutes"),

        F.count(
            "departure_delay_minutes"
        ).alias("departure_delay_observations"),

        # Arrival delay aggregation components
        F.sum(
            F.when(
                (F.col("cancelled") == 0) &
                (F.col("diverted") == 0),
                F.col("arrival_delay_minutes")
            )
        ).alias("total_arrival_delay_minutes"),

        F.count(
            F.when(
                (F.col("cancelled") == 0) &
                (F.col("diverted") == 0),
                F.col("arrival_delay_minutes")
            )
        ).alias("arrival_delay_observations"),
            
            )

            # Business KPI calculations
            .withColumn(
                "cancellation_rate_pct",
                F.round(
                    F.col("cancelled_flights") /
                    F.col("total_flights") * 100, 2
                )
            )

            .withColumn(
                "diversion_rate_pct",
                F.round(
                    F.col("diverted_flights") /
                    F.col("total_flights") * 100, 2
                )
            )

            .withColumn(
                "on_time_arrival_rate_pct",
                F.when(
                    F.col("completed_flights") > 0,
                    F.round(
                        F.col("on_time_arrivals") /
                        F.col("completed_flights") * 100, 2
                    )
                )
            )
)


# COMMAND ----------

df_airport = (
    df_silver
    .groupBy(
        "flight_date",
        "origin_airport"
    )
    .agg(
        # Total scheduled departures
        F.count("*").alias("total_departures"),

        # Cancelled departures
        F.sum(
            F.when(
                F.col("cancelled") == 1, 1
            ).otherwise(0)
        ).alias("cancelled_departures"),

        # Flights with observed departure delay
        F.count(
            "departure_delay_minutes"
        ).alias("departure_delay_observations"),

        # Total departure delay minutes
        F.sum(
            "departure_delay_minutes"
        ).alias("total_departure_delay_minutes"),

        # Average departure delay
        F.round(
            F.avg("departure_delay_minutes"), 2
        ).alias("avg_departure_delay_minutes"),

        # Flights departing 15+ minutes late
        F.sum(
            F.when(
                F.col("departure_delay_minutes") >= 15,
                1
            ).otherwise(0)
        ).alias("delayed_departures_15"),

        # Taxi-out observations
        F.count(
            "taxi_out_minutes"
        ).alias("taxi_out_observations"),

        # Total taxi-out minutes
        F.sum(
            "taxi_out_minutes"
        ).alias("total_taxi_out_minutes"),

        # Average taxi-out time
        F.round(
            F.avg("taxi_out_minutes"), 2
        ).alias("avg_taxi_out_minutes")
    )

    .withColumn(
        "cancellation_rate_pct",
        F.round(
            F.col("cancelled_departures") /
            F.col("total_departures") * 100,
            2
        )
    )

    .withColumn(
        "departure_delay_rate_pct",
        F.when(
            F.col("departure_delay_observations") > 0,
            F.round(
                F.col("delayed_departures_15") /
                F.col("departure_delay_observations") * 100,
                2
            )
        )
    )
)


# COMMAND ----------

# Carrier Performance Aggregation

from pyspark.sql import functions as F

df_carrier = (
    df_silver
    .groupBy("flight_date", "airline")
    .agg(
        F.count("*").alias("total_flights"),

        F.sum(
            F.when(F.col("cancelled") == 1, 1).otherwise(0)
        ).alias("cancelled_flights"),

        F.sum(
            F.when(F.col("diverted") == 1, 1).otherwise(0)
        ).alias("diverted_flights"),

        F.sum(
            F.when(
                (F.col("cancelled") == 0) &
                (F.col("diverted") == 0),
                1
            ).otherwise(0)
        ).alias("completed_flights"),

        F.sum(
            F.when(
                F.col("is_arrival_delayed_15") == False,
                1
            ).otherwise(0)
        ).alias("on_time_arrivals"),

        F.count("arrival_delay_minutes").alias(
            "arrival_delay_observations"
        ),

        F.sum("arrival_delay_minutes").alias(
            "total_arrival_delay_minutes"
        ),

        F.count("departure_delay_minutes").alias(
            "departure_delay_observations"
        ),

        F.sum("departure_delay_minutes").alias(
            "total_departure_delay_minutes"
        )
    )

    .withColumn(
        "cancellation_rate_pct",
        F.round(
            F.col("cancelled_flights") /
            F.col("total_flights") * 100,
            2
        )
    )

    .withColumn(
        "on_time_arrival_rate_pct",
        F.when(
            F.col("arrival_delay_observations") > 0,
            F.round(
                F.col("on_time_arrivals") /
                F.col("arrival_delay_observations") * 100,
                2
            )
        )
    )

    .withColumn(
        "avg_arrival_delay_minutes",
        F.round(
            F.col("total_arrival_delay_minutes") /
            F.col("arrival_delay_observations"),
            2
        )
    )

    .withColumn(
        "avg_departure_delay_minutes",
        F.round(
            F.col("total_departure_delay_minutes") /
            F.col("departure_delay_observations"),
            2
        )
    )
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validate all new Gold data before writing anything
# MAGIC
# MAGIC Validate output schema (names/types), exact coverage, grain uniqueness, totals, cancellation/diversion/completed counts, delay observation counts, and month-level flight totals.

# COMMAND ----------

DATAFRAMES = {
    "daily_flight_performance": df_daily,
    "airport_congestion_metrics": df_airport,
    "carrier_performance": df_carrier,
}

# All schemas must match existing target field names and types.
for name, df in DATAFRAMES.items():
    target = spark.table(PREFIX + name)
    fields_target = {field.name: field.dataType.simpleString() for field in target.schema.fields}
    fields_new = {field.name: field.dataType.simpleString() for field in df.schema.fields}
    assert fields_new == fields_target, f"Schema mismatch in {name}: expected {fields_target}, found {fields_new}"
    assert len(df.columns) == len(target.columns), f"Column count mismatch: {name}"

expected_flights = 6370191
summary_silver = df_silver.agg(
    F.count("*").alias("flights"),
    F.sum(F.when(F.col("cancelled") == 1, 1).otherwise(0)).alias("cancelled"),
    F.sum(F.when(F.col("diverted") == 1, 1).otherwise(0)).alias("diverted"),
    F.sum(F.when((F.col("cancelled") == 0) & (F.col("diverted") == 0), 1).otherwise(0)).alias("completed"),
    F.count("departure_delay_minutes").alias("departure_obs"),
    F.count("arrival_delay_minutes").alias("arrival_obs"),
    F.count("taxi_out_minutes").alias("taxi_obs"),
).first()
assert summary_silver["flights"] == expected_flights

summary_daily = df_daily.agg(
    F.count("*").alias("rows"),
    F.sum("total_flights").alias("flights"),
    F.sum("cancelled_flights").alias("cancelled"),
    F.sum("diverted_flights").alias("diverted"),
    F.sum("completed_flights").alias("completed"),
    F.sum("departure_delay_observations").alias("departure_obs"),
    F.sum("arrival_delay_observations").alias("arrival_obs"),
).first()
assert summary_daily["rows"] == 334
for gold_key, silver_key in [("flights", "flights"), ("cancelled", "cancelled"), ("diverted", "diverted"), ("completed", "completed"), ("departure_obs", "departure_obs")]:
    assert summary_daily[gold_key] == summary_silver[silver_key], f"Daily {gold_key} mismatch"

summary_airport = df_airport.agg(
    F.sum("total_departures").alias("flights"),
    F.sum("cancelled_departures").alias("cancelled"),
    F.sum("departure_delay_observations").alias("departure_obs"),
    F.sum("taxi_out_observations").alias("taxi_obs"),
).first()
for key in ["flights", "cancelled", "departure_obs", "taxi_obs"]:
    assert summary_airport[key] == summary_silver[key], f"Airport {key} mismatch"

summary_carrier = df_carrier.agg(
    F.sum("total_flights").alias("flights"),
    F.sum("cancelled_flights").alias("cancelled"),
    F.sum("diverted_flights").alias("diverted"),
    F.sum("completed_flights").alias("completed"),
    F.sum("departure_delay_observations").alias("departure_obs"),
    F.sum("arrival_delay_observations").alias("arrival_obs"),
).first()
for key in ["flights", "cancelled", "diverted", "completed", "departure_obs", "arrival_obs"]:
    assert summary_carrier[key] == summary_silver[key], f"Carrier {key} mismatch"

# Verify original Gold grain and business rules.
for name, df in DATAFRAMES.items():
    grain = TARGETS[name][0]
    assert df.filter(F.col("flight_date").isNull() | (F.month("flight_date") == 7)).limit(1).count() == 0, f"July or null date in {name} backfill"
    for key in grain[1:]:
        assert df.filter(F.col(key).isNull()).limit(1).count() == 0, f"Null {key} in {name}"
    dup_count = df.groupBy(*grain).count().filter("count > 1").limit(1).count()
    assert dup_count == 0, f"Duplicate grain in {name}"

assert df_daily.filter(F.col("total_flights") != (F.col("completed_flights") + F.col("cancelled_flights") + F.col("diverted_flights"))).limit(1).count() == 0
assert df_airport.filter((F.col("total_departures") <= 0) | (F.col("cancelled_departures") > F.col("total_departures")) | (F.col("departure_delay_observations") > F.col("total_departures")) | (F.col("taxi_out_observations") > F.col("total_departures"))).limit(1).count() == 0
assert df_carrier.filter((F.col("total_flights") <= 0) | (F.col("total_flights") != F.col("completed_flights") + F.col("cancelled_flights") + F.col("diverted_flights")) | (F.col("on_time_arrivals") > F.col("completed_flights")) | (F.col("arrival_delay_observations") > F.col("total_flights"))).limit(1).count() == 0

# Ensure each day has a Gold record, and the daily totals match source monthly totals.
monthly_daily = {r["month"]:r["total"] for r in df_daily.groupBy(F.month("flight_date").alias("month")).agg(F.sum("total_flights").alias("total")).collect()}
for m in range(1,13):
    if m != 7:
        assert monthly_daily.get(m) == source_counts[(2025,m)], f"Daily Gold month {m} mismatch"
print("ALL PRE-WRITE GOLD VALIDATIONS PASSED")
print("New Gold daily rows:", summary_daily["rows"])
print("New Gold airport rows:", df_airport.count())
print("New Gold carrier rows:", df_carrier.count())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Append only missing Gold dates — safe rerun guard
# MAGIC
# MAGIC This cell pre-checks **all three targets** before writing. Each target must contain **only the original 31 July days** or **all 365 2025 days** (completed on a previous run). It refuses partially populated or unexpected states. It does **not** overwrite tables or delete July.

# COMMAND ----------

# Determine safe target states before touching any Gold table.
statuses = {}
july_before = {}
for name, (_, expected_july_rows) in TARGETS.items():
    target = spark.table(PREFIX + name)
    existing_dates = {r["flight_date"] for r in target.select("flight_date").distinct().collect()}
    july_df = target.filter((F.year("flight_date") == 2025) & (F.month("flight_date") == 7))
    july_rows = july_df.count()
    assert july_rows == expected_july_rows, f"{name}: July row count changed: {july_rows}"
    july_before[name] = july_rows
    assert target.groupBy(*TARGETS[name][0]).count().filter("count > 1").limit(1).count() == 0, f"Existing duplicate key in {name}"
    if existing_dates == july_dates:
        statuses[name] = "NEEDS_BACKFILL"
    elif existing_dates == all_dates_2025:
        statuses[name] = "ALREADY_FULL_YEAR"
        new_existing = target.filter((F.year("flight_date") == 2025) & (F.month("flight_date") != 7))
        expected_count = DATAFRAMES[name].count()
        assert new_existing.count() == expected_count, f"{name}: existing full-year row count differs from backfill"
        expected_sum_key = "total_departures" if name == "airport_congestion_metrics" else "total_flights"
        assert new_existing.agg(F.sum(expected_sum_key)).first()[0] == expected_flights, f"{name}: existing full-year flight total mismatch"
    else:
        raise AssertionError(f"{name}: unexpected Gold date coverage; do not append. Dates count={len(existing_dates)}, unexpected={sorted(existing_dates - all_dates_2025)[:5]}, missing={len(all_dates_2025-existing_dates)}")
print("SAFE GOLD TARGET STATES:", statuses)

for name, state in statuses.items():
    if state == "ALREADY_FULL_YEAR":
        print(f"{name}: SKIPPED — previously completed")
        continue
    df = DATAFRAMES[name]
    target_columns = spark.table(PREFIX + name).columns
    assert df.filter(F.month("flight_date") == 7).limit(1).count() == 0
    print(f"{name}: appending {df.count():,} non-July aggregate rows")
    (df.select(*target_columns).write.format("delta").mode("append").saveAsTable(PREFIX + name))
    after_dates = {r["flight_date"] for r in spark.table(PREFIX + name).select("flight_date").distinct().collect()}
    assert after_dates == all_dates_2025, f"{name}: coverage mismatch after append"
    july_after = spark.table(PREFIX + name).filter((F.year("flight_date") == 2025) & (F.month("flight_date") == 7)).count()
    assert july_after == july_before[name], f"{name}: July changed after append"
    print(f"{name}: APPEND SUCCESS — July {july_after:,} rows preserved")

print("GOLD BACKFILL WRITE PHASE COMPLETE")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Final full-year Gold reconciliation
# MAGIC
# MAGIC All three Gold tables must cover the complete year and reconcile to the 7,001,619 Silver flights. July's Gold row counts remain unchanged.

# COMMAND ----------

for name, (grain, expected_july_rows) in TARGETS.items():
    g = spark.table(PREFIX + name)
    metric = "total_departures" if name == "airport_congestion_metrics" else "total_flights"
    totals = g.agg(F.count("*").alias("gold_rows"),
                   F.sum(metric).alias("flight_count"),
                   F.countDistinct("flight_date").alias("days"),
                   F.min("flight_date").alias("start"),
                   F.max("flight_date").alias("end")).first()
    july_rows = g.filter((F.year("flight_date") == 2025) & (F.month("flight_date") == 7)).count()
    duplicate_grains = g.groupBy(*grain).count().filter("count > 1").limit(1).count()
    assert totals["flight_count"] == 7001619, f"{name} flight total mismatch"
    assert totals["days"] == 365 and totals["start"] == date(2025,1,1) and totals["end"] == date(2025,12,31), f"{name} coverage mismatch"
    assert july_rows == expected_july_rows, f"{name} July row count mismatch"
    assert duplicate_grains == 0, f"{name} duplicate Gold grain"
    print(f"{name}: PASS | rows={totals['gold_rows']:,}, flights={totals['flight_count']:,}, dates={totals['days']}, July rows={july_rows:,}")

# Display source-vs-Gold monthly totals for daily table.
monthly_gold = (spark.table(PREFIX + "daily_flight_performance")
    .groupBy(F.month("flight_date").alias("month"))
    .agg(F.count("*").alias("reporting_days"), F.sum("total_flights").alias("gold_flights"))
    .orderBy("month"))
display(monthly_gold)
print("PHASE 12.3: FULL-YEAR GOLD RECONCILIATION PASSED")


# COMMAND ----------

