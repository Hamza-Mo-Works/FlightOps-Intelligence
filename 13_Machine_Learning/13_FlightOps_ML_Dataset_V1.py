# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 13.2 — ML Dataset Preparation V1
# MAGIC
# MAGIC ## FlightOps Intelligence | Feature Engineering
# MAGIC
# MAGIC ### Objective
# MAGIC Create a reproducible, leakage-free machine learning dataset
# MAGIC using the existing Silver flight data.
# MAGIC
# MAGIC ### Source
# MAGIC `flightops_intelligence.silver.bts_flights_clean`
# MAGIC
# MAGIC ### Feature Engineering
# MAGIC - Airline, origin and destination airports
# MAGIC - Route identifier
# MAGIC - Scheduled departure and arrival times
# MAGIC - Flight distance
# MAGIC - Month, day of week and weekend indicator
# MAGIC
# MAGIC ### Target
# MAGIC `delay_label`: 1 if arrival delay >= 15 minutes, otherwise 0.
# MAGIC
# MAGIC ### Data Quality Checks
# MAGIC - Validate scheduled times and flight distance.
# MAGIC - Reject missing or invalid required features.
# MAGIC - Reconcile eligible flight counts against Phase 13.1.
# MAGIC
# MAGIC **Important:** Actual arrival delays are used only to create
# MAGIC the training label, never as prediction features.

# COMMAND ----------

from pyspark.sql import functions as F

SILVER = "flightops_intelligence.silver.bts_flights_clean"
ML_TABLE = "flightops_intelligence.ml.flight_delay_features_v1"

silver = spark.table(SILVER)

# Only flights with valid normal-arrival outcomes
eligible = silver.filter(
    (F.col("cancelled") == 0) &
    (F.col("diverted") == 0) &
    F.col("arrival_delay_minutes").isNotNull()
)

# Convert scheduled HHMM times into hour/minute features
# Supports values such as 700, 0730, 1430 and 2400.
def add_time_features(df, source_col, prefix):
    raw = F.trim(F.col(source_col))

    df = df.withColumn(
        f"_{prefix}_hhmm",
        F.when(
            raw.rlike(r"^[0-9]{1,4}$"),
            F.lpad(raw, 4, "0")
        )
    )

    hhmm = F.col(f"_{prefix}_hhmm")
    hours = F.substring(hhmm, 1, 2).cast("int")
    minutes = F.substring(hhmm, 3, 2).cast("int")

    valid = (
        ((hours >= 0) & (hours <= 23) &
         (minutes >= 0) & (minutes <= 59))
        | (hhmm == "2400")
    )

    return (
        df
        .withColumn(
            f"{prefix}_hour",
            F.when(valid, F.when(hhmm == "2400", 0)
                   .otherwise(hours))
        )
        .withColumn(
            f"{prefix}_minute",
            F.when(valid, F.when(hhmm == "2400", 0)
                   .otherwise(minutes))
        )
        .drop(f"_{prefix}_hhmm")
    )

features = add_time_features(
    eligible, "scheduled_departure_time", "departure"
)

features = add_time_features(
    features, "scheduled_arrival_time", "arrival"
)

ml_df = features.select(
    "flight_date",
    "airline",
    "origin_airport",
    "destination_airport",
    F.concat_ws(
        "_", "origin_airport", "destination_airport"
    ).alias("route"),
    "month",
    "day_of_week",
    F.when(
        F.dayofweek("flight_date").isin(1, 7), 1
    ).otherwise(0).alias("is_weekend"),
    "departure_hour",
    "departure_minute",
    "arrival_hour",
    "arrival_minute",
    "distance_miles",

    # Target only — never use this column as an input feature
    F.when(
        F.col("arrival_delay_minutes") >= 15, 1
    ).otherwise(0).alias("delay_label")
)

# Validate engineered features before writing
invalid = ml_df.filter(
    F.col("flight_date").isNull()
    | F.col("airline").isNull()
    | F.col("origin_airport").isNull()
    | F.col("destination_airport").isNull()
    | F.col("departure_hour").isNull()
    | F.col("arrival_hour").isNull()
    | F.col("distance_miles").isNull()
    | (F.col("distance_miles") <= 0)
    | F.col("departure_minute").isNull()
    | F.col("arrival_minute").isNull()
).count()

total = ml_df.count()

print("ML dataset records:", total)
print("Invalid feature records:", invalid)
print("Feature columns:", len(ml_df.columns))

assert total == 6879484, "Eligible flight count mismatch"
assert invalid == 0, "Invalid ML features detected"

print("ML V1 FEATURE VALIDATION PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Persist the Validated ML Dataset
# MAGIC
# MAGIC ### Target Delta Table
# MAGIC `flightops_intelligence.ml.flight_delay_features_v1`
# MAGIC
# MAGIC ### Write Strategy
# MAGIC - Create the ML schema if necessary.
# MAGIC - Check whether the target table already exists.
# MAGIC - Write the dataset only if the table does not exist.
# MAGIC - Never overwrite or blindly append duplicate records.
# MAGIC
# MAGIC ### Post-Write Validation
# MAGIC Verify:
# MAGIC 1. Total record count.
# MAGIC 2. Monthly flight distribution.
# MAGIC 3. Monthly delayed-arrival percentage.
# MAGIC 4. Successful Delta table creation.
# MAGIC
# MAGIC ### Expected Outcome
# MAGIC A reusable feature dataset ready for chronological
# MAGIC train/validation/test splitting and MLflow experimentation.
# MAGIC
# MAGIC **V2 compatibility:** Weather features will be added in a
# MAGIC separate dataset without replacing V1.

# COMMAND ----------

from pyspark.sql import functions as F

ML_TABLE = "flightops_intelligence.ml.flight_delay_features_v1"

spark.sql("CREATE SCHEMA IF NOT EXISTS flightops_intelligence.ml")

# Prevent overwrites and duplicate writes
if spark.catalog.tableExists(ML_TABLE):
    existing_count = spark.table(ML_TABLE).count()

    print(f"ML table already exists: {ML_TABLE}")
    print(f"Existing records: {existing_count:,}")
    print("No new data written. Review the existing table.")

else:
    assert ml_df.count() == 6879484, "Unexpected ML row count"

    (
        ml_df.write
        .format("delta")
        .mode("append")
        .saveAsTable(ML_TABLE)
    )

    print("ML V1 table created successfully.")

# Validate persisted data if the table exists
saved = spark.table(ML_TABLE)

print("Saved ML records:", saved.count())
print("ML table:", ML_TABLE)

display(
    saved.groupBy("month")
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("delay_label") * 100, 2)
            .alias("delay_rate_pct")
    )
    .orderBy("month")
)

# COMMAND ----------

