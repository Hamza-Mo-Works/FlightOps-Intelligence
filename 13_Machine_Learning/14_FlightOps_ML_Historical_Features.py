# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 13.3 — Historical Feature Engineering
# MAGIC
# MAGIC Build prior-month airline, airport and route delay features from ML V1.
# MAGIC
# MAGIC - Use only earlier months to prevent target leakage.
# MAGIC - Preserve the original ML V1 dataset.
# MAGIC - Keep historical observation counts for model reliability.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

SOURCE = "flightops_intelligence.ml.flight_delay_features_v1"
TARGET = "flightops_intelligence.ml.flight_delay_features_v1_history"

base = spark.table(SOURCE)

# All 12 reporting months
months = spark.range(1, 13).select(
    F.col("id").cast("int").alias("month")
)

def build_history(df, entity, prefix):
    monthly = (
        df.groupBy(entity, "month")
        .agg(
            F.count("*").alias("_flights"),
            F.sum("delay_label").alias("_delays")
        )
    )

    # Include every month for every entity, even months with no flights
    grid = (
        df.select(entity).distinct()
        .crossJoin(months)
        .join(monthly, [entity, "month"], "left")
        .fillna({"_flights": 0, "_delays": 0})
    )

    previous = (
        Window.partitionBy(entity)
        .orderBy("month")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    return (
        grid
        .withColumn(
            f"{prefix}_prior_flights",
            F.coalesce(F.sum("_flights").over(previous), F.lit(0))
        )
        .withColumn(
            f"{prefix}_prior_delay_rate",
            F.when(
                F.col(f"{prefix}_prior_flights") > 0,
                F.sum("_delays").over(previous)
                / F.col(f"{prefix}_prior_flights")
            )
        )
        .select(
            entity, "month",
            f"{prefix}_prior_flights",
            f"{prefix}_prior_delay_rate"
        )
    )

airline_history = build_history(base, "airline", "airline")
airport_history = build_history(base, "origin_airport", "airport")
route_history = build_history(base, "route", "route")

enriched = (
    base
    .join(airline_history, ["airline", "month"], "left")
    .join(airport_history, ["origin_airport", "month"], "left")
    .join(route_history, ["route", "month"], "left")
)

# Verify row preservation
assert enriched.count() == 6879484

# January must not contain historical flight data
january_leakage = enriched.filter(
    (F.col("month") == 1)
    & (
        (F.col("airline_prior_flights") != 0)
        | (F.col("airport_prior_flights") != 0)
        | (F.col("route_prior_flights") != 0)
    )
).count()

assert january_leakage == 0, (
    f"Historical leakage detected: {january_leakage} rows"
)

print("Enriched ML rows:", enriched.count())
print("Feature columns:", len(enriched.columns))
print("Historical leakage check: PASSED")

# Create only if not already present
if spark.catalog.tableExists(TARGET):
    print("Target table already exists; no write performed.")
else:
    (
        enriched.write
        .format("delta")
        .mode("append")
        .saveAsTable(TARGET)
    )
    print("Historical feature table created.")

print("Saved records:", spark.table(TARGET).count())

display(
    spark.table(TARGET)
    .filter(F.col("month").isin(1, 7, 12))
    .groupBy("month")
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("airline_prior_delay_rate") * 100, 2)
            .alias("avg_historical_airline_delay_pct")
    )
    .orderBy("month")
)

# COMMAND ----------

