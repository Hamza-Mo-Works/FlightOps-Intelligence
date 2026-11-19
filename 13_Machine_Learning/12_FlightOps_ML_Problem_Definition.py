# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 13.1 — ML Problem Definition & Target Profiling
# MAGIC
# MAGIC ## FlightOps Intelligence | Flight Delay Prediction V1
# MAGIC
# MAGIC ### Objective
# MAGIC Evaluate the feasibility of predicting whether a flight will arrive
# MAGIC 15 minutes or more late using information available before departure.
# MAGIC
# MAGIC ### Data Source
# MAGIC `flightops_intelligence.silver.bts_flights_clean`
# MAGIC
# MAGIC ### Prediction Target
# MAGIC - **1:** Arrival delay >= 15 minutes
# MAGIC - **0:** Arrival delay < 15 minutes
# MAGIC
# MAGIC Only completed, non-diverted flights with valid arrival-delay
# MAGIC observations are eligible for supervised training.
# MAGIC
# MAGIC ### Analysis
# MAGIC 1. Assess target distribution and class imbalance.
# MAGIC 2. Examine monthly arrival-delay patterns.
# MAGIC 3. Evaluate missing values in candidate predictors.
# MAGIC
# MAGIC ### Output
# MAGIC A validated ML problem definition and feature-quality assessment.
# MAGIC
# MAGIC **Future extension:** Weather-enhanced prediction (V2).

# COMMAND ----------

from pyspark.sql import functions as F

SILVER = "flightops_intelligence.silver.bts_flights_clean"

flights = spark.table(SILVER)

# Valid observed arrival outcome
eligible = (
    (F.col("cancelled") == 0) &
    (F.col("diverted") == 0) &
    (F.col("arrival_delay_minutes").isNotNull())
)

ml_profile = (
    flights
    .withColumn("eligible_for_ml", eligible)
    .withColumn(
        "delay_label",
        F.when(
            eligible,
            F.when(
                F.col("arrival_delay_minutes") >= 15, 1
            ).otherwise(0)
        )
    )
)

summary = ml_profile.agg(
    F.count("*").alias("total_flights"),

    F.sum(
        F.col("eligible_for_ml").cast("int")
    ).alias("eligible_flights"),

    F.sum(
        F.when(F.col("cancelled") == 1, 1).otherwise(0)
    ).alias("cancelled_flights"),

    F.sum(
        F.when(F.col("diverted") == 1, 1).otherwise(0)
    ).alias("diverted_flights"),

    F.sum(
        F.when(F.col("delay_label") == 1, 1).otherwise(0)
    ).alias("delayed_flights"),

    F.avg("delay_label").alias("delay_rate")
).first()

print("=== FLIGHTOPS ML TARGET PROFILE ===")
print("Total flights:", summary["total_flights"])
print("Eligible flights:", summary["eligible_flights"])
print("Cancelled flights:", summary["cancelled_flights"])
print("Diverted flights:", summary["diverted_flights"])
print("Delayed flights:", summary["delayed_flights"])
print("Delay rate:", round(summary["delay_rate"] * 100, 2), "%")

print("\n=== MONTHLY LABEL DISTRIBUTION ===")

display(
    ml_profile
    .filter("eligible_for_ml")
    .groupBy("month")
    .agg(
        F.count("*").alias("eligible_flights"),
        F.sum("delay_label").alias("delayed_flights"),
        F.round(F.avg("delay_label") * 100, 2)
            .alias("delay_rate_pct")
    )
    .orderBy("month")
)

print("\n=== CANDIDATE FEATURE COMPLETENESS ===")

candidate_features = [
    "airline",
    "origin_airport",
    "destination_airport",
    "scheduled_departure_time",
    "scheduled_arrival_time",
    "distance_miles",
    "day_of_week",
    "month"
]

display(
    flights.agg(*[
        F.round(
            F.avg(
                F.when(
                    F.col(c).isNull() |
                    (F.trim(F.col(c).cast("string")) == ""),
                    1
                ).otherwise(0)
            ) * 100, 2
        ).alias(c + "_missing_pct")
        for c in candidate_features
    ])
)

# COMMAND ----------

