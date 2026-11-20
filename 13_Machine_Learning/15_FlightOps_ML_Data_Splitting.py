# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 13.4 — Chronological Data Splitting
# MAGIC
# MAGIC Split ML V1 into training (Jan–Aug), validation (Sep–Oct), and testing (Nov–Dec).
# MAGIC
# MAGIC Preserve chronological order, validate class distributions, and keep test data separate from model selection.

# COMMAND ----------

from pyspark.sql import functions as F

SOURCE = "flightops_intelligence.ml.flight_delay_features_v1_history"

df = spark.table(SOURCE)

# Verify date coverage and required columns
assert df.filter(
    F.col("flight_date").isNull()
    | (F.col("flight_date") < F.lit("2025-01-01"))
    | (F.col("flight_date") > F.lit("2025-12-31"))
).count() == 0

assert df.count() == 6879484

# Chronological split
splits = (
    df.withColumn(
        "dataset_split",
        F.when(
            F.col("flight_date") < F.lit("2025-09-01"),
            "train"
        )
        .when(
            F.col("flight_date") < F.lit("2025-11-01"),
            "validation"
        )
        .otherwise("test")
    )
)

# Verify sizes, date ranges and label distribution
summary = (
    splits.groupBy("dataset_split")
    .agg(
        F.count("*").alias("flights"),
        F.min("flight_date").alias("start_date"),
        F.max("flight_date").alias("end_date"),
        F.sum("delay_label").alias("delayed_flights"),
        F.round(
            F.avg("delay_label") * 100, 2
        ).alias("delay_rate_pct")
    )
)

display(summary.orderBy(
    F.when(F.col("dataset_split") == "train", 1)
     .when(F.col("dataset_split") == "validation", 2)
     .otherwise(3)
))

counts = {
    row["dataset_split"]: row["flights"]
    for row in summary.collect()
}

assert set(counts) == {"train", "validation", "test"}
assert sum(counts.values()) == 6879484

train_df = splits.filter(F.col("dataset_split") == "train")
val_df = splits.filter(F.col("dataset_split") == "validation")
test_df = splits.filter(F.col("dataset_split") == "test")

print("Chronological split validation: PASSED")
print("Train:", counts["train"])
print("Validation:", counts["validation"])
print("Test:", counts["test"])

# COMMAND ----------

