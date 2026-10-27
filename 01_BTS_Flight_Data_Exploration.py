# Databricks notebook source

from pyspark.sql import functions as F

# Original July 2025 BTS dataset
file_path = (
    "/Volumes/flightops_intelligence/landing/raw_files/bts/"
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_7.csv"
)

# Read source file without modifying it
df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("mode", "PERMISSIVE")
    .csv(file_path)
)

# Inspect dataset structure
print("Total columns:", len(df_raw.columns))
df_raw.printSchema()

display(df_raw.limit(10))

# COMMAND ----------

