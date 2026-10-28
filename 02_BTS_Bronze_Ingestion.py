# Databricks notebook source
# MAGIC %md
# MAGIC ### Configure source and destination

# COMMAND ----------


from pyspark.sql import functions as F

catalog = "flightops_intelligence"
schema = "bronze"
table_name = "bts_flights_raw"

source_path = (
    "/Volumes/flightops_intelligence/landing/raw_files/bts/"
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_7.csv"
)

target_table = f"{catalog}.{schema}.{table_name}"

print("Source:", source_path)
print("Target:", target_table)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Read and enrich raw data

# COMMAND ----------


df_bronze = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("mode", "PERMISSIVE")
    .csv(source_path)
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file", F.lit(source_path))
    .withColumn("_source_system", F.lit("US_DOT_BTS"))
    .withColumn("_ingestion_year", F.lit(2025))
    .withColumn("_ingestion_month", F.lit(7))
)

print("Source columns:", 109)
print("Bronze columns:", len(df_bronze.columns))

display(df_bronze.limit(5))

# COMMAND ----------


# Verify source structure before writing Bronze

required_columns = [
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DepDelay",
    "ArrDelay",
    "Cancelled",
    "Diverted"
]

missing_columns = [
    col for col in required_columns
    if col not in df_bronze.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if len(df_bronze.columns) != 114:
    raise ValueError(
        f"Unexpected number of columns: {len(df_bronze.columns)}"
    )

print("Bronze structural validation passed.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Write the Bronze Delta table

# COMMAND ----------


(
    df_bronze.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(target_table)
)

print(f"Bronze table created: {target_table}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate ingestion

# COMMAND ----------


df_check = spark.table(target_table)

print("Total records:", df_check.count())
print("Total columns:", len(df_check.columns))

display(
    df_check.select(
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "DepDelay",
        "ArrDelay",
        "_ingestion_timestamp",
        "_source_system"
    ).limit(10)
)


# COMMAND ----------

