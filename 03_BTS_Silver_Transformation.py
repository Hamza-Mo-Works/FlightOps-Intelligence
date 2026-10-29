# Databricks notebook source
# MAGIC %md
# MAGIC ### Read Bronze and establish a baseline

# COMMAND ----------


from pyspark.sql import functions as F
from pyspark.sql.window import Window

source_table = "flightops_intelligence.bronze.bts_flights_raw"
target_table = "flightops_intelligence.silver.bts_flights_clean"

df_bronze = spark.table(source_table)

bronze_count = df_bronze.count()

print("Bronze records:", bronze_count)
print("Bronze columns:", len(df_bronze.columns))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying explicit data types

# COMMAND ----------


df_typed = df_bronze.select(
    F.to_date(
        F.col("FlightDate"),
        "M/d/yyyy"
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

    F.col("Cancelled").cast("int").alias("cancelled"),
    F.col("Diverted").cast("int").alias("diverted"),
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

print("Typed Silver columns:", len(df_typed.columns))
df_typed.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Add operational flags and quality checks

# COMMAND ----------


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

display(
    df_validated.groupBy("is_valid")
    .count()
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Quarantine invalid records

# COMMAND ----------


df_rejected = df_validated.filter(~F.col("is_valid"))

rejected_count = df_rejected.count()
print("Rejected records:", rejected_count)

if rejected_count > 0:
    (
        df_rejected.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(
            "flightops_intelligence.silver.bts_flights_rejected"
        )
    )

df_valid = df_validated.filter(F.col("is_valid"))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Detect and remove duplicate flights

# COMMAND ----------


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

display(duplicate_groups.limit(20))

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

print("Silver records after deduplication:", df_deduplicated.count())


# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing-value analysis

# COMMAND ----------

columns_to_check = [
    "flight_date",
    "airline",
    "flight_number",
    "tail_number",
    "origin_airport",
    "destination_airport",
    "scheduled_departure_time",
    "departure_delay_minutes",
    "arrival_delay_minutes",
    "carrier_delay_minutes",
    "weather_delay_minutes",
    "nas_delay_minutes",
    "security_delay_minutes",
    "late_aircraft_delay_minutes"
]

total_records = df_deduplicated.count()

missing_stats = df_deduplicated.select([
    F.sum(
        F.when(
            F.col(c).isNull() |
            (F.trim(F.col(c).cast("string")) == ""),
            1
        ).otherwise(0)
    ).alias(c)
    for c in columns_to_check
])

missing_counts = missing_stats.collect()[0].asDict()

results = [
    (
        col_name,
        int(missing_count),
        round(missing_count * 100 / total_records, 2)
    )
    for col_name, missing_count in missing_counts.items()
]

df_missing_report = spark.createDataFrame(
    results,
    ["column_name", "missing_count", "missing_percentage"]
)

display(
    df_missing_report.orderBy(
        F.desc("missing_percentage")
    )
)

# COMMAND ----------

display(
    df_deduplicated.groupBy(
        "cancelled",
        "diverted"
    ).agg(
        F.count("*").alias("total_flights"),

        F.sum(
            F.col("departure_delay_minutes").isNull().cast("int")
        ).alias("missing_departure_delay"),

        F.sum(
            F.col("arrival_delay_minutes").isNull().cast("int")
        ).alias("missing_arrival_delay")
    ).orderBy("cancelled", "diverted")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write the Silver Delta table

# COMMAND ----------


(
    df_deduplicated.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(target_table)
)

print("Silver table created:", target_table)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Silver

# COMMAND ----------


df_silver = spark.table(target_table)

print("Bronze records:", bronze_count)
print("Silver records:", df_silver.count())
print("Silver columns:", len(df_silver.columns))

display(
    df_silver.groupBy(
        "cancelled",
        "diverted"
    ).count()
)

display(
    df_silver.select(
        "flight_date",
        "airline",
        "flight_number",
        "origin_airport",
        "destination_airport",
        "departure_delay_minutes",
        "arrival_delay_minutes",
        "is_arrival_delayed_15"
    ).limit(10)
)


# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL flightops_intelligence.silver.bts_flights_clean;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY flightops_intelligence.silver.bts_flights_clean;

# COMMAND ----------

from pyspark.sql import functions as F

BRONZE = "flightops_intelligence.bronze.bts_flights_raw"
SILVER = "flightops_intelligence.silver.bts_flights_clean"

bronze = spark.table(BRONZE)
silver = spark.table(SILVER)

print("Bronze rows:", bronze.count())
print("Existing Silver rows:", silver.count())
print("Existing Silver columns:", len(silver.columns))

print("\nSilver schema:")
silver.printSchema()

print("\nNormalized Bronze date coverage:")
bronze.selectExpr("""
    COALESCE(
        CAST(TRY_TO_TIMESTAMP(FlightDate, 'yyyy-MM-dd') AS DATE),
        CAST(TRY_TO_TIMESTAMP(FlightDate, 'M/d/yyyy') AS DATE)
    ) AS parsed_date
""").agg(
    F.min("parsed_date").alias("first_date"),
    F.max("parsed_date").alias("last_date"),
    F.countDistinct("parsed_date").alias("reporting_days"),
    F.sum(F.col("parsed_date").isNull().cast("int")).alias("invalid_dates")
).show()

# COMMAND ----------

