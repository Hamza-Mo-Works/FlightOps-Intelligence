# Databricks notebook source
# MAGIC %md
# MAGIC ### Read table

# COMMAND ----------

from pyspark.sql import functions as F

source_table = "flightops_intelligence.silver.bts_flights_clean"

df_silver = spark.table(source_table)

print("Silver records:", df_silver.count())
print("Silver columns:", len(df_silver.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Daily Flight Performance

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

    .orderBy("flight_date")
)


display(df_daily)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Gold against Silver

# COMMAND ----------

# 1. Calculate expected totals directly from Silver
silver_metrics = df_silver.agg(
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
    ).alias("completed_flights")
).first()


# 2. Calculate actual totals from Gold
gold_metrics = df_daily.agg(
    F.sum("total_flights").alias("total_flights"),
    F.sum("cancelled_flights").alias("cancelled_flights"),
    F.sum("diverted_flights").alias("diverted_flights"),
    F.sum("completed_flights").alias("completed_flights")
).first()


# 3. Compare Silver and Gold
metrics_to_validate = [
    "total_flights",
    "cancelled_flights",
    "diverted_flights",
    "completed_flights"
]

for metric in metrics_to_validate:

    silver_value = silver_metrics[metric]
    gold_value = gold_metrics[metric]

    print(
        f"{metric}: Silver={silver_value}, Gold={gold_value}"
    )

    assert silver_value == gold_value, (
        f"Reconciliation failed for {metric}: "
        f"Silver={silver_value}, Gold={gold_value}"
    )


# 4. Validate daily business rules
invalid_days = df_daily.filter(
    F.col("total_flights") != (
        F.col("completed_flights") +
        F.col("cancelled_flights") +
        F.col("diverted_flights")
    )
).count()

assert invalid_days == 0, (
    f"Found {invalid_days} days with inconsistent flight counts"
)

print("All Gold reconciliation checks passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write table

# COMMAND ----------

target_table = (
    "flightops_intelligence.gold.daily_flight_performance"
)

(
    df_daily.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(target_table)
)

print("Gold table created:", target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify table

# COMMAND ----------

df_gold = spark.table(
    "flightops_intelligence.gold.daily_flight_performance"
)

print("Gold records:", df_gold.count())
print("Gold columns:", len(df_gold.columns))

display(
    df_gold.orderBy("flight_date")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Airport Congestion Metrics

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

    .orderBy(
        "flight_date",
        "origin_airport"
    )
)

print("Airport Gold records:", df_airport.count())

display(df_airport.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Airport Gold Reconciliation

# COMMAND ----------


# 1. Expected values from Silver
silver_airport_metrics = df_silver.agg(
    F.count("*").alias("total_departures"),

    F.sum(
        F.when(F.col("cancelled") == 1, 1).otherwise(0)
    ).alias("cancelled_departures"),

    F.count("departure_delay_minutes").alias(
        "departure_delay_observations"
    ),

    F.count("taxi_out_minutes").alias(
        "taxi_out_observations"
    )
).first()


# 2. Actual aggregated values from Airport Gold
gold_airport_metrics = df_airport.agg(
    F.sum("total_departures").alias("total_departures"),

    F.sum("cancelled_departures").alias(
        "cancelled_departures"
    ),

    F.sum("departure_delay_observations").alias(
        "departure_delay_observations"
    ),

    F.sum("taxi_out_observations").alias(
        "taxi_out_observations"
    )
).first()


# 3. Compare Silver against Gold
metrics_to_validate = [
    "total_departures",
    "cancelled_departures",
    "departure_delay_observations",
    "taxi_out_observations"
]

for metric in metrics_to_validate:

    silver_value = silver_airport_metrics[metric]
    gold_value = gold_airport_metrics[metric]

    print(
        f"{metric}: Silver={silver_value}, Gold={gold_value}"
    )

    assert silver_value == gold_value, (
        f"Reconciliation failed for {metric}: "
        f"Silver={silver_value}, Gold={gold_value}"
    )

print("\nAirport Gold reconciliation passed!")

# COMMAND ----------

duplicate_airport_dates = (
    df_airport
    .groupBy("flight_date", "origin_airport")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_count = duplicate_airport_dates.count()

print("Duplicate airport/date combinations:", duplicate_count)

assert duplicate_count == 0, (
    "Airport Gold contains duplicate airport/date combinations!"
)

print("Airport Gold grain validation passed!")

# COMMAND ----------

invalid_airport_records = df_airport.filter(
    (F.col("cancelled_departures") > F.col("total_departures")) |
    (F.col("cancelled_departures") < 0) |
    (F.col("total_departures") <= 0) |
    (F.col("departure_delay_observations") > F.col("total_departures")) |
    (F.col("taxi_out_observations") > F.col("total_departures"))
)

invalid_count = invalid_airport_records.count()

print("Invalid airport records:", invalid_count)

assert invalid_count == 0, (
    "Airport Gold contains invalid operational metrics!"
)

print("Airport Gold business-rule validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Airport Congestion Metrics

# COMMAND ----------

airport_target_table = (
    "flightops_intelligence.gold.airport_congestion_metrics"
)

(
    df_airport.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(airport_target_table)
)

print("Airport Gold table created:", airport_target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Airport Gold table

# COMMAND ----------

df_airport_gold = spark.table(
    "flightops_intelligence.gold.airport_congestion_metrics"
)

print("Airport Gold records:", df_airport_gold.count())
print("Airport Gold columns:", len(df_airport_gold.columns))

display(
    df_airport_gold.orderBy(
        "flight_date",
        "origin_airport"
    ).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gold Product 3 - Carrier Performance

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

    .orderBy("flight_date", "airline")
)

print("Carrier Gold records:", df_carrier.count())
print("Carrier Gold columns:", len(df_carrier.columns))

display(df_carrier.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Carrier Gold Reconciliation
# MAGIC

# COMMAND ----------

# Expected metrics from Silver
silver_metrics = df_silver.agg(
    F.count("*").alias("total_flights"),

    F.sum(F.col("cancelled")).alias("cancelled_flights"),

    F.sum(F.col("diverted")).alias("diverted_flights"),

    F.sum(
        F.when(
            (F.col("cancelled") == 0) &
            (F.col("diverted") == 0), 1
        ).otherwise(0)
    ).alias("completed_flights"),

    F.count("arrival_delay_minutes").alias(
        "arrival_delay_observations"
    ),

    F.count("departure_delay_minutes").alias(
        "departure_delay_observations"
    ),

    F.sum("total_placeholder").alias("unused")
    if False else F.sum(
        F.when(F.col("is_arrival_delayed_15") == False, 1)
        .otherwise(0)
    ).alias("on_time_arrivals")
).first()


# Actual metrics from Carrier Gold
gold_metrics = df_carrier.agg(
    *[
        F.sum(metric).alias(metric)
        for metric in [
            "total_flights",
            "cancelled_flights",
            "diverted_flights",
            "completed_flights",
            "arrival_delay_observations",
            "departure_delay_observations",
            "on_time_arrivals"
        ]
    ]
).first()


# Compare
metrics = [
    "total_flights",
    "cancelled_flights",
    "diverted_flights",
    "completed_flights",
    "arrival_delay_observations",
    "departure_delay_observations",
    "on_time_arrivals"
]

for metric in metrics:
    silver_value = silver_metrics[metric]
    gold_value = gold_metrics[metric]

    print(f"{metric}: Silver={silver_value}, Gold={gold_value}")

    assert silver_value == gold_value, (
        f"Mismatch in {metric}: "
        f"Silver={silver_value}, Gold={gold_value}"
    )

print("\nCarrier Gold reconciliation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Carrier Grain and Business Rules
# MAGIC
# MAGIC

# COMMAND ----------

duplicates = (
    df_carrier
    .groupBy("flight_date", "airline")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

invalid_records = df_carrier.filter(
    (
        F.col("total_flights") !=
        (
            F.col("completed_flights") +
            F.col("cancelled_flights") +
            F.col("diverted_flights")
        )
    ) |
    (F.col("total_flights") <= 0) |
    (F.col("on_time_arrivals") > F.col("completed_flights")) |
    (F.col("arrival_delay_observations") > F.col("total_flights")) |
    (F.col("departure_delay_observations") > F.col("total_flights"))
).count()

print("Duplicate airline/date combinations:", duplicates)
print("Invalid carrier records:", invalid_records)

assert duplicates == 0, "Duplicate airline/date combinations!"
assert invalid_records == 0, "Invalid carrier metrics!"

print("Carrier Gold validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Carrier Performance

# COMMAND ----------

carrier_target_table = (
    "flightops_intelligence.gold.carrier_performance"
)

(
    df_carrier.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(carrier_target_table)
)

print("Carrier Gold table created:", carrier_target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Carrier Gold Table

# COMMAND ----------

df_carrier_gold = spark.table(
    "flightops_intelligence.gold.carrier_performance"
)

print("Carrier Gold records:", df_carrier_gold.count())
print("Carrier Gold columns:", len(df_carrier_gold.columns))

display(
    df_carrier_gold
    .orderBy("flight_date", "airline")
    .limit(20)
)

# COMMAND ----------

gold_tables = [
    "daily_flight_performance",
    "airport_congestion_metrics",
    "carrier_performance"
]

for table in gold_tables:
    df = spark.table(f"flightops_intelligence.gold.{table}")

    print(f"\n=== {table} ===")
    print("Rows:", df.count())
    print("Columns:", len(df.columns))
    print("Schema:")
    df.printSchema()

# COMMAND ----------

