# Databricks notebook source
# MAGIC %md
# MAGIC #### Metadata-Driven Semantic Metric Calculator
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "flightops_intelligence"

def calculate_metric(
    domain_id,
    metric_id,
    start_date="2025-07-01",
    end_date="2025-07-31"
):

    # 1. Load approved semantic metadata
    mappings = spark.table(
        f"{CATALOG}.semantic.domain_metric_mapping"
    )

    metrics = spark.table(
        f"{CATALOG}.semantic.metric_definitions"
    )

    products = spark.table(
        f"{CATALOG}.semantic.data_product_catalog"
    )

    # 2. Resolve metric within the requested business domain
    resolved = (
        mappings
        .filter(
            (F.col("domain_id") == domain_id) &
            (F.col("metric_id") == metric_id)
        )
        .join(metrics, "metric_id")
        .join(products, "gold_table")
        .collect()
    )

    if len(resolved) != 1:
        raise ValueError(
            "Metric not found or not mapped to this domain"
        )

    metric = resolved[0]

    # 3. Read only an approved registered Gold table
    allowed_tables = {
        "daily_flight_performance",
        "airport_congestion_metrics",
        "carrier_performance"
    }

    gold_table = metric["gold_table"]

    if gold_table not in allowed_tables:
        raise ValueError("Unapproved Gold table")

    df_gold = (
        spark.table(f"{CATALOG}.gold.{gold_table}")
        .filter(
            (F.col("flight_date") >= F.to_date(F.lit(start_date))) &
            (F.col("flight_date") <= F.to_date(F.lit(end_date)))
        )
    )

    numerator = metric["numerator_column"]
    denominator = metric["denominator_column"]
    rule = metric["aggregation_rule"]

    # 4. Validate referenced columns
    referenced_columns = [numerator]

    if rule == "RATIO_OF_SUMS":
        referenced_columns.append(denominator)

    if any(
        col_name not in df_gold.columns
        for col_name in referenced_columns
    ):
        raise ValueError("Metric references an invalid Gold column")

    # 5. Execute the approved aggregation
    if rule == "SUM":
        expression = F.sum(F.col(numerator))

    elif rule == "RATIO_OF_SUMS":

        if not denominator:
            raise ValueError("Denominator is required")

        numerator_sum = F.sum(F.col(numerator))
        denominator_sum = F.sum(F.col(denominator))

        expression = F.when(
            denominator_sum != 0,
            numerator_sum / denominator_sum
        )

        if metric["unit"] == "percent":
            expression = expression * 100

    else:
        raise ValueError("Unsupported aggregation rule")

    # 6. Return calculated KPI
    result = df_gold.agg(
        F.round(expression, 2).alias("metric_value")
    ).first()

    return {
        "domain_id": domain_id,
        "metric_id": metric_id,
        "metric_name": metric["metric_name"],
        "value": result["metric_value"],
        "unit": metric["unit"],
        "gold_table": gold_table,
        "start_date": start_date,
        "end_date": end_date
    }

print("Semantic Metric Calculator initialized!")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Testing

# COMMAND ----------

# Total Flights
result_1 = calculate_metric(
    domain_id="OPS",
    metric_id="M001"
)

print("Total Flights:")
print(result_1)

# Cancellation Rate
result_2 = calculate_metric(
    domain_id="OPS",
    metric_id="M002"
)

print("\nCancellation Rate:")
print(result_2)

# On-Time Arrival Rate
result_3 = calculate_metric(
    domain_id="OPS",
    metric_id="M003"
)

print("\nOn-Time Arrival Rate:")
print(result_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Measure Semantic Query Execution Time
# MAGIC
# MAGIC

# COMMAND ----------

import time

metrics_to_test = ["M001", "M002", "M003"]

for metric_id in metrics_to_test:

    start = time.perf_counter()

    result = calculate_metric(
        domain_id="OPS",
        metric_id=metric_id
    )

    elapsed = time.perf_counter() - start

    print(
        f"{metric_id} | "
        f"{result['metric_name']} | "
        f"Value: {result['value']} | "
        f"Execution: {elapsed:.2f} seconds"
    )

# COMMAND ----------

# Multi-Metric Semantic Calculator

import time
from pyspark.sql import functions as F

def calculate_multiple_metrics(
    domain_id,
    metric_ids,
    start_date="2025-07-01",
    end_date="2025-07-31"
):
    start = time.perf_counter()

    if not metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise ValueError("Metric IDs must be non-empty and unique")

    # 1. Resolve all requested metrics together
    metadata = (
        spark.table(
            "flightops_intelligence.semantic.domain_metric_mapping"
        )
        .filter(F.col("domain_id") == domain_id)
        .join(
            spark.table(
                "flightops_intelligence.semantic.metric_definitions"
            ),
            "metric_id"
        )
        .join(
            spark.table(
                "flightops_intelligence.semantic.data_product_catalog"
            ),
            "gold_table"
        )
        .filter(F.col("metric_id").isin(metric_ids))
        .collect()
    )

    if len(metadata) != len(metric_ids):
        raise ValueError("Missing or invalid metric mappings")

    # 2. MVP supports metrics from one Gold table per request
    gold_tables = {row["gold_table"] for row in metadata}

    if len(gold_tables) != 1:
        raise ValueError(
            "This version supports one Gold table per request"
        )

    gold_table = next(iter(gold_tables))

    allowed_tables = {
        "daily_flight_performance",
        "airport_congestion_metrics",
        "carrier_performance"
    }

    if gold_table not in allowed_tables:
        raise ValueError("Unapproved Gold table")

    df_gold = (
        spark.table(
            f"flightops_intelligence.gold.{gold_table}"
        )
        .filter(
            F.col("flight_date").between(start_date, end_date)
        )
    )

    # 3. Build all aggregation expressions
    expressions = []

    for metric in metadata:

        numerator = metric["numerator_column"]
        denominator = metric["denominator_column"]
        rule = metric["aggregation_rule"]

        required_columns = [numerator]

        if rule == "RATIO_OF_SUMS":
            if not denominator:
                raise ValueError("Missing denominator")
            required_columns.append(denominator)
        elif rule != "SUM":
            raise ValueError("Unsupported aggregation rule")

        if any(c not in df_gold.columns for c in required_columns):
            raise ValueError("Invalid metric column")

        numerator_sum = F.sum(F.col(numerator))

        if rule == "SUM":
            calculation = numerator_sum

        else:
            denominator_sum = F.sum(F.col(denominator))

            calculation = F.when(
                denominator_sum != 0,
                numerator_sum / denominator_sum
            )

            if metric["unit"] == "percent":
                calculation = calculation * 100

        expressions.append(
            F.round(calculation, 2).alias(metric["metric_id"])
        )

    # 4. Execute ONE aggregation
    results = df_gold.agg(*expressions).first()

    elapsed = time.perf_counter() - start

    print(f"Metrics calculated: {len(metadata)}")
    print(f"Gold table: {gold_table}")
    print(f"Total execution time: {elapsed:.2f} seconds")

    return {
        metric["metric_id"]: {
            "metric_name": metric["metric_name"],
            "value": results[metric["metric_id"]],
            "unit": metric["unit"]
        }
        for metric in metadata
    }

print("Multi-Metric Semantic Calculator initialized!")

# COMMAND ----------

# Test Multi-Metric Calculation

results = calculate_multiple_metrics(
    domain_id="OPS",
    metric_ids=["M001", "M002", "M003"]
)

for metric_id in sorted(results):
    metric = results[metric_id]

    print(
        f"{metric_id} | "
        f"{metric['metric_name']} | "
        f"{metric['value']} {metric['unit']}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Independent Gold-to-Semantic Validation
# MAGIC
# MAGIC

# COMMAND ----------

# Reference calculations directly against Gold
df_expected = spark.sql("""
    SELECT
        SUM(total_flights) AS M001,

        ROUND(
            SUM(cancelled_flights) * 100.0
            / NULLIF(SUM(total_flights), 0),
            2
        ) AS M002,

        ROUND(
            SUM(on_time_arrivals) * 100.0
            / NULLIF(SUM(completed_flights), 0),
            2
        ) AS M003

    FROM flightops_intelligence.gold.daily_flight_performance

    WHERE flight_date BETWEEN
        DATE '2025-07-01'
        AND DATE '2025-07-31'
""")

expected = df_expected.first()


import math

for metric_id in ["M001", "M002", "M003"]:

    expected_value = expected[metric_id]
    actual_value = results[metric_id]["value"]

    print(
        f"{metric_id} | "
        f"Gold: {expected_value} | "
        f"Semantic Engine: {actual_value}"
    )

    # Compare numeric values with a small tolerance
    assert expected_value is not None
    assert actual_value is not None

    assert math.isclose(
        float(expected_value),
        float(actual_value),
        rel_tol=0,
        abs_tol=1e-9
    ), f"Mismatch found for {metric_id}"

print("\nAll semantic metric calculations validated!")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Metadata-Driven Ranking Engine

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import date

def rank_metric(
    domain_id,
    metric_id,
    group_by_dimension,
    start_date="2025-07-01",
    end_date="2025-07-31",
    top_n=5,
    min_observations=100
):

    # 1. Validate request parameters
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("Start date must be before end date")

    if not 1 <= top_n <= 50:
        raise ValueError("top_n must be between 1 and 50")

    if min_observations < 1:
        raise ValueError("min_observations must be positive")

    # 2. Resolve approved semantic metadata
    metadata = (
        spark.table(
            "flightops_intelligence.semantic.domain_metric_mapping"
        )
        .filter(
            (F.col("domain_id") == domain_id) &
            (F.col("metric_id") == metric_id)
        )
        .join(
            spark.table(
                "flightops_intelligence.semantic.metric_definitions"
            ),
            "metric_id"
        )
        .join(
            spark.table(
                "flightops_intelligence.semantic.data_product_catalog"
            ),
            "gold_table"
        )
        .collect()
    )

    if len(metadata) != 1:
        raise ValueError("Invalid domain or metric")

    metric = metadata[0]

    # 3. Validate approved Gold table and dimension
    approved_tables = {
        "daily_flight_performance",
        "airport_congestion_metrics",
        "carrier_performance"
    }

    gold_table = metric["gold_table"]

    if gold_table not in approved_tables:
        raise ValueError("Unapproved Gold table")

    approved_dimensions = {
        d.strip()
        for d in metric["supported_dimensions"].split(",")
    }

    if group_by_dimension not in approved_dimensions:
        raise ValueError("Dimension not approved for this product")

    df_gold = spark.table(
        f"flightops_intelligence.gold.{gold_table}"
    ).filter(
        F.col("flight_date").between(start, end)
    )

    numerator = metric["numerator_column"]
    denominator = metric["denominator_column"]
    rule = metric["aggregation_rule"]

    # 4. Validate actual source columns
    required_columns = [group_by_dimension, numerator]

    if rule == "RATIO_OF_SUMS":
        if not denominator:
            raise ValueError("Missing metric denominator")
        required_columns.append(denominator)

    if any(c not in df_gold.columns for c in required_columns):
        raise ValueError("Invalid source column")

    # 5. Build aggregation from approved metric definition
    if rule == "RATIO_OF_SUMS":

        grouped = df_gold.groupBy(
            group_by_dimension
        ).agg(
            F.sum(numerator).alias("numerator_total"),
            F.sum(denominator).alias("observation_count")
        )

        multiplier = 100 if metric["unit"] == "percent" else 1

        ranked = (
            grouped
            .filter(
                F.col("observation_count") >= min_observations
            )
            .withColumn(
                "metric_value",
                F.round(
                    F.col("numerator_total") /
                    F.col("observation_count") * multiplier,
                    2
                )
            )
        )

    else:
        raise ValueError(
            "This ranking version supports ratio metrics only"
        )

    # 6. Rank and return top N
    ranked = (
        ranked
        .orderBy(
            F.col("metric_value").desc_nulls_last(),
            F.col(group_by_dimension).asc()
        )
        .limit(top_n)
    )

    print("Metric:", metric["metric_name"])
    print("Gold table:", gold_table)
    print("Grouping dimension:", group_by_dimension)
    print("Minimum observations:", min_observations)

    return ranked.select(
        group_by_dimension,
        "metric_value",
        "observation_count"
    )

print("Semantic Ranking Engine initialized!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Airport Rankings

# COMMAND ----------

df_airport_ranking = rank_metric(
    domain_id="AIRPORT",
    metric_id="M006",
    group_by_dimension="origin_airport",
    start_date="2025-07-01",
    end_date="2025-07-31",
    top_n=5,
    min_observations=100
)

display(df_airport_ranking)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Independent Airport Ranking Validation
# MAGIC

# COMMAND ----------

import math

# 1. Independently calculate rankings using SQL
df_expected_airports = spark.sql("""
    SELECT
        origin_airport,

        ROUND(
            SUM(total_departure_delay_minutes) /
            NULLIF(SUM(departure_delay_observations), 0),
            2
        ) AS metric_value,

        SUM(departure_delay_observations) AS observation_count

    FROM flightops_intelligence.gold.airport_congestion_metrics

    WHERE flight_date BETWEEN
        DATE '2025-07-01'
        AND DATE '2025-07-31'

    GROUP BY origin_airport

    HAVING SUM(departure_delay_observations) >= 100

    ORDER BY
        metric_value DESC,
        origin_airport ASC

    LIMIT 5
""")

# 2. Compare SQL results with Semantic Query Engine
expected_rows = df_expected_airports.collect()
actual_rows = df_airport_ranking.collect()

assert len(expected_rows) == len(actual_rows) == 5

for expected, actual in zip(expected_rows, actual_rows):

    print(
        f"{actual['origin_airport']} | "
        f"SQL: {expected['metric_value']} | "
        f"Semantic: {actual['metric_value']}"
    )

    assert (
        expected["origin_airport"] ==
        actual["origin_airport"]
    ), "Airport ranking mismatch"

    assert math.isclose(
        float(expected["metric_value"]),
        float(actual["metric_value"]),
        rel_tol=0,
        abs_tol=1e-9
    ), "Delay calculation mismatch"

    assert (
        expected["observation_count"] ==
        actual["observation_count"]
    ), "Observation count mismatch"

print("\nAirport Ranking validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Carrier Performance Rankings

# COMMAND ----------

# 1. Airlines with the highest cancellation rates
df_carrier_cancellations = rank_metric(
    domain_id="CARRIER",
    metric_id="M009",
    group_by_dimension="airline",
    start_date="2025-07-01",
    end_date="2025-07-31",
    top_n=5,
    min_observations=100
)

print("Top 5 Airlines by Cancellation Rate")
display(df_carrier_cancellations)


# 2. Airlines with the highest on-time arrival rates
df_carrier_punctuality = rank_metric(
    domain_id="CARRIER",
    metric_id="M010",
    group_by_dimension="airline",
    start_date="2025-07-01",
    end_date="2025-07-31",
    top_n=5,
    min_observations=100
)

print("Top 5 Airlines by On-Time Arrival Rate")
display(df_carrier_punctuality)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metadata-Driven Airline Comparison
# MAGIC
# MAGIC

# COMMAND ----------

from datetime import date
from pyspark.sql import functions as F

def compare_carriers(
    airlines,
    metric_ids=["M009", "M010"],
    start_date="2025-07-01",
    end_date="2025-07-31"
):

    # 1. Validate input
    if not airlines or len(airlines) > 10:
        raise ValueError("Provide between 1 and 10 airlines")

    if len(airlines) != len(set(airlines)):
        raise ValueError("Duplicate airline codes")

    if not metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise ValueError("Provide unique metric IDs")

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("Invalid date range")

    # 2. Resolve carrier metric definitions
    metadata = (
        spark.table(
            "flightops_intelligence.semantic.domain_metric_mapping"
        )
        .filter(F.col("domain_id") == "CARRIER")
        .join(
            spark.table(
                "flightops_intelligence.semantic.metric_definitions"
            ),
            "metric_id"
        )
        .join(
            spark.table(
                "flightops_intelligence.semantic.data_product_catalog"
            ),
            "gold_table"
        )
        .filter(F.col("metric_id").isin(metric_ids))
        .collect()
    )

    if len(metadata) != len(metric_ids):
        raise ValueError("Invalid metric mapping")

    # 3. Validate source data product
    if any(
        m["gold_table"] != "carrier_performance"
        for m in metadata
    ):
        raise ValueError("Unsupported carrier data product")

    df_gold = (
        spark.table(
            "flightops_intelligence.gold.carrier_performance"
        )
        .filter(F.col("flight_date").between(start, end))
        .filter(F.col("airline").isin(airlines))
    )

    # 4. Build calculations dynamically
    expressions = []

    for metric in metadata:

        numerator = metric["numerator_column"]
        denominator = metric["denominator_column"]
        rule = metric["aggregation_rule"]
        metric_id = metric["metric_id"]

        if rule != "RATIO_OF_SUMS" or not denominator:
            raise ValueError("Unsupported calculation rule")

        if (
            numerator not in df_gold.columns or
            denominator not in df_gold.columns
        ):
            raise ValueError("Missing Gold columns")

        numerator_sum = F.sum(F.col(numerator))
        denominator_sum = F.sum(F.col(denominator))

        multiplier = (
            100 if metric["unit"] == "percent" else 1
        )

        expressions.extend([
            F.round(
                F.when(
                    denominator_sum != 0,
                    numerator_sum / denominator_sum * multiplier
                ),
                2
            ).alias(metric_id),

            denominator_sum.alias(
                f"{metric_id}_observations"
            )
        ])

    # 5. One grouped aggregation
    df_comparison = (
        df_gold
        .groupBy("airline")
        .agg(*expressions)
        .orderBy("airline")
    )

    print("Airlines requested:", airlines)
    print("Metrics:", metric_ids)
    print("Gold table: carrier_performance")

    return df_comparison

print("Carrier Comparison Engine initialized!")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Compare American Airline and Delta
# MAGIC

# COMMAND ----------

df_carrier_comparison = compare_carriers(
    airlines=["AA", "DL"],
    metric_ids=["M009", "M010"],
    start_date="2025-07-01",
    end_date="2025-07-31"
)

display(df_carrier_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Independent Carrier Comparison Validation
# MAGIC

# COMMAND ----------

import math

# 1. Independently calculate carrier KPIs using SQL
df_expected_carriers = spark.sql("""
    SELECT
        airline,

        ROUND(
            SUM(cancelled_flights) * 100.0 /
            NULLIF(SUM(total_flights), 0),
            2
        ) AS M009,

        SUM(total_flights) AS M009_observations,

        ROUND(
            SUM(on_time_arrivals) * 100.0 /
            NULLIF(SUM(arrival_delay_observations), 0),
            2
        ) AS M010,

        SUM(arrival_delay_observations) AS M010_observations

    FROM flightops_intelligence.gold.carrier_performance

    WHERE flight_date BETWEEN
        DATE '2025-07-01'
        AND DATE '2025-07-31'

    AND airline IN ('AA', 'DL')

    GROUP BY airline
    ORDER BY airline
""")

# 2. Compare against our Semantic Query Engine
expected_rows = {
    row["airline"]: row
    for row in df_expected_carriers.collect()
}

actual_rows = {
    row["airline"]: row
    for row in df_carrier_comparison.collect()
}

assert set(expected_rows) == set(actual_rows) == {"AA", "DL"}

for airline in ["AA", "DL"]:

    expected = expected_rows[airline]
    actual = actual_rows[airline]

    for metric_id in ["M009", "M010"]:

        sql_value = expected[metric_id]
        semantic_value = actual[metric_id]

        print(
            f"{airline} | {metric_id} | "
            f"SQL: {sql_value} | "
            f"Semantic: {semantic_value}"
        )

        # Compare numeric values with tolerance
        assert math.isclose(
            float(sql_value),
            float(semantic_value),
            rel_tol=0,
            abs_tol=1e-9
        ), f"{airline}: {metric_id} mismatch"

        # Validate denominator / observation count
        observation_col = f"{metric_id}_observations"

        assert (
            expected[observation_col]
            == actual[observation_col]
        ), f"{airline}: {observation_col} mismatch"

print("\nCarrier Comparison validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metadata-Driven Time-Series Engine

# COMMAND ----------

from datetime import date
from pyspark.sql import functions as F

def calculate_metric_trend(
    domain_id,
    metric_id,
    start_date="2025-07-01",
    end_date="2025-07-31",
    grain="day"
):

    # 1. Validate request
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("Invalid date range")

    if grain not in {"day", "month"}:
        raise ValueError("Supported grains: day, month")

    # 2. Resolve semantic metadata
    metadata = (
        spark.table(
            "flightops_intelligence.semantic.domain_metric_mapping"
        )
        .filter(
            (F.col("domain_id") == domain_id) &
            (F.col("metric_id") == metric_id)
        )
        .join(
            spark.table(
                "flightops_intelligence.semantic.metric_definitions"
            ),
            "metric_id"
        )
        .join(
            spark.table(
                "flightops_intelligence.semantic.data_product_catalog"
            ),
            "gold_table"
        )
        .collect()
    )

    if len(metadata) != 1:
        raise ValueError("Invalid domain or metric mapping")

    metric = metadata[0]

    # 3. Validate the approved data product
    approved_tables = {
        "daily_flight_performance",
        "airport_congestion_metrics",
        "carrier_performance"
    }

    gold_table = metric["gold_table"]

    if gold_table not in approved_tables:
        raise ValueError("Unapproved Gold table")

    dimensions = {
        d.strip()
        for d in metric["supported_dimensions"].split(",")
    }

    if "flight_date" not in dimensions:
        raise ValueError("Date dimension not supported")

    df_gold = (
        spark.table(
            f"flightops_intelligence.gold.{gold_table}"
        )
        .filter(F.col("flight_date").between(start, end))
    )

    # 4. Select reporting grain
    if grain == "day":
        period = F.col("flight_date")
    else:
        period = F.trunc(F.col("flight_date"), "month")

    df_gold = df_gold.withColumn(
        "period_start", period
    )

    # 5. Read approved calculation metadata
    numerator = metric["numerator_column"]
    denominator = metric["denominator_column"]
    rule = metric["aggregation_rule"]

    required_columns = [numerator]

    if rule == "RATIO_OF_SUMS":
        if not denominator:
            raise ValueError("Missing denominator")
        required_columns.append(denominator)
    elif rule != "SUM":
        raise ValueError("Unsupported aggregation rule")

    if any(c not in df_gold.columns for c in required_columns):
        raise ValueError("Metric references missing Gold columns")

    # 6. Aggregate by date or month
    aggregations = [
        F.sum(F.col(numerator)).alias("numerator_total")
    ]

    if rule == "RATIO_OF_SUMS":
        aggregations.append(
            F.sum(F.col(denominator)).alias("denominator_total")
        )
    else:
        aggregations.append(
            F.lit(None).cast("double").alias("denominator_total")
        )

    df_trend = df_gold.groupBy(
        "period_start"
    ).agg(*aggregations)

    # 7. Apply approved aggregation rule
    if rule == "SUM":
        calculation = F.col("numerator_total")

    else:
        multiplier = (
            100 if metric["unit"] == "percent" else 1
        )

        calculation = F.when(
            F.col("denominator_total") != 0,
            F.col("numerator_total") /
            F.col("denominator_total") * multiplier
        )

        calculation = F.round(calculation, 2)

    df_trend = (
        df_trend
        .withColumn("metric_value", calculation)
        .select(
            "period_start",
            "metric_value",
            "numerator_total",
            "denominator_total"
        )
        .orderBy("period_start")
    )

    print("Metric:", metric["metric_name"])
    print("Gold table:", gold_table)
    print("Reporting grain:", grain)
    print("Date range:", start_date, "to", end_date)

    return df_trend

print("Semantic Time-Series Engine initialized!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Daily Arrival Punctuality Trend

# COMMAND ----------

df_punctuality_trend = calculate_metric_trend(
    domain_id="OPS",
    metric_id="M003",
    start_date="2025-07-01",
    end_date="2025-07-31",
    grain="day"
)

print("Daily trend records:", df_punctuality_trend.count())

display(df_punctuality_trend)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Independent Daily Trend Validation
# MAGIC

# COMMAND ----------

import math

# Independent SQL calculation
df_expected_trend = spark.sql("""
    SELECT
        flight_date AS period_start,

        ROUND(
            SUM(on_time_arrivals) * 100.0 /
            NULLIF(SUM(completed_flights), 0),
            2
        ) AS metric_value,

        SUM(on_time_arrivals) AS numerator_total,

        SUM(completed_flights) AS denominator_total

    FROM flightops_intelligence.gold.daily_flight_performance

    WHERE flight_date BETWEEN
        DATE '2025-07-01'
        AND DATE '2025-07-31'

    GROUP BY flight_date
    ORDER BY flight_date
""")

expected_rows = {
    row["period_start"]: row
    for row in df_expected_trend.collect()
}

actual_rows = {
    row["period_start"]: row
    for row in df_punctuality_trend.collect()
}

# Validate all dates are present
assert len(expected_rows) == len(actual_rows) == 31
assert set(expected_rows) == set(actual_rows)

# Validate values for every day
for day in sorted(expected_rows):

    expected = expected_rows[day]
    actual = actual_rows[day]

    assert math.isclose(
        float(expected["metric_value"]),
        float(actual["metric_value"]),
        rel_tol=0,
        abs_tol=1e-9
    ), f"Percentage mismatch on {day}"

    assert (
        expected["numerator_total"]
        == actual["numerator_total"]
    ), f"Numerator mismatch on {day}"

    assert (
        expected["denominator_total"]
        == actual["denominator_total"]
    ), f"Denominator mismatch on {day}"

print("Daily records validated:", len(expected_rows))
print("Metric: M003 — On-Time Arrival Rate")
print("All daily percentages: MATCH")
print("All numerators: MATCH")
print("All denominators: MATCH")

print("\nDaily Time-Series validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monthly On-Time Arrival Performance

# COMMAND ----------

df_monthly_punctuality = calculate_metric_trend(
    domain_id="OPS",
    metric_id="M003",
    start_date="2025-07-01",
    end_date="2025-07-31",
    grain="month"
)

print("Monthly trend records:", df_monthly_punctuality.count())

display(df_monthly_punctuality)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Validate Monthly Rollup

# COMMAND ----------

import math
from pyspark.sql import functions as F

daily_totals = df_punctuality_trend.agg(
    F.sum("numerator_total").alias("numerator"),
    F.sum("denominator_total").alias("denominator")
).first()

monthly = df_monthly_punctuality.first()

expected_rate = round(
    daily_totals["numerator"] /
    daily_totals["denominator"] * 100,
    2
)

print("Daily records:", df_punctuality_trend.count())
print("Monthly records:", df_monthly_punctuality.count())
print("Total on-time arrivals:", daily_totals["numerator"])
print("Total completed flights:", daily_totals["denominator"])
print("Expected monthly rate:", expected_rate)
print("Semantic monthly rate:", monthly["metric_value"])

assert df_monthly_punctuality.count() == 1

assert (
    daily_totals["numerator"]
    == monthly["numerator_total"]
)

assert (
    daily_totals["denominator"]
    == monthly["denominator_total"]
)

assert math.isclose(
    float(expected_rate),
    float(monthly["metric_value"]),
    rel_tol=0,
    abs_tol=1e-9
)

print("\nMonthly Time-Series validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Semantic Query Engine Integration Test

# COMMAND ----------

import math
from pyspark.sql import functions as F

checks = []

def validate(name, condition):
    assert condition, f"FAILED: {name}"
    checks.append(name)
    print(f"PASS: {name}")


# 1. Multi-metric calculation
kpis = calculate_multiple_metrics(
    domain_id="OPS",
    metric_ids=["M001", "M002", "M003"]
)

validate(
    "Total Flights",
    kpis["M001"]["value"] == 631428
)

validate(
    "Cancellation Rate",
    math.isclose(kpis["M002"]["value"], 2.45, abs_tol=1e-9)
)

validate(
    "On-Time Arrival Rate",
    math.isclose(kpis["M003"]["value"], 71.11, abs_tol=1e-9)
)


# 2. Airport rankings
airports = df_airport_ranking.collect()

validate(
    "Airport Ranking - 5 results",
    len(airports) == 5
)

validate(
    "Airport Ranking - Observation threshold",
    all(row["observation_count"] >= 100 for row in airports)
)

validate(
    "Airport Ranking - Descending order",
    all(
        airports[i]["metric_value"] >= airports[i + 1]["metric_value"]
        for i in range(len(airports) - 1)
    )
)


# 3. Carrier comparisons
carriers = {
    row["airline"]: row
    for row in df_carrier_comparison.collect()
}

validate(
    "Carrier Comparison - AA and DL",
    set(carriers.keys()) == {"AA", "DL"}
)

validate(
    "Carrier Comparison - Delta punctuality",
    math.isclose(carriers["DL"]["M010"], 75.16, abs_tol=1e-9)
)

validate(
    "Carrier Comparison - American cancellations",
    math.isclose(carriers["AA"]["M009"], 3.37, abs_tol=1e-9)
)


# 4. Time-series validation
daily_count = df_punctuality_trend.count()
monthly_count = df_monthly_punctuality.count()

validate(
    "Daily Trend - 31 days",
    daily_count == 31
)

validate(
    "Monthly Trend - 1 month",
    monthly_count == 1
)


# 5. Daily-to-monthly reconciliation
daily_totals = df_punctuality_trend.agg(
    F.sum("numerator_total").alias("numerator"),
    F.sum("denominator_total").alias("denominator")
).first()

monthly = df_monthly_punctuality.first()

validate(
    "Daily-to-Monthly Numerator",
    daily_totals["numerator"] == monthly["numerator_total"]
)

validate(
    "Daily-to-Monthly Denominator",
    daily_totals["denominator"] == monthly["denominator_total"]
)

validate(
    "Monthly KPI consistency",
    math.isclose(
        float(monthly["metric_value"]),
        float(kpis["M003"]["value"]),
        abs_tol=1e-9
    )
)


# Final result
print("\n" + "=" * 50)
print("SEMANTIC QUERY ENGINE - INTEGRATION SUMMARY")
print("=" * 50)

print("Checks passed:", len(checks))
print("Checks failed: 0")
print("Status: ALL INTEGRATION CHECKS PASSED")