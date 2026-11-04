# Databricks notebook source
# MAGIC %md
# MAGIC # FlightOps Intelligence — Reusable Semantic Query Functions
# MAGIC
# MAGIC Extracted **without changing the metric-calculation implementations** from `06_Semantic_Query_Engine.ipynb`. This notebook contains function definitions only (no test queries or table modifications).
# MAGIC
# MAGIC Import it from `07_FlightOps_AI_Context_Layer` when both notebooks are in the same Databricks workspace directory with a **standalone** notebook cell:
# MAGIC
# MAGIC ```python
# MAGIC %run ./06A_Semantic_Query_Functions
# MAGIC ```
# MAGIC
# MAGIC All functions use `spark` and Unity Catalog tables available in Databricks. See the original notebook for benchmark and validation tests.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single metric — copied from original cell 2

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
# MAGIC ## Multiple metrics — copied from original cell 7

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

# MAGIC %md
# MAGIC ## Ranking — copied from original cell 12

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
# MAGIC ## Carrier comparison — copied from original cell 20

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
# MAGIC ## Time trends — copied from original cell 26

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