# Databricks notebook source
# MAGIC %md
# MAGIC ### Business Domain Definitions
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

business_domains = [
    (
        "OPS",
        "Airline Operations",
        "Network-wide flight performance, punctuality, "
        "cancellations and diversions"
    ),
    (
        "AIRPORT",
        "Airport Operations",
        "Airport departure performance, taxi-out times, "
        "delays and operational indicators"
    ),
    (
        "CARRIER",
        "Airline Performance",
        "Carrier-level punctuality, cancellation rates "
        "and operational comparisons"
    )
]

df_domains = spark.createDataFrame(
    business_domains,
    [
        "domain_id",
        "domain_name",
        "description"
    ]
)

display(df_domains)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Business Domains
# MAGIC
# MAGIC

# COMMAND ----------

domains_table = (
    "flightops_intelligence.semantic.business_domains"
)

(
    df_domains.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(domains_table)
)

print("Business domains table created:", domains_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify

# COMMAND ----------

df_domains_saved = spark.table(
    "flightops_intelligence.semantic.business_domains"
)

print("Business domains:", df_domains_saved.count())
print("Columns:", len(df_domains_saved.columns))

display(df_domains_saved)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Business Metrics
# MAGIC

# COMMAND ----------

metric_definitions = [
    (
        "M001",
        "Total Flights",
        "Total scheduled flight records",
        "daily_flight_performance",
        "total_flights",
        None,
        "SUM",
        "flights",
        "flight volume, scheduled flights, total flights"
    ),
    (
        "M002",
        "Cancellation Rate",
        "Percentage of flights that were cancelled",
        "daily_flight_performance",
        "cancelled_flights",
        "total_flights",
        "RATIO_OF_SUMS",
        "percent",
        "cancellation percentage, cancelled flights rate"
    ),
    (
        "M003",
        "On-Time Arrival Rate",
        "Percentage of completed flights arriving less than 15 minutes late",
        "daily_flight_performance",
        "on_time_arrivals",
        "completed_flights",
        "RATIO_OF_SUMS",
        "percent",
        "punctuality, OTP, on-time performance"
    ),
    (
        "M004",
        "Average Arrival Delay",
        "Average observed arrival delay in minutes",
        "daily_flight_performance",
        "total_arrival_delay_minutes",
        "arrival_delay_observations",
        "RATIO_OF_SUMS",
        "minutes",
        "arrival delay, average arrival lateness"
    ),
    (
        "M005",
        "Airport Departures",
        "Total departing flights recorded at an origin airport",
        "airport_congestion_metrics",
        "total_departures",
        None,
        "SUM",
        "flights",
        "airport traffic, departure volume"
    ),
    (
        "M006",
        "Average Departure Delay",
        "Average observed departure delay at an airport",
        "airport_congestion_metrics",
        "total_departure_delay_minutes",
        "departure_delay_observations",
        "RATIO_OF_SUMS",
        "minutes",
        "departure delays, average departure lateness"
    ),
    (
        "M007",
        "Average Taxi-Out Time",
        "Average time between gate departure and takeoff",
        "airport_congestion_metrics",
        "total_taxi_out_minutes",
        "taxi_out_observations",
        "RATIO_OF_SUMS",
        "minutes",
        "taxi time, ground movement time, taxi-out"
    ),
    (
        "M008",
        "Airport Departure Delay Rate",
        "Percentage of observed departures delayed by at least 15 minutes",
        "airport_congestion_metrics",
        "delayed_departures_15",
        "departure_delay_observations",
        "RATIO_OF_SUMS",
        "percent",
        "airport departure punctuality, departure delay percentage"
    ),
    (
        "M009",
        "Carrier Cancellation Rate",
        "Percentage of a carrier's scheduled flights that were cancelled",
        "carrier_performance",
        "cancelled_flights",
        "total_flights",
        "RATIO_OF_SUMS",
        "percent",
        "airline cancellations, carrier cancellation percentage"
    ),
    (
        "M010",
        "Carrier On-Time Arrival Rate",
        "Percentage of a carrier's observed arrivals that were on time",
        "carrier_performance",
        "on_time_arrivals",
        "arrival_delay_observations",
        "RATIO_OF_SUMS",
        "percent",
        "airline punctuality, carrier OTP, carrier on-time performance"
    )
]

columns = [
    "metric_id",
    "metric_name",
    "business_definition",
    "gold_table",
    "numerator_column",
    "denominator_column",
    "aggregation_rule",
    "unit",
    "synonyms"
]

df_metrics = spark.createDataFrame(
    metric_definitions,
    columns
)

print("Metric definitions:", df_metrics.count())
print("Columns:", len(df_metrics.columns))

display(df_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Semantic Metric Definitions
# MAGIC

# COMMAND ----------

from pyspark.sql.types import NumericType

# Approved configuration
catalog_name = "flightops_intelligence"

approved_gold_tables = {
    "daily_flight_performance",
    "airport_congestion_metrics",
    "carrier_performance"
}

approved_aggregation_rules = {
    "SUM",
    "RATIO_OF_SUMS"
}

approved_units = {
    "flights",
    "percent",
    "minutes"
}

# Collect only our small metadata catalog (10 records)
metric_rows = df_metrics.collect()

errors = []
seen_ids = set()

for metric in metric_rows:

    metric_id = metric["metric_id"]
    metric_name = metric["metric_name"]
    gold_table = metric["gold_table"]
    numerator = metric["numerator_column"]
    denominator = metric["denominator_column"]
    rule = metric["aggregation_rule"]
    unit = metric["unit"]

    # 1. Required fields
    required_values = [
        metric_id,
        metric_name,
        metric["business_definition"],
        gold_table,
        numerator,
        rule,
        unit
    ]

    if any(
        not isinstance(value, str) or not value.strip()
        for value in required_values
    ):
        errors.append(f"{metric_id}: Missing required field")
        continue

    # 2. Duplicate metric IDs
    if metric_id in seen_ids:
        errors.append(f"{metric_id}: Duplicate metric ID")

    seen_ids.add(metric_id)

    # 3. Approved aggregation rules
    if rule not in approved_aggregation_rules:
        errors.append(f"{metric_id}: Invalid aggregation rule")

    # 4. Approved units
    if unit not in approved_units:
        errors.append(f"{metric_id}: Invalid unit")

    # 5. Ratio requires a denominator
    if rule == "RATIO_OF_SUMS" and not denominator:
        errors.append(f"{metric_id}: Missing denominator")

    if rule == "SUM" and denominator:
        errors.append(f"{metric_id}: SUM should not have denominator")

    # 6. Validate Gold table
    if gold_table not in approved_gold_tables:
        errors.append(f"{metric_id}: Unapproved Gold table")
        continue

    full_table_name = (
        f"{catalog_name}.gold.{gold_table}"
    )

    if not spark.catalog.tableExists(full_table_name):
        errors.append(f"{metric_id}: Gold table does not exist")
        continue

    # 7. Validate actual Gold column names and types
    gold_schema = spark.table(full_table_name).schema
    gold_fields = {
        field.name: field.dataType
        for field in gold_schema.fields
    }

    for column_name in [numerator, denominator]:

        if column_name is None:
            continue

        if column_name not in gold_fields:
            errors.append(
                f"{metric_id}: Column '{column_name}' "
                f"not found in {gold_table}"
            )
        elif not isinstance(
            gold_fields[column_name],
            NumericType
        ):
            errors.append(
                f"{metric_id}: Column '{column_name}' "
                "is not numeric"
            )

# Final validation
print("Metrics checked:", len(metric_rows))
print("Validation errors:", len(errors))

if errors:
    for error in errors:
        print("ERROR:", error)
    raise ValueError("Metric catalog validation failed!")

print("All semantic metric definitions passed validation!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Persist Metric Definitions

# COMMAND ----------

metric_target_table = (
    "flightops_intelligence.semantic.metric_definitions"
)

(
    df_metrics.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(metric_target_table)
)

print("Metric definitions table created:", metric_target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Persisted Metric Definitions

# COMMAND ----------

df_metrics_saved = spark.table(
    "flightops_intelligence.semantic.metric_definitions"
)

print("Metric definitions:", df_metrics_saved.count())
print("Columns:", len(df_metrics_saved.columns))

display(
    df_metrics_saved.orderBy("metric_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Domain-to-Metric Mappings

# COMMAND ----------

domain_metric_mappings = [

    # Airline Operations
    ("OPS", "M001", 1),
    ("OPS", "M002", 2),
    ("OPS", "M003", 3),
    ("OPS", "M004", 4),

    # Airport Operations
    ("AIRPORT", "M005", 1),
    ("AIRPORT", "M006", 2),
    ("AIRPORT", "M007", 3),
    ("AIRPORT", "M008", 4),

    # Airline Performance
    ("CARRIER", "M009", 1),
    ("CARRIER", "M010", 2)
]

df_domain_metric_mapping = spark.createDataFrame(
    domain_metric_mappings,
    [
        "domain_id",
        "metric_id",
        "display_priority"
    ]
)

print("Domain metric mappings:",
      df_domain_metric_mapping.count())

print("Columns:",
      len(df_domain_metric_mapping.columns))

display(
    df_domain_metric_mapping.orderBy(
        "domain_id",
        "display_priority"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Domain-to-Metric Mappings
# MAGIC

# COMMAND ----------

df_domains_saved = spark.table(
    "flightops_intelligence.semantic.business_domains"
)

df_metrics_saved = spark.table(
    "flightops_intelligence.semantic.metric_definitions"
)

# Detect references to missing business domains
invalid_domains = (
    df_domain_metric_mapping
    .join(
        df_domains_saved.select("domain_id"),
        on="domain_id",
        how="left_anti"
    )
    .count()
)

# Detect references to missing metrics
invalid_metrics = (
    df_domain_metric_mapping
    .join(
        df_metrics_saved.select("metric_id"),
        on="metric_id",
        how="left_anti"
    )
    .count()
)

# Detect duplicate domain-metric combinations
duplicates = (
    df_domain_metric_mapping
    .groupBy("domain_id", "metric_id")
    .count()
    .filter("count > 1")
    .count()
)

# Check priority values
invalid_priorities = (
    df_domain_metric_mapping
    .filter("display_priority < 1 OR display_priority IS NULL")
    .count()
)

print("Invalid domains:", invalid_domains)
print("Invalid metrics:", invalid_metrics)
print("Duplicate mappings:", duplicates)
print("Invalid priorities:", invalid_priorities)

assert invalid_domains == 0
assert invalid_metrics == 0
assert duplicates == 0
assert invalid_priorities == 0

print("Domain-to-Metric Mapping validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Domain-to-Metric Mapping

# COMMAND ----------

mapping_target_table = (
    "flightops_intelligence.semantic.domain_metric_mapping"
)

(
    df_domain_metric_mapping.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(mapping_target_table)
)

print("Domain metric mapping table created:", mapping_target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Saved Mapping

# COMMAND ----------

df_mapping_saved = spark.table(
    "flightops_intelligence.semantic.domain_metric_mapping"
)

print("Domain metric mappings:", df_mapping_saved.count())
print("Columns:", len(df_mapping_saved.columns))

display(
    df_mapping_saved
    .orderBy("domain_id", "display_priority")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Business Semantic Layer Relationships

# COMMAND ----------

df_semantic_view = (
    spark.table("flightops_intelligence.semantic.domain_metric_mapping")
    .join(
        spark.table("flightops_intelligence.semantic.business_domains"),
        on="domain_id",
        how="inner"
    )
    .join(
        spark.table("flightops_intelligence.semantic.metric_definitions"),
        on="metric_id",
        how="inner"
    )
    .select(
        "domain_id",
        "domain_name",
        "metric_id",
        "metric_name",
        "gold_table",
        "aggregation_rule",
        "display_priority"
    )
    .orderBy("domain_id", "display_priority")
)

print("Resolved semantic mappings:", df_semantic_view.count())

display(df_semantic_view)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Business Questions

# COMMAND ----------

business_questions = [

    # Airline Operations
    ("Q001", "OPS",
     "How many flights were scheduled?",
     "M001", "AGGREGATION", None),

    ("Q002", "OPS",
     "What percentage of flights were cancelled?",
     "M002", "AGGREGATION", None),

    ("Q003", "OPS",
     "What was our on-time arrival performance?",
     "M003", "AGGREGATION", None),

    ("Q004", "OPS",
     "How did average arrival delays change over time?",
     "M004", "TREND", "flight_date"),

    # Airport Operations
    ("Q005", "AIRPORT",
     "Which airports handled the most departures?",
     "M005", "RANKING", "origin_airport"),

    ("Q006", "AIRPORT",
     "Which airports had the longest departure delays?",
     "M006", "RANKING", "origin_airport"),

    ("Q007", "AIRPORT",
     "What was the average taxi-out time by airport?",
     "M007", "COMPARISON", "origin_airport"),

    ("Q008", "AIRPORT",
     "Which airports had the highest departure delay rates?",
     "M008", "RANKING", "origin_airport"),

    # Airline Performance
    ("Q009", "CARRIER",
     "Which airlines had the highest cancellation rates?",
     "M009", "RANKING", "airline"),

    ("Q010", "CARRIER",
     "Compare on-time arrival performance across airlines",
     "M010", "COMPARISON", "airline"),

    # Same question, different business contexts
    ("Q011", "OPS",
     "How was flight performance?",
     "M003", "AGGREGATION", None),

    ("Q012", "AIRPORT",
     "How was flight performance?",
     "M006", "AGGREGATION", None)
]

question_schema = """
    question_id STRING,
    domain_id STRING,
    example_question STRING,
    primary_metric_id STRING,
    intent_type STRING,
    group_by_dimension STRING
"""

df_questions = spark.createDataFrame(
    business_questions,
    schema=question_schema
)

print("Business questions:", df_questions.count())
print("Columns:", len(df_questions.columns))

display(
    df_questions.orderBy("question_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Business Question Catalog

# COMMAND ----------

from pyspark.sql import functions as F

# Read existing semantic metadata
df_domains = spark.table(
    "flightops_intelligence.semantic.business_domains"
)

df_metrics = spark.table(
    "flightops_intelligence.semantic.metric_definitions"
)

df_mapping = spark.table(
    "flightops_intelligence.semantic.domain_metric_mapping"
)

# 1. Validate question IDs are unique
duplicate_questions = (
    df_questions
    .groupBy("question_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

# 2. Validate referenced business domains
invalid_domains = (
    df_questions
    .join(
        df_domains.select("domain_id"),
        on="domain_id",
        how="left_anti"
    )
    .count()
)

# 3. Validate referenced metrics
invalid_metrics = (
    df_questions
    .join(
        df_metrics.select(
            F.col("metric_id").alias("primary_metric_id")
        ),
        on="primary_metric_id",
        how="left_anti"
    )
    .count()
)

# 4. Verify the metric belongs to the question's domain
valid_mappings = df_mapping.select(
    "domain_id",
    F.col("metric_id").alias("primary_metric_id")
)

invalid_mappings = (
    df_questions
    .join(
        valid_mappings,
        on=["domain_id", "primary_metric_id"],
        how="left_anti"
    )
    .count()
)

# 5. Validate supported question intents
approved_intents = [
    "AGGREGATION",
    "TREND",
    "RANKING",
    "COMPARISON"
]

invalid_intents = (
    df_questions
    .filter(
        F.col("intent_type").isNull() |
        ~F.col("intent_type").isin(approved_intents)
    )
    .count()
)

# 6. Validate required question fields
invalid_required_fields = df_questions.filter(
    F.col("question_id").isNull() |
    F.col("domain_id").isNull() |
    F.col("primary_metric_id").isNull() |
    F.col("example_question").isNull() |
    (F.trim(F.col("example_question")) == "")
).count()

# 7. Verify grouping dimensions against Gold tables
metric_sources = {
    row["metric_id"]: row["gold_table"]
    for row in df_metrics.select(
        "metric_id", "gold_table"
    ).collect()
}

invalid_dimensions = []

for question in df_questions.collect():

    dimension = question["group_by_dimension"]

    if dimension is None:
        continue

    metric_id = question["primary_metric_id"]
    gold_table = metric_sources.get(metric_id)

    if gold_table is None:
        continue

    full_table = f"flightops_intelligence.gold.{gold_table}"

    if not spark.catalog.tableExists(full_table):
        invalid_dimensions.append(
            f"{question['question_id']}: Missing table {full_table}"
        )
        continue

    available_columns = spark.table(full_table).columns

    if dimension not in available_columns:
        invalid_dimensions.append(
            f"{question['question_id']}: "
            f"'{dimension}' not found in {gold_table}"
        )

# Display results
print("Business questions checked:", df_questions.count())
print("Duplicate question IDs:", duplicate_questions)
print("Invalid domains:", invalid_domains)
print("Invalid metrics:", invalid_metrics)
print("Invalid domain-metric mappings:", invalid_mappings)
print("Invalid intents:", invalid_intents)
print("Missing required fields:", invalid_required_fields)
print("Invalid grouping dimensions:", len(invalid_dimensions))

for error in invalid_dimensions:
    print("ERROR:", error)

assert duplicate_questions == 0
assert invalid_domains == 0
assert invalid_metrics == 0
assert invalid_mappings == 0
assert invalid_intents == 0
assert invalid_required_fields == 0
assert len(invalid_dimensions) == 0

print("\nBusiness Question Catalog validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Persist Business Question Catalog
# MAGIC
# MAGIC

# COMMAND ----------

questions_target_table = (
    "flightops_intelligence.semantic.business_questions"
)

(
    df_questions.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(questions_target_table)
)

print("Business Question Catalog created:", questions_target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Business Question Catalog

# COMMAND ----------

df_questions_saved = spark.table(
    "flightops_intelligence.semantic.business_questions"
)

print("Business questions:", df_questions_saved.count())
print("Columns:", len(df_questions_saved.columns))

display(
    df_questions_saved.orderBy("question_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Gold Data Products

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DateType, LongType
)

data_products = [
    (
        "DP001",
        "Daily Flight Performance",
        "daily_flight_performance",
        "Network-wide flight volumes, punctuality, "
        "cancellations and delay trends",
        "flight_date",
        "flight_date"
    ),
    (
        "DP002",
        "Airport Congestion Metrics",
        "airport_congestion_metrics",
        "Airport departure performance, taxi-out times, "
        "cancellations and delay indicators",
        "flight_date,origin_airport",
        "flight_date,origin_airport"
    ),
    (
        "DP003",
        "Carrier Performance",
        "carrier_performance",
        "Airline-level punctuality, cancellations "
        "and operational comparisons",
        "flight_date,airline",
        "flight_date,airline"
    )
]

catalog_records = []

for (
    product_id,
    product_name,
    gold_table,
    business_purpose,
    grain_columns,
    supported_dimensions
) in data_products:

    full_table = (
        f"flightops_intelligence.gold.{gold_table}"
    )

    df_gold = spark.table(full_table)

    stats = df_gold.agg(
        F.count("*").alias("record_count"),
        F.min("flight_date").alias("date_start"),
        F.max("flight_date").alias("date_end")
    ).first()

    catalog_records.append((
        product_id,
        product_name,
        gold_table,
        business_purpose,
        grain_columns,
        supported_dimensions,
        stats["date_start"],
        stats["date_end"],
        stats["record_count"]
    ))

product_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("product_name", StringType(), False),
    StructField("gold_table", StringType(), False),
    StructField("business_purpose", StringType(), False),
    StructField("grain_columns", StringType(), False),
    StructField("supported_dimensions", StringType(), False),
    StructField("date_start", DateType(), True),
    StructField("date_end", DateType(), True),
    StructField("record_count", LongType(), False)
])

df_data_products = spark.createDataFrame(
    catalog_records,
    schema=product_schema
)

print("Data products:", df_data_products.count())
print("Columns:", len(df_data_products.columns))

display(df_data_products)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Data Product Catalog

# COMMAND ----------

catalog_name = "flightops_intelligence"

df_metrics_saved = spark.table(
    f"{catalog_name}.semantic.metric_definitions"
)

product_rows = df_data_products.collect()
errors = []
seen_product_ids = set()
seen_gold_tables = set()

for product in product_rows:

    product_id = product["product_id"]
    gold_table = product["gold_table"]

    full_table = f"{catalog_name}.gold.{gold_table}"

    # 1. Validate unique product IDs and tables
    if product_id in seen_product_ids:
        errors.append(f"{product_id}: Duplicate product ID")

    if gold_table in seen_gold_tables:
        errors.append(f"{product_id}: Duplicate Gold table")

    seen_product_ids.add(product_id)
    seen_gold_tables.add(gold_table)

    # 2. Validate Gold table exists
    if not spark.catalog.tableExists(full_table):
        errors.append(f"{product_id}: Gold table missing")
        continue

    df_gold = spark.table(full_table)
    available_columns = set(df_gold.columns)

    # 3. Validate grain columns
    grain_columns = [
        col.strip()
        for col in product["grain_columns"].split(",")
    ]

    # 4. Validate supported dimensions
    dimensions = [
        col.strip()
        for col in product["supported_dimensions"].split(",")
    ]

    for col in set(grain_columns + dimensions):
        if not col or col not in available_columns:
            errors.append(
                f"{product_id}: Invalid column '{col}'"
            )

    # 5. Validate record count and date coverage
    stats = df_gold.agg(
        F.count("*").alias("record_count"),
        F.min("flight_date").alias("date_start"),
        F.max("flight_date").alias("date_end")
    ).first()

    if stats["record_count"] != product["record_count"]:
        errors.append(
            f"{product_id}: Record count mismatch"
        )

    if (
        stats["date_start"] != product["date_start"]
        or stats["date_end"] != product["date_end"]
    ):
        errors.append(
            f"{product_id}: Date coverage mismatch"
        )

    # 6. Validate unique grain
    if all(col in available_columns for col in grain_columns):

        duplicate_grains = (
            df_gold
            .groupBy(*grain_columns)
            .count()
            .filter(F.col("count") > 1)
            .limit(1)
            .count()
        )

        null_grains = df_gold.filter(
            " OR ".join(
                f"`{col}` IS NULL"
                for col in grain_columns
            )
        ).limit(1).count()

        if duplicate_grains > 0:
            errors.append(
                f"{product_id}: Duplicate grain records"
            )

        if null_grains > 0:
            errors.append(
                f"{product_id}: Null grain values"
            )

# 7. Validate Metric Definitions reference registered products
registered_tables = {
    product["gold_table"]
    for product in product_rows
}

metric_tables = {
    row["gold_table"]
    for row in df_metrics_saved.select("gold_table").distinct().collect()
}

for table in metric_tables - registered_tables:
    errors.append(
        f"Metric definitions reference unregistered table: {table}"
    )

# Final results
print("Data products checked:", len(product_rows))
print("Validation errors:", len(errors))

for error in errors:
    print("ERROR:", error)

assert len(errors) == 0, (
    "Data Product Catalog validation failed!"
)

print("Data Product Catalog validation passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Data Product Catalog
# MAGIC
# MAGIC

# COMMAND ----------

product_target_table = (
    "flightops_intelligence.semantic.data_product_catalog"
)

(
    df_data_products.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(product_target_table)
)

print("Data Product Catalog created:", product_target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Data Product Catalog

# COMMAND ----------

df_products_saved = spark.table(
    "flightops_intelligence.semantic.data_product_catalog"
)

print("Data products:", df_products_saved.count())
print("Columns:", len(df_products_saved.columns))

display(
    df_products_saved.orderBy("product_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### End-to-End Semantic Layer Integration

# COMMAND ----------

catalog = "flightops_intelligence"

domains = spark.table(
    f"{catalog}.semantic.business_domains"
)

mappings = spark.table(
    f"{catalog}.semantic.domain_metric_mapping"
)

metrics = spark.table(
    f"{catalog}.semantic.metric_definitions"
)

products = spark.table(
    f"{catalog}.semantic.data_product_catalog"
)

# Connect all four metadata tables
df_semantic_integrated = (
    mappings
    .join(domains, "domain_id", "inner")
    .join(metrics, "metric_id", "inner")
    .join(products, "gold_table", "inner")
    .select(
        "domain_id",
        "domain_name",
        "metric_id",
        "metric_name",
        "aggregation_rule",
        "numerator_column",
        "denominator_column",
        "product_id",
        "product_name",
        "gold_table",
        "display_priority"
    )
    .orderBy("domain_id", "display_priority")
)

print(
    "Integrated semantic mappings:",
    df_semantic_integrated.count()
)

print(
    "Business domains:",
    df_semantic_integrated
    .select("domain_id")
    .distinct()
    .count()
)

print(
    "Distinct metrics:",
    df_semantic_integrated
    .select("metric_id")
    .distinct()
    .count()
)

print(
    "Referenced data products:",
    df_semantic_integrated
    .select("product_id")
    .distinct()
    .count()
)

display(df_semantic_integrated)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Complete Semantic Layer Integration
# MAGIC

# COMMAND ----------

questions = spark.table(
    "flightops_intelligence.semantic.business_questions"
)

# Reuse df_semantic_integrated from Step 8.10.1
# Connect business questions to validated domain-metric mappings

df_complete_semantic = (
    questions.alias("q")
    .join(
        df_semantic_integrated.alias("s"),
        on=(
            (F.col("q.domain_id") == F.col("s.domain_id")) &
            (F.col("q.primary_metric_id") == F.col("s.metric_id"))
        ),
        how="left"
    )
    .select(
        F.col("q.question_id").alias("question_id"),
        F.col("q.example_question").alias("business_question"),
        F.col("q.domain_id").alias("domain_id"),
        F.col("s.domain_name").alias("domain_name"),
        F.col("s.metric_name").alias("metric_name"),
        F.col("q.intent_type").alias("intent_type"),
        F.col("q.group_by_dimension").alias("group_by_dimension"),
        F.col("s.aggregation_rule").alias("aggregation_rule"),
        F.col("s.gold_table").alias("gold_table"),
        F.col("s.product_id").alias("product_id")
    )
)

total_questions = questions.count()
resolved_questions = df_complete_semantic.filter(
    F.col("product_id").isNotNull()
).count()

unresolved_questions = df_complete_semantic.filter(
    F.col("product_id").isNull()
).count()

print("Business questions:", total_questions)
print("Resolved questions:", resolved_questions)
print("Unresolved questions:", unresolved_questions)

assert total_questions == 12
assert resolved_questions == total_questions
assert unresolved_questions == 0

print("\nComplete Semantic Layer integration passed!")

display(
    df_complete_semantic.orderBy("question_id")
)

# COMMAND ----------

