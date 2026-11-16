# Databricks notebook source
import csv
import re
from io import StringIO

landing_path = "/Volumes/flightops_intelligence/landing/raw_files/bts/"

files = sorted(
    [f for f in dbutils.fs.ls(landing_path) if f.name.lower().endswith(".csv")]
)

print("Total CSV files:", len(files))

schemas = {}
months = []

for file in files:
    match = re.search(r"_2025_(\d{1,2})\.csv$", file.name)

    if not match:
        print("Unexpected filename:", file.name)
        continue

    month = int(match.group(1))
    months.append(month)

    header = next(csv.reader(
        StringIO(dbutils.fs.head(file.path, 10000))
    ))

    schemas[month] = header

    print(
        f"Month {month:02d}: "
        f"{len(header)} columns | "
        f"{file.size / (1024**2):.1f} MB"
    )

print("\nAll months present:", sorted(months) == list(range(1, 13)))

july_schema = schemas.get(7)

if july_schema:
    for month in sorted(schemas):
        status = "MATCH" if schemas[month] == july_schema else "DIFFERENT"
        print(f"Month {month:02d} schema: {status}")

    print("\nJuly source columns:", len(july_schema))

# COMMAND ----------

import csv
import os

base_path = "/Volumes/flightops_intelligence/landing/raw_files/bts"

def read_csv_header(month):
    filename = (
        "On_Time_Reporting_Carrier_On_Time_Performance_"
        f"(1987_present)_2025_{month}.csv"
    )

    filepath = os.path.join(base_path, filename)

    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader, [])

    return header, len(first_row)


# Read complete headers (no 10,000-byte limit)
july_cols, july_row_len = read_csv_header(7)
jan_cols, jan_row_len = read_csv_header(1)

print("JULY COLUMN COUNT:", len(july_cols))
print("JANUARY COLUMN COUNT:", len(jan_cols))

print("\nColumns in January but NOT July:")
print([col for col in jan_cols if col not in july_cols])

print("\nColumns in July but NOT January:")
print([col for col in july_cols if col not in jan_cols])

print("\nFirst data row field counts:")
print("January:", jan_row_len)
print("July:", july_row_len)

print("\nLast 10 columns — January:")
print(jan_cols[-10:])

print("\nLast 10 columns — July:")
print(july_cols[-10:])

print("\nColumn order identical:", jan_cols == july_cols)

print("\nSCHEMA CHECK ACROSS ALL MONTHS")

for month in range(1, 13):
    cols, row_len = read_csv_header(month)

    missing = [c for c in july_cols if c not in cols]
    extra = [c for c in cols if c not in july_cols]

    print(
        f"Month {month:02d}: "
        f"{len(cols)} columns | "
        f"Missing: {missing} | "
        f"Extra: {extra} | "
        f"First row fields: {row_len}"
    )

# COMMAND ----------

# Check that every month's named columns match July
for month in range(1, 13):
    cols, _ = read_csv_header(month)

    named_cols = [c for c in cols if c.strip()]

    print(
        f"Month {month:02d}: "
        f"{'MATCH' if named_cols == july_cols else 'DIFFERENT'}"
    )

# COMMAND ----------

BRONZE_TABLE = "flightops_intelligence.bronze.bts_flights_raw"

bronze_df = spark.table(BRONZE_TABLE)

print("=== CURRENT BRONZE TABLE ===")
print("Total rows:", bronze_df.count())
print("Total columns:", len(bronze_df.columns))

print("\n=== COLUMN NAMES AND TYPES ===")
bronze_df.printSchema()

print("\n=== BRONZE COLUMNS NOT FOUND IN BTS CSV ===")

csv_columns = set(july_cols)

for field in bronze_df.schema.fields:
    if field.name not in csv_columns:
        print(f"{field.name}: {field.dataType.simpleString()}")

print("\n=== DATE-RELATED COLUMNS ===")
print([
    col for col in bronze_df.columns
    if "date" in col.lower()
    or "month" in col.lower()
    or "source" in col.lower()
    or "ingest" in col.lower()
])

# COMMAND ----------

import re
from pyspark.sql import functions as F

BRONZE = "flightops_intelligence.bronze.bts_flights_raw"
LANDING = "/Volumes/flightops_intelligence/landing/raw_files/bts/"

bronze = spark.table(BRONZE)
source_columns = bronze.columns[:109]

# Locate the January CSV
january_files = [
    f for f in dbutils.fs.ls(LANDING)
    if re.search(r"_2025_1\.csv$", f.name)
]

assert len(january_files) == 1, "Expected exactly one January file"

january_file = january_files[0]

# Read the source without inferring data types
raw_january = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("mode", "FAILFAST")
    .csv(january_file.path)
)

print("Source file:", january_file.name)
print("CSV column count:", len(raw_january.columns))
print("Bronze source columns:", len(source_columns))

# Confirm exact named-column alignment
assert raw_january.columns[:109] == source_columns, (
    "Source column names/order differ from Bronze"
)

assert len(raw_january.columns) == 110, (
    "Expected 109 BTS columns plus one trailing unnamed field"
)

# Verify the extra field contains no actual information
extra_column = raw_january.columns[-1]

extra_nonempty = raw_january.filter(
    F.col(extra_column).isNotNull()
    & (F.trim(F.col(extra_column)) != "")
).count()

print("Nonempty values in extra column:", extra_nonempty)

assert extra_nonempty == 0, (
    "The extra CSV column contains data. Stop ingestion."
)

# Retain only the 109 validated BTS fields
january = raw_january.select(*source_columns)

# Validate reporting year, month, and flight date
invalid_rows = january.filter(
    F.col("Year").cast("int") != 2025
).unionByName(
    january.filter(
        F.col("Month").cast("int") != 1
    )
).unionByName(
    january.filter(
        F.to_date("FlightDate").isNull()
        | (F.year(F.to_date("FlightDate")) != 2025)
        | (F.month(F.to_date("FlightDate")) != 1)
    )
).count()

assert invalid_rows == 0, (
    f"January has {invalid_rows} invalid reporting fields"
)

january_count = january.count()

existing_january = bronze.filter(
    (F.col("Year").cast("int") == 2025)
    & (F.col("Month").cast("int") == 1)
).count()

july_count_before = bronze.filter(
    (F.col("Year").cast("int") == 2025)
    & (F.col("Month").cast("int") == 7)
).count()

print("\n=== JANUARY VALIDATION ===")
print("January CSV records:", january_count)
print("Existing January Bronze records:", existing_january)
print("Existing July Bronze records:", july_count_before)
print("Valid source fields:", len(january.columns))
print("Invalid reporting fields:", invalid_rows)

assert existing_january == 0, "January already exists in Bronze"
assert july_count_before == 631428, "Existing July count differs"

print("\nJANUARY VALIDATION PASSED")

# COMMAND ----------

# Reuse July's existing source-system metadata convention
july_sources = [
    row["_source_system"]
    for row in (
        bronze
        .filter(
            (F.col("Year").cast("int") == 2025)
            & (F.col("Month").cast("int") == 7)
        )
        .select("_source_system")
        .distinct()
        .collect()
    )
]

assert len(july_sources) == 1, (
    "Expected one existing source-system value"
)

source_system = july_sources[0]

# Match July's source-file naming convention
july_file_example = (
    bronze
    .filter(
        (F.col("Year").cast("int") == 2025)
        & (F.col("Month").cast("int") == 7)
    )
    .select("_source_file")
    .first()[0]
)

source_file_value = (
    january_file.path
    if july_file_example and "/" in july_file_example
    else january_file.name
)

# Prevent accidental repeated January ingestion
assert spark.table(BRONZE).filter(
    (F.col("Year").cast("int") == 2025)
    & (F.col("Month").cast("int") == 1)
).count() == 0, "January is already loaded"

# Add the same five technical fields used in July
january_bronze = (
    january
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file", F.lit(source_file_value))
    .withColumn("_source_system", F.lit(source_system))
    .withColumn("_ingestion_year", F.lit(2025).cast("int"))
    .withColumn("_ingestion_month", F.lit(1).cast("int"))
    .select(*bronze.columns)
)

assert len(january_bronze.columns) == 114

# Append, never overwrite
(
    january_bronze.write
    .format("delta")
    .mode("append")
    .saveAsTable(BRONZE)
)

print("January ingestion completed")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     CAST(Year AS INT) AS reporting_year,
# MAGIC     CAST(Month AS INT) AS reporting_month,
# MAGIC     COUNT(*) AS total_flights,
# MAGIC     COUNT(DISTINCT FlightDate) AS reporting_days,
# MAGIC     MIN(FlightDate) AS first_flight_date,
# MAGIC     MAX(FlightDate) AS last_flight_date
# MAGIC FROM flightops_intelligence.bronze.bts_flights_raw
# MAGIC WHERE CAST(Year AS INT) = 2025
# MAGIC GROUP BY CAST(Year AS INT), CAST(Month AS INT)
# MAGIC ORDER BY reporting_month;

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH parsed_flights AS (
# MAGIC     SELECT
# MAGIC         CAST(Year AS INT) AS reporting_year,
# MAGIC         CAST(Month AS INT) AS reporting_month,
# MAGIC         COALESCE(
# MAGIC             CAST(TRY_TO_TIMESTAMP(FlightDate, 'yyyy-MM-dd') AS DATE),
# MAGIC             CAST(TRY_TO_TIMESTAMP(FlightDate, 'M/d/yyyy') AS DATE)
# MAGIC         ) AS flight_date
# MAGIC     FROM flightops_intelligence.bronze.bts_flights_raw
# MAGIC )
# MAGIC SELECT
# MAGIC     reporting_year,
# MAGIC     reporting_month,
# MAGIC     COUNT(*) AS total_flights,
# MAGIC     COUNT(DISTINCT flight_date) AS reporting_days,
# MAGIC     MIN(flight_date) AS first_flight_date,
# MAGIC     MAX(flight_date) AS last_flight_date,
# MAGIC     SUM(CASE WHEN flight_date IS NULL THEN 1 ELSE 0 END)
# MAGIC         AS invalid_dates,
# MAGIC     SUM(
# MAGIC         CASE
# MAGIC             WHEN flight_date IS NOT NULL
# MAGIC               AND (
# MAGIC                 YEAR(flight_date) <> reporting_year
# MAGIC                 OR MONTH(flight_date) <> reporting_month
# MAGIC               )
# MAGIC             THEN 1 ELSE 0
# MAGIC         END
# MAGIC     ) AS mismatched_dates
# MAGIC FROM parsed_flights
# MAGIC GROUP BY reporting_year, reporting_month
# MAGIC ORDER BY reporting_year, reporting_month;

# COMMAND ----------

import re
from pyspark.sql import functions as F

BRONZE = "flightops_intelligence.bronze.bts_flights_raw"
LANDING = "/Volumes/flightops_intelligence/landing/raw_files/bts/"

bronze = spark.table(BRONZE)
source_columns = bronze.columns[:109]
bronze_columns = bronze.columns

# Discover all 2025 monthly source files
monthly_files = {}

for file in dbutils.fs.ls(LANDING):
    match = re.search(r"_2025_(\d{1,2})\.csv$", file.name)

    if match:
        month = int(match.group(1))
        assert month not in monthly_files, f"Duplicate source for month {month}"
        monthly_files[month] = file

assert set(monthly_files) == set(range(1, 13)), (
    "Expected exactly 12 monthly source files"
)

# Find months already present in Bronze
def loaded_months():
    rows = (
        spark.table(BRONZE)
        .filter(F.col("Year").cast("int") == 2025)
        .groupBy(F.col("Month").cast("int").alias("month"))
        .count()
        .collect()
    )
    return {r["month"]: r["count"] for r in rows}

existing = loaded_months()

assert existing.get(1) == 539747, "January count differs"
assert existing.get(7) == 631428, "July count differs"

# Reuse existing ingestion metadata convention
source_systems = [
    r[0] for r in bronze
    .filter(F.col("Month").cast("int") == 1)
    .select("_source_system")
    .distinct()
    .collect()
]

assert len(source_systems) == 1
source_system = source_systems[0]

source_file_example = (
    bronze
    .filter(F.col("Month").cast("int") == 1)
    .select("_source_file")
    .first()[0]
)

loaded_counts = {}

for month in range(1, 13):

    if month in existing:
        print(f"Month {month:02d}: SKIPPED (already loaded)")
        continue

    file = monthly_files[month]

    raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .option("mode", "FAILFAST")
        .csv(file.path)
    )

    # Validate source schema
    assert raw.columns[:109] == source_columns, (
        f"Month {month:02d}: Source schema mismatch"
    )

    assert len(raw.columns) in (109, 110), (
        f"Month {month:02d}: Unexpected column count"
    )

    # Check extra trailing column and reporting dates
    extra_invalid = F.lit(False)

    if len(raw.columns) == 110:
        extra = raw.columns[-1]
        extra_invalid = (
            F.col(f"`{extra}`").isNotNull()
            & (F.trim(F.col(f"`{extra}`")) != "")
        )

    parsed_date = F.coalesce(
        F.expr("CAST(TRY_TO_TIMESTAMP(FlightDate, 'yyyy-MM-dd') AS DATE)"),
        F.expr("CAST(TRY_TO_TIMESTAMP(FlightDate, 'M/d/yyyy') AS DATE)")
    )

    checked = raw.withColumn("_parsed_date_check", parsed_date)

    invalid_date = (
        F.col("_parsed_date_check").isNull()
        | (F.year("_parsed_date_check") != 2025)
        | (F.month("_parsed_date_check") != month)
        | (F.col("Year").cast("int").isNull())
        | (F.col("Year").cast("int") != 2025)
        | (F.col("Month").cast("int").isNull())
        | (F.col("Month").cast("int") != month)
    )

    stats = checked.agg(
        F.count("*").alias("rows"),
        F.sum(F.when(extra_invalid, 1).otherwise(0)).alias("extra_errors"),
        F.sum(F.when(invalid_date, 1).otherwise(0)).alias("date_errors")
    ).first()

    assert stats["rows"] > 0
    assert stats["extra_errors"] == 0, (
        f"Month {month:02d}: Extra column contains data"
    )
    assert stats["date_errors"] == 0, (
        f"Month {month:02d}: Invalid reporting dates"
    )

    # Retain the 109 original source fields
    df = raw.select(*source_columns)

    file_value = (
        file.path
        if source_file_example and "/" in source_file_example
        else file.name
    )

    bronze_month = (
        df
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_source_file", F.lit(file_value))
        .withColumn("_source_system", F.lit(source_system))
        .withColumn("_ingestion_year", F.lit(2025).cast("int"))
        .withColumn("_ingestion_month", F.lit(month).cast("int"))
        .select(*bronze_columns)
    )

    # Final rerun guard
    already_loaded = (
        spark.table(BRONZE)
        .filter(
            (F.col("Year").cast("int") == 2025)
            & (F.col("Month").cast("int") == month)
        )
        .limit(1)
        .count()
    )

    assert already_loaded == 0, (
        f"Month {month:02d} was loaded during this run"
    )

    # Append validated records
    (
        bronze_month.write
        .format("delta")
        .mode("append")
        .saveAsTable(BRONZE)
    )

    # Verify the write completed with the correct row count
    actual_count = (
        spark.table(BRONZE)
        .filter(
            (F.col("Year").cast("int") == 2025)
            & (F.col("Month").cast("int") == month)
        )
        .count()
    )

    assert actual_count == stats["rows"], (
        f"Month {month:02d}: Row-count reconciliation failed"
    )

    loaded_counts[month] = actual_count

    print(
        f"Month {month:02d}: LOADED "
        f"{actual_count:,} records"
    )

print("\nBRONZE INGESTION RUN COMPLETED")
print("Newly loaded months:", len(loaded_counts))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     CAST(Month AS INT) AS month,
# MAGIC     COUNT(*) AS total_flights,
# MAGIC     COUNT(DISTINCT FlightDate) AS reporting_days
# MAGIC FROM flightops_intelligence.bronze.bts_flights_raw
# MAGIC WHERE CAST(Year AS INT) = 2025
# MAGIC GROUP BY CAST(Month AS INT)
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH parsed AS (
# MAGIC     SELECT
# MAGIC         CAST(Year AS INT) AS yr,
# MAGIC         CAST(Month AS INT) AS mo,
# MAGIC         FlightDate,
# MAGIC         COALESCE(
# MAGIC             CAST(TRY_TO_TIMESTAMP(FlightDate, 'yyyy-MM-dd') AS DATE),
# MAGIC             CAST(TRY_TO_TIMESTAMP(FlightDate, 'M/d/yyyy') AS DATE)
# MAGIC         ) AS parsed_date
# MAGIC     FROM flightops_intelligence.bronze.bts_flights_raw
# MAGIC     WHERE CAST(Year AS INT) = 2025
# MAGIC ),
# MAGIC monthly AS (
# MAGIC     SELECT
# MAGIC         yr,
# MAGIC         mo,
# MAGIC         COUNT(*) AS total_flights,
# MAGIC         COUNT(DISTINCT parsed_date) AS reporting_days,
# MAGIC         MIN(parsed_date) AS start_date,
# MAGIC         MAX(parsed_date) AS end_date,
# MAGIC         SUM(CASE WHEN parsed_date IS NULL THEN 1 ELSE 0 END)
# MAGIC             AS invalid_dates,
# MAGIC         SUM(CASE
# MAGIC             WHEN parsed_date IS NOT NULL
# MAGIC              AND (YEAR(parsed_date) <> yr OR MONTH(parsed_date) <> mo)
# MAGIC             THEN 1 ELSE 0 END) AS mismatched_dates,
# MAGIC         SUM(CASE WHEN FlightDate LIKE '%/%' THEN 1 ELSE 0 END)
# MAGIC             AS slash_format_rows,
# MAGIC         SUM(CASE WHEN FlightDate LIKE '%-%' THEN 1 ELSE 0 END)
# MAGIC             AS iso_format_rows
# MAGIC     FROM parsed
# MAGIC     GROUP BY yr, mo
# MAGIC )
# MAGIC SELECT
# MAGIC     mo AS month,
# MAGIC     total_flights,
# MAGIC     reporting_days,
# MAGIC     start_date,
# MAGIC     end_date,
# MAGIC     invalid_dates,
# MAGIC     mismatched_dates,
# MAGIC     slash_format_rows,
# MAGIC     iso_format_rows,
# MAGIC     CASE
# MAGIC         WHEN start_date = MAKE_DATE(yr, mo, 1)
# MAGIC          AND end_date = LAST_DAY(MAKE_DATE(yr, mo, 1))
# MAGIC          AND reporting_days = DAY(LAST_DAY(MAKE_DATE(yr, mo, 1)))
# MAGIC          AND invalid_dates = 0
# MAGIC          AND mismatched_dates = 0
# MAGIC         THEN 'PASS'
# MAGIC         ELSE 'CHECK'
# MAGIC     END AS validation_status
# MAGIC FROM monthly
# MAGIC ORDER BY month;

# COMMAND ----------

