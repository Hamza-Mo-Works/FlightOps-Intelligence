# FlightOps Intelligence

FlightOps Intelligence is a data engineering and analytics project exploring operational insights from publicly available aviation datasets. The repository is organized around **Databricks, PySpark, SQL, Delta Lake, and a Bronze–Silver–Gold (medallion) architecture**, with room for analytical and AI applications as development continues.

## Project objective

Build a reusable data foundation for answering questions about US airline operations, including flight volume, punctuality, cancellations, airport activity, and airline performance. The project also explores how curated data can support natural-language analytics and predictive modelling.

The initial data source is the **US Bureau of Transportation Statistics (BTS) airline on-time performance dataset for 2025**, supplied as twelve monthly CSV files.

## First week: October 26–30, 2026

Initial source work and repository organization focused on:

- **Data exploration:** Inspecting monthly BTS source files, field definitions, coverage, date formats, and schema differences.
- **Bronze ingestion:** Establishing a raw Delta landing layer with ingestion metadata and traceability to the source files.
- **Silver transformations:** Standardizing columns and data types, parsing dates, validating operational flags, handling nulls, and checking duplicates.
- **Gold data modelling:** Defining daily flight, airport, and carrier performance aggregations for reporting and analysis.
- **Early reporting metrics:** Designing reusable flight counts, cancellation rates, on-time arrival rates, and taxi-out averages for the Gold reporting layer.

## November 2, 2026 — Bronze validation and data documentation

- Added a **read-only Bronze ingestion integration test** in `tests/test_bronze_ingestion.py` to check essential BTS source columns, ingestion metadata (`_ingestion_timestamp`, `_source_file`, `_source_system`), table non-emptiness and source attribution on a bounded sample.
- Added `docs/bts_data_dictionary.md` describing the raw BTS fields, intended Silver mappings, nullable outcomes and data-quality considerations.

The integration checks **require access to an active Databricks Spark session and the Bronze Delta table**; they have not been represented as passed without execution against that table. No model training or AI application release is claimed for this update.

This README records the repository's early development scope, not a feature-complete release statement. More modules and documentation will be introduced as the repository develops.

## Architecture overview

```text
BTS monthly CSV files
         |
         v
Bronze — source-fidelity Delta ingestion
         |
         v
Silver — standardized, validated flight records
         |
         +----------------------------+
         |                            |
         v                            v
Gold — operational aggregates     Flight-level data exploration
         |                            |
         v                            v
Semantic metric definitions       ML feature investigations
         |
         v
SQL-based reporting and analytics
```

## Main repository areas

| Area | Scope |
|---|---|
| `01_BTS_Flight_Data_Exploration.py` | Source inspection and initial dataset profiling |
| `02_BTS_Bronze_Ingestion.py` | Raw Delta ingestion |
| `03_BTS_Silver_Transformation.py` | Cleaning, standardization, and validation |
| `04_BTS_Gold_Analytics.py` | Operational aggregation datasets |
| `tests/test_bronze_ingestion.py` | Read-only integration checks for Bronze input fields and metadata |
| `docs/bts_data_dictionary.md` | BTS source-field reference and Silver mapping notes |
| `05_Business_Semantic_Layer.py` | Metric definitions and business-domain mappings (subsequent phase) |
| `13_Machine_Learning/` | Flight-delay ML development notebooks |
| `ML_V2/` | Weather data and further model experiments |
| `databricks_apps/` | Application source and deployment configuration |

Additional source modules will be introduced in subsequent phases. Notebook exports are Python source files intended for use in a Databricks workspace.

## Running the Bronze checks

From a Python environment with `pytest`, PySpark, an active Databricks/Databricks Connect Spark session, and catalog `SELECT` access:

```bash
python -m pytest -q tests/test_bronze_ingestion.py
```

The tests skip when no active Spark session is available. You can override the target with `FLIGHTOPS_BRONZE_TABLE` if necessary. The tests are read-only and do not ingest data or modify Delta tables.

## Technology

- **Data processing:** Databricks, Apache Spark / PySpark, SQL
- **Data storage:** Delta Lake and Unity Catalog
- **Data architecture:** Bronze, Silver, Gold
- **Exploration and modelling:** Python, MLflow, XGBoost
- **Application development:** Streamlit and Databricks Apps

## Data and configuration

The repository stores source code and documentation, **not** the multi-million-row BTS data, METAR archives, local credentials, or model binary artifacts. Source locations, Unity Catalog permissions, SQL Warehouse access, and model resources must be configured for the target Databricks environment.

## Data source

US Bureau of Transportation Statistics (BTS), Airline On-Time Performance data. Historical airport weather (METAR) is also used in related experimental work.

> This is an engineering and research project. Experimental model outputs are not validated operational flight forecasts.
