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
- **Governed metrics:** Establishing reusable business measures such as flight counts, cancellation rates, on-time arrival rates, and taxi-out averages.
- **Application and ML exploration:** Organizing the supporting source notebooks and prototyping approaches to conversational queries and flight-delay risk modelling.

This README is an **initial development overview**, not a feature-complete release statement. Implementation notes, tests, and additional technical documentation will be expanded as source modules are organized in the repository.

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
| `05_Business_Semantic_Layer.py` | Metric definitions and business-domain mappings |
| `13_Machine_Learning/` | Flight-delay ML development notebooks |
| `ML_V2/` | Weather data and further model experiments |
| `databricks_apps/` | Application source and deployment configuration |

Source files may be added incrementally. Notebook exports are Python source files intended for use in a Databricks workspace.

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
