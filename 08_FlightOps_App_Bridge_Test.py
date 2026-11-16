# Databricks notebook source
# MAGIC %md
# MAGIC # FlightOps Intelligence — App-to-Spark Bridge Test
# MAGIC
# MAGIC Step 11.6.2A: Tests Lakeflow Job execution, input parameters, Unity Catalog access and JSON output. This is **not** the AI agent yet.

# COMMAND ----------

# Job parameters. Define the same keys in the Lakeflow Job settings.
dbutils.widgets.text("question", "How many flights were scheduled in July 2025?")
dbutils.widgets.dropdown("domain", "OPS", ["OPS", "AIRPORT", "CARRIER"])

question = dbutils.widgets.get("question").strip()
domain = dbutils.widgets.get("domain")

if not question or len(question) > 2000:
    raise ValueError("Question must have 1-2000 characters")

TABLES = {
    "OPS": "flightops_intelligence.gold.daily_flight_performance",
    "AIRPORT": "flightops_intelligence.gold.airport_congestion_metrics",
    "CARRIER": "flightops_intelligence.gold.carrier_performance",
}

import json
result = {
    "status": "ok",
    "phase": "11.6.2A",
    "domain": domain,
    "question": question,
    "source_table": TABLES[domain],
    "gold_rows": spark.table(TABLES[domain]).count(),
}
print(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))