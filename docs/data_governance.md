# Data ownership and catalog responsibilities

`flightops_intelligence` separates Bronze source records, Silver flight/weather
data, Gold aggregates, semantic definitions and ML resources into schema roles.

Bronze preserves received source fields and ingestion attribution. Silver owns
typing, validity, deduplication and flight-level operational flags. Gold owns
stable analytical grains and additive KPI components. Semantic metadata owns
metric identifiers, definitions, allowed aggregations and domain mappings.
ML resources own feature versions, training/evaluation artifacts and releases.

Assign actual table ownership and permissions through Unity Catalog. The code's
domain names express analytical scope, not authenticated business roles.
Separate the rights needed for notebook writes from those needed for app reads.

Keep source month/file attribution and ingestion timestamps. Track notebook
versions and training/model identifiers with released outputs. The Git history
alone does not prove data lineage, access enforcement or model reproducibility.
