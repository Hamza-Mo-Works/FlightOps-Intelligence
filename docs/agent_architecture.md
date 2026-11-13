# Analytical agent flow

The project contains notebook-based AI context/runtime components and a direct
SQL app backend. The app path calls the existing SQL Warehouse and AI Gateway;
it does not need to rerun the transformation notebooks for each user question.

1. Collect the question and selected domain.
2. Supply allowed metric/context information to the model.
3. Receive a structured analytical tool request.
4. Validate intent, metrics, dates, entities, grain and ranking parameters in Python.
5. Build SQL from approved Gold tables and semantic expressions.
6. Execute with bound entity/date parameters.
7. Preserve complete SQL rows for the UI and supply a compact summary for narration.

Ranking direction and explicitly requested result counts are resolved from the
user's wording in Python. The model's suggested arguments do not override clear
highest/lowest wording. Ratio-based rankings require at least 100 observations.

Tests call the real validation, ranking and presentation function bodies with
controlled SQL/metadata adapters. They do not assess the quality of a live model's
intent selection or prose. Check live questions against their returned SQL rows.
