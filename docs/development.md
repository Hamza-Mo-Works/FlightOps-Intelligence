# Running the supplied tests

Extract the package into the main project root. Existing notebooks and application
modules are used in place. The package adds tests/docs/configuration; it does not
replace implementation notebooks or the existing Bronze test.

## Windows: fast source tests

```cmd
python -m pip install -r requirements-test.txt
python -m pytest -q -m "not spark and not databricks"
```

## Windows: include bounded Spark expression tests

Use Python 3.10–3.12 and Java 17 with JAVA_HOME configured.

```cmd
python -m pip install -r requirements-test-spark.txt
set "FLIGHTOPS_TEST_ALLOW_LOCAL_SPARK=1"
python -m pytest -q -m "not databricks"
set "FLIGHTOPS_TEST_ALLOW_LOCAL_SPARK="
```

The Spark fixture uses an isolated session with UTC, corrected date parsing and
ANSI mode disabled for the bounded transformation checks. It does not write Delta
tables. Function/constant loading omits notebook setup, table writes and cache
decorators; tests call the real selected source bodies with controlled I/O.

## Databricks: table reconciliation

Install pytest in the notebook environment if needed. Set the working directory
to the project root, then execute in the Python kernel:

```python
import os
import pytest
os.environ["FLIGHTOPS_TEST_LIVE_TABLES"] = "1"
pytest.main(["-q", "tests/test_bronze_ingestion.py", "tests/test_data_reconciliation.py"])
```

Use the notebook kernel, not `%sh python`, when relying on an active Spark session.
The live checks read existing tables. A skipped live check means it has not been
executed. The conftest marks the existing Bronze tests as Databricks checks without
editing that committed file.
