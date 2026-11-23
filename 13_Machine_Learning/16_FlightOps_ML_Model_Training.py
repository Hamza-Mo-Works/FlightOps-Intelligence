# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 13.5 — ML Model Training
# MAGIC
# MAGIC Train Logistic Regression and XGBoost for flight arrival-delay prediction.
# MAGIC
# MAGIC - Training: Jan–Aug 2025
# MAGIC - Validation: Sep–Oct 2025
# MAGIC - Test: Nov–Dec 2025 (reserved)
# MAGIC - Track model performance using MLflow.

# COMMAND ----------

from pyspark.sql import functions as F

SOURCE = "flightops_intelligence.ml.flight_delay_features_v1_history"

df = spark.table(SOURCE)

CATEGORICAL = [
    "airline",
    "origin_airport",
    "destination_airport"
]

NUMERICAL = [
    "month",
    "day_of_week",
    "is_weekend",
    "departure_hour",
    "departure_minute",
    "arrival_hour",
    "arrival_minute",
    "distance_miles"
]

FEATURES = CATEGORICAL + NUMERICAL

train_full = df.filter(
    F.col("flight_date") < "2025-09-01"
)

validation = df.filter(
    (F.col("flight_date") >= "2025-09-01")
    & (F.col("flight_date") < "2025-11-01")
)

# Reproducible training sample
train = train_full.sample(
    withReplacement=False,
    fraction=0.15,
    seed=42
)

print("Selected features:", FEATURES)
print("Training sample:", train.count())
print("Validation records:", validation.count())
print("Test data remains untouched.")

# COMMAND ----------

import mlflow

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

indexed = [c + "_idx" for c in CATEGORICAL]
encoded = [c + "_vec" for c in CATEGORICAL]

indexer = StringIndexer(
    inputCols=CATEGORICAL,
    outputCols=indexed,
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=indexed,
    outputCols=encoded,
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=encoded + NUMERICAL,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=False
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="delay_label",
    maxIter=30,
    regParam=0.01,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=[
    indexer,
    encoder,
    assembler,
    scaler,
    lr
])

with mlflow.start_run(run_name="FlightOps_Logistic_Regression_V1") as run:

    mlflow.log_params({
        "model": "LogisticRegression",
        "train_period": "Jan-Aug 2025",
        "validation_period": "Sep-Oct 2025",
        "training_fraction": 0.15,
        "seed": 42,
        "feature_count": len(FEATURES),
        "max_iter": 30,
        "reg_param": 0.01
    })

    model = pipeline.fit(train)

    predictions = model.transform(validation)

    roc_auc = BinaryClassificationEvaluator(
        labelCol="delay_label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)

    pr_auc = BinaryClassificationEvaluator(
        labelCol="delay_label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    ).evaluate(predictions)

    results = predictions.agg(
        F.sum(F.when(
            (F.col("prediction") == 1) &
            (F.col("delay_label") == 1), 1
        ).otherwise(0)).alias("tp"),

        F.sum(F.when(
            (F.col("prediction") == 1) &
            (F.col("delay_label") == 0), 1
        ).otherwise(0)).alias("fp"),

        F.sum(F.when(
            (F.col("prediction") == 0) &
            (F.col("delay_label") == 1), 1
        ).otherwise(0)).alias("fn")
    ).first()

    tp, fp, fn = results["tp"], results["fp"], results["fn"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    mlflow.log_metrics({
        "validation_roc_auc": roc_auc,
        "validation_pr_auc": pr_auc,
        "validation_precision": precision,
        "validation_recall": recall,
        "validation_f1": f1
    })

    print("=== LOGISTIC REGRESSION BASELINE ===")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("MLflow Run ID:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13.5.2 — XGBoost Baseline
# MAGIC
# MAGIC Train XGBoost using the same 11 predictors as Logistic Regression.
# MAGIC
# MAGIC Evaluate ROC-AUC, PR-AUC, precision, recall and F1 on September–October 2025. Keep the November–December test set untouched.

# COMMAND ----------

# MAGIC %pip install xgboost==3.0.5

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import xgboost as xgb
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

print("XGBoost version:", xgb.__version__)

# Reuse the exact 11 predictor names from our LR baseline
CATEGORICAL = [
    "airline",
    "origin_airport",
    "destination_airport"
]

NUMERICAL = [
    "month",
    "day_of_week",
    "is_weekend",
    "departure_hour",
    "departure_minute",
    "arrival_hour",
    "arrival_minute",
    "distance_miles"
]

FEATURES = CATEGORICAL + NUMERICAL
SOURCE = "flightops_intelligence.ml.flight_delay_features_v1_history"

df = spark.table(SOURCE)

# Bounded initial samples for Free Edition
train_spark = (
    df.filter("flight_date < '2025-09-01'")
    .select(*(FEATURES + ["delay_label"]))
    .sample(False, 0.04, seed=42)
)

val_spark = (
    df.filter(
        "flight_date >= '2025-09-01' "
        "AND flight_date < '2025-11-01'"
    )
    .select(*(FEATURES + ["delay_label"]))
    .sample(False, 0.10, seed=42)
)

train_pd = train_spark.toPandas()
val_pd = val_spark.toPandas()

print("XGBoost training rows:", len(train_pd))
print("XGBoost validation rows:", len(val_pd))

X_train = train_pd[FEATURES]
y_train = train_pd["delay_label"].astype(int)

X_val = val_pd[FEATURES]
y_val = val_pd["delay_label"].astype(int)

print("Training delay rate:", round(y_train.mean() * 100, 2), "%")
print("Validation delay rate:", round(y_val.mean() * 100, 2), "%")

# COMMAND ----------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            CATEGORICAL
        ),
        ("numerical", "passthrough", NUMERICAL)
    ]
)

xgb_model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    n_jobs=4
)

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

with mlflow.start_run(run_name="FlightOps_XGBoost_V1") as run:

    xgb_pipeline.fit(X_train, y_train)

    probabilities = xgb_pipeline.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "validation_roc_auc": roc_auc_score(
            y_val, probabilities
        ),
        "validation_pr_auc": average_precision_score(
            y_val, probabilities
        ),
        "validation_precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "validation_recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "validation_f1": f1_score(
            y_val, predictions, zero_division=0
        )
    }

    mlflow.log_params({
        "model": "XGBoost",
        "version": "V1",
        "train_fraction": 0.04,
        "validation_fraction": 0.10,
        "features": len(FEATURES),
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42
    })

    mlflow.log_metrics(metrics)

    print("=== XGBOOST V1 ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("MLflow Run ID:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13.5.3 — XGBoost V1: Main Training
# MAGIC
# MAGIC Train with all 18 candidate features, evaluate on the complete Sep–Oct validation period, and select a classification threshold. The Nov–Dec test set remains untouched.

# COMMAND ----------

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow

from pyspark.sql import functions as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)

SOURCE = "flightops_intelligence.ml.flight_delay_features_v1_history"

CATEGORICAL = [
    "airline",
    "origin_airport",
    "destination_airport",
    "route"
]

NUMERICAL = [
    "month", "day_of_week", "is_weekend",
    "departure_hour", "departure_minute",
    "arrival_hour", "arrival_minute",
    "distance_miles",
    "airline_prior_flights", "airline_prior_delay_rate",
    "airport_prior_flights", "airport_prior_delay_rate",
    "route_prior_flights", "route_prior_delay_rate"
]

FEATURES = CATEGORICAL + NUMERICAL

assert len(FEATURES) == 18

source = spark.table(SOURCE)

# Larger training sample; increase only after checking memory
TRAIN_FRACTION = 0.25

train_spark = (
    source
    .filter(F.col("flight_date") < "2025-09-01")
    .select(*(FEATURES + ["delay_label"]))
    .sample(False, TRAIN_FRACTION, seed=42)
)

# Entire validation period, not a sample
val_spark = (
    source
    .filter(
        (F.col("flight_date") >= "2025-09-01") &
        (F.col("flight_date") < "2025-11-01")
    )
    .select(*(FEATURES + ["delay_label"]))
)

train_pd = train_spark.toPandas()
val_pd = val_spark.toPandas()

# Native XGBoost categorical handling
for col in CATEGORICAL:
    train_pd[col] = train_pd[col].astype("category")
    val_pd[col] = pd.Categorical(
        val_pd[col],
        categories=train_pd[col].cat.categories
    )

# Reduce numerical memory usage
for col in NUMERICAL:
    train_pd[col] = pd.to_numeric(
        train_pd[col], errors="coerce"
    ).astype("float32")

    val_pd[col] = pd.to_numeric(
        val_pd[col], errors="coerce"
    ).astype("float32")

X_train = train_pd[FEATURES]
y_train = train_pd["delay_label"].astype("int32")

X_val = val_pd[FEATURES]
y_val = val_pd["delay_label"].astype("int32")

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Predictor columns:", len(FEATURES))
print("Validation delay rate:", round(y_val.mean() * 100, 2), "%")

# COMMAND ----------

model = xgb.XGBClassifier(
    n_estimators=350,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=8,
    max_cat_threshold=64,
    random_state=42,
    n_jobs=4
)

with mlflow.start_run(
    run_name="FlightOps_XGBoost_V1_Main"
) as run:

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)

    precision_curve, recall_curve, thresholds = (
        precision_recall_curve(y_val, proba)
    )

    f1_curve = (
        2 * precision_curve[:-1] * recall_curve[:-1]
        / np.maximum(
            precision_curve[:-1] + recall_curve[:-1],
            1e-12
        )
    )

    best_idx = int(np.argmax(f1_curve))
    best_threshold = float(thresholds[best_idx])

    print("=== XGBOOST MAIN TRAINING ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    print("\n=== THRESHOLD COMPARISON ===")

    for threshold in [0.20, 0.30, 0.40, 0.50, best_threshold]:

        predictions = (proba >= threshold).astype(int)

        print(
            f"Threshold={threshold:.3f} | "
            f"Precision={precision_score(y_val, predictions, zero_division=0):.4f} | "
            f"Recall={recall_score(y_val, predictions, zero_division=0):.4f} | "
            f"F1={f1_score(y_val, predictions, zero_division=0):.4f}"
        )

    mlflow.log_params({
        "model": "XGBoost",
        "version": "V1_main",
        "training_fraction": TRAIN_FRACTION,
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "feature_count": len(FEATURES),
        "historical_features": True,
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.05
    })

    mlflow.log_metrics({
        "validation_roc_auc": roc_auc,
        "validation_pr_auc": pr_auc,
        "validation_best_f1": float(f1_curve[best_idx]),
        "validation_best_threshold": best_threshold
    })

    # Persist trained model in the MLflow run
    mlflow.xgboost.log_model(model, name="xgboost_v1_main")

    print("\nMLflow Run ID:", run.info.run_id)
    print("Model training and validation complete.")

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from pyspark.sql import functions as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)

SOURCE = "flightops_intelligence.ml.flight_delay_features_v1_history"
TRAINING_RUN_ID = "173c380c0c814d62941a486e1a6106bb"

# Use the threshold selected on Sep–Oct validation
threshold = float(
    mlflow.get_run(TRAINING_RUN_ID)
    .data.metrics["validation_best_threshold"]
)

# Never refit the model using test data
test_spark = (
    spark.table(SOURCE)
    .filter(
        (F.col("flight_date") >= "2025-11-01") &
        (F.col("flight_date") <= "2025-12-31")
    )
    .select("flight_date", *(FEATURES + ["delay_label"]))
)

assert test_spark.count() == 1127220

test_pd = test_spark.toPandas()

for col in CATEGORICAL:
    test_pd[col] = pd.Categorical(
        test_pd[col],
        categories=X_train[col].cat.categories
    )

for col in NUMERICAL:
    test_pd[col] = pd.to_numeric(
        test_pd[col], errors="coerce"
    ).astype("float32")

X_test = test_pd[FEATURES]
y_test = test_pd["delay_label"].astype(int)

# Predict with the EXISTING trained model
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "test_roc_auc": roc_auc_score(y_test, probabilities),
    "test_pr_auc": average_precision_score(y_test, probabilities),
    "test_precision": precision_score(y_test, predictions, zero_division=0),
    "test_recall": recall_score(y_test, predictions, zero_division=0),
    "test_f1": f1_score(y_test, predictions, zero_division=0),
    "test_brier_score": brier_score_loss(y_test, probabilities)
}

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print("=== FINAL XGBOOST V1 TEST ===")
print("Test flights:", len(y_test))
print("Threshold:", round(threshold, 4))

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print(f"\nTP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")

# Check both test months independently
test_pd["predicted_probability"] = probabilities
test_pd["predicted_label"] = predictions

print("\n=== MONTHLY TEST PERFORMANCE ===")
for month in [11, 12]:
    subset = test_pd[test_pd["flight_date"].dt.month == month]
    print(
        f"Month {month}: "
        f"rows={len(subset):,}, "
        f"ROC-AUC={roc_auc_score(subset['delay_label'], subset['predicted_probability']):.4f}, "
        f"PR-AUC={average_precision_score(subset['delay_label'], subset['predicted_probability']):.4f}"
    )

# Log evaluation without training another model
with mlflow.start_run(run_name="FlightOps_XGBoost_V1_Final_Test") as run:
    mlflow.log_metrics(metrics)
    mlflow.log_param("training_run_id", TRAINING_RUN_ID)
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("test_period", "Nov-Dec 2025")

    print("\nFinal test MLflow Run ID:", run.info.run_id)

# COMMAND ----------

import pandas as pd
import mlflow

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

# Convert existing test dates to pandas datetime
test_pd["flight_date"] = pd.to_datetime(
    test_pd["flight_date"],
    errors="raise"
)

print("=== MONTHLY TEST PERFORMANCE ===")

for month in [11, 12]:
    subset = test_pd[
        test_pd["flight_date"].dt.month == month
    ]

    print(
        f"Month {month}: "
        f"rows={len(subset):,}, "
        f"ROC-AUC={roc_auc_score(subset['delay_label'], subset['predicted_probability']):.4f}, "
        f"PR-AUC={average_precision_score(subset['delay_label'], subset['predicted_probability']):.4f}"
    )

# Log the completed test results to MLflow
with mlflow.start_run(
    run_name="FlightOps_XGBoost_V1_Final_Test"
) as run:

    mlflow.log_metrics(metrics)

    mlflow.log_params({
        "training_run_id": TRAINING_RUN_ID,
        "threshold": float(threshold),
        "test_period": "Nov-Dec 2025"
    })

    print("\nFinal test MLflow Run ID:", run.info.run_id)

print("FINAL MODEL EVALUATION COMPLETE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13.5.4 — XGBoost Refinement
# MAGIC
# MAGIC Improve delay prediction using early stopping and validation PR-AUC.
# MAGIC Compare against the existing model without using the held-out test data.

# COMMAND ----------

import numpy as np
import xgboost as xgb
import mlflow

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

# Keep the previously trained model unchanged
improved_model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.04,
    min_child_weight=10,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=3.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=8,
    max_cat_threshold=64,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=4
)

with mlflow.start_run(
    run_name="FlightOps_XGBoost_V1_Refinement"
) as run:

    improved_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    probabilities = improved_model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, probabilities)
    pr_auc = average_precision_score(y_val, probabilities)

    precision, recall, thresholds = precision_recall_curve(
        y_val, probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    predicted = (probabilities >= best_threshold).astype(int)

    results = {
        "validation_roc_auc": roc_auc,
        "validation_pr_auc": pr_auc,
        "validation_precision": precision_score(
            y_val, predicted, zero_division=0
        ),
        "validation_recall": recall_score(
            y_val, predicted, zero_division=0
        ),
        "validation_f1": f1_score(
            y_val, predicted, zero_division=0
        ),
        "best_threshold": best_threshold,
        "best_iteration": improved_model.best_iteration
    }

    mlflow.log_params({
        "model": "XGBoost",
        "experiment": "Early stopping refinement",
        "max_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.04,
        "early_stopping_rounds": 50,
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "feature_count": len(X_train.columns)
    })

    mlflow.log_metrics(results)

    print("=== REFINED XGBOOST VALIDATION ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("\n=== COMPARISON ===")
    print("Previous ROC-AUC: 0.6666")
    print(f"Updated  ROC-AUC: {roc_auc:.4f}")
    print("Previous PR-AUC:  0.2949")
    print(f"Updated  PR-AUC:  {pr_auc:.4f}")

    print("\nMLflow Run ID:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # Phase 13.5.5 — Full XGBoost Training
# MAGIC
# MAGIC Train the selected XGBoost configuration on all eligible January–August 2025 flights.
# MAGIC
# MAGIC Evaluate on September–October, optimize the warning threshold, and track the model in MLflow.

# COMMAND ----------

import gc
import pandas as pd
import xgboost as xgb
import mlflow

from pyspark.sql import functions as F

SOURCE = "flightops_intelligence.ml.flight_delay_features_v1_history"

CATEGORICAL = [
    "airline",
    "origin_airport",
    "destination_airport",
    "route"
]

NUMERICAL = [
    "month", "day_of_week", "is_weekend",
    "departure_hour", "departure_minute",
    "arrival_hour", "arrival_minute",
    "distance_miles",
    "airline_prior_flights", "airline_prior_delay_rate",
    "airport_prior_flights", "airport_prior_delay_rate",
    "route_prior_flights", "route_prior_delay_rate"
]

FEATURES = CATEGORICAL + NUMERICAL

def load_features(date_filter):
    df = spark.table(SOURCE).filter(date_filter)

    return df.select(
        *[F.col(c) for c in CATEGORICAL],
        *[F.col(c).cast("float") for c in NUMERICAL],
        F.col("delay_label").cast("int")
    ).toPandas()

# Full January-August training dataset
train_pd = load_features(
    F.col("flight_date") < "2025-09-01"
)

assert len(train_pd) == 4592366

for col in CATEGORICAL:
    train_pd[col] = train_pd[col].astype("category")

# Preserve categories for consistent inference
category_levels = {
    col: train_pd[col].cat.categories
    for col in CATEGORICAL
}

X_train = train_pd[FEATURES]
y_train = train_pd["delay_label"]

print("Full training records:", len(X_train))
print("Features:", len(FEATURES))

full_model = xgb.XGBClassifier(
    n_estimators=350,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=8,
    max_cat_threshold=64,
    random_state=42,
    n_jobs=4
)

with mlflow.start_run(
    run_name="FlightOps_XGBoost_V1_Full_Training"
) as run:

    full_model.fit(X_train, y_train)

    mlflow.log_params({
        "model": "XGBoost",
        "training_fraction": 1.0,
        "training_rows": len(X_train),
        "feature_count": len(FEATURES),
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.05,
        "training_period": "Jan-Aug 2025"
    })

    mlflow.xgboost.log_model(
        full_model,
        name="xgboost_v1_full"
    )

    full_training_run_id = run.info.run_id

print("FULL TRAINING COMPLETE")
print("MLflow Run ID:", full_training_run_id)

# Release training data before loading validation
del train_pd, X_train, y_train
gc.collect()

# COMMAND ----------

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

val_pd = load_features(
    (F.col("flight_date") >= "2025-09-01") &
    (F.col("flight_date") < "2025-11-01")
)

assert len(val_pd) == 1159898

for col in CATEGORICAL:
    val_pd[col] = pd.Categorical(
        val_pd[col],
        categories=category_levels[col]
    )

X_val = val_pd[FEATURES]
y_val = val_pd["delay_label"]

probabilities = full_model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, probabilities)
pr_auc = average_precision_score(y_val, probabilities)

precision, recall, thresholds = precision_recall_curve(
    y_val, probabilities
)

f1 = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12
    )
)

best_idx = int(np.argmax(f1))
best_threshold = float(thresholds[best_idx])

with mlflow.start_run(run_id=full_training_run_id):
    mlflow.log_metrics({
        "validation_roc_auc": roc_auc,
        "validation_pr_auc": pr_auc,
        "validation_best_f1": float(f1[best_idx]),
        "validation_best_threshold": best_threshold
    })

print("=== FULL-DATA XGBOOST VALIDATION ===")
print(f"Training records: 4,592,366")
print(f"Validation records: {len(X_val):,}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Best F1: {f1[best_idx]:.4f}")
print(f"Best threshold: {best_threshold:.4f}")

print("\n=== COMPARISON ===")
print(f"25% training PR-AUC: 0.2949")
print(f"100% training PR-AUC: {pr_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 13.6 — Final Model Evaluation
# MAGIC
# MAGIC Evaluate the trained XGBoost V1 on November–December 2025.
# MAGIC Report overall and monthly performance without retraining or changing the selected threshold.

# COMMAND ----------

import gc
import pandas as pd
import mlflow

from pyspark.sql import functions as F
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

# Reuse the validation-selected threshold
threshold = best_threshold

# Release validation data to reduce memory use
del X_val, y_val, val_pd
gc.collect()

# Load November–December test data
test_pd = load_features(
    (F.col("flight_date") >= "2025-11-01") &
    (F.col("flight_date") < "2026-01-01")
)

assert len(test_pd) == 1127220

for col in CATEGORICAL:
    test_pd[col] = pd.Categorical(
        test_pd[col],
        categories=category_levels[col]
    )

X_test = test_pd[FEATURES]
y_test = test_pd["delay_label"].astype(int)

# Existing model — no retraining
probabilities = full_model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "roc_auc": roc_auc_score(y_test, probabilities),
    "pr_auc": average_precision_score(y_test, probabilities),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "f1": f1_score(y_test, predictions, zero_division=0)
}

print("=== XGBOOST V1 — FULL DATA TEST ===")
print(f"Training records: 4,592,366")
print(f"Test records: {len(X_test):,}")
print(f"Threshold: {threshold:.4f}")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Record test metrics against the original full-training run
with mlflow.start_run(run_id=full_training_run_id):
    mlflow.log_metrics({
        f"final_test_{k}": v for k, v in metrics.items()
    })

print("\nEvaluation saved to MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 13.7 — Model Packaging
# MAGIC
# MAGIC Package the evaluated XGBoost V1 with preprocessing, threshold, and MLflow input/output signature. Verify predictions after reloading the model.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn

from mlflow.models import infer_signature

MODEL_THRESHOLD = float(best_threshold)

class FlightOpsPredictor(mlflow.pyfunc.PythonModel):

    def __init__(self, estimator, categories, categorical, numerical, threshold):
        self.estimator = estimator
        self.categories = {
            col: list(categories[col])
            for col in categorical
        }
        self.categorical = list(categorical)
        self.numerical = list(numerical)
        self.threshold = float(threshold)

    def predict(self, context, model_input, params=None):
        df = model_input.copy()

        for col in self.categorical:
            df[col] = pd.Categorical(
                df[col].astype("string"),
                categories=self.categories[col]
            )

        for col in self.numerical:
            df[col] = pd.to_numeric(
                df[col], errors="raise"
            ).astype("float32")

        features = df[self.categorical + self.numerical]
        probabilities = self.estimator.predict_proba(features)[:, 1]

        return pd.DataFrame({
            "delay_probability": probabilities.astype("float64"),
            "delay_alert": (
                probabilities >= self.threshold
            ).astype("int64")
        })


predictor = FlightOpsPredictor(
    estimator=full_model,
    categories=category_levels,
    categorical=CATEGORICAL,
    numerical=NUMERICAL,
    threshold=MODEL_THRESHOLD
)

# Small example for defining and checking the schema
example = X_test.head(5).copy()

for col in CATEGORICAL:
    example[col] = example[col].astype(str)

for col in NUMERICAL:
    example[col] = example[col].astype("float32")

expected = predictor.predict(None, example)
signature = infer_signature(example, expected)

with mlflow.start_run(
    run_name="FlightOps_XGBoost_V1_Packaged"
) as run:

    model_info = mlflow.pyfunc.log_model(
        name="flight_delay_predictor_v1",
        python_model=predictor,
        input_example=example,
        signature=signature,
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"xgboost=={xgb.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}"
        ],
        metadata={
            "source_training_run": "3a8e5e36524742fd8edf1cea4b896828",
            "model_version": "V1",
            "prediction_point": "60 minutes before scheduled departure",
            "status": "experimental"
        }
    )

    mlflow.log_metric("prediction_threshold", MODEL_THRESHOLD)
    mlflow.log_param("feature_count", len(FEATURES))

    packaged_run_id = run.info.run_id

# Verify reloading, without retraining
loaded = mlflow.pyfunc.load_model(model_info.model_uri)
actual = loaded.predict(example)

np.testing.assert_allclose(
    actual["delay_probability"],
    expected["delay_probability"],
    rtol=1e-5
)

assert actual["delay_alert"].equals(
    expected["delay_alert"]
)

print("MODEL PACKAGING VERIFIED")
print("Model URI:", model_info.model_uri)
print("MLflow Run ID:", packaged_run_id)
print("Input features:", len(FEATURES))
print("Threshold:", round(MODEL_THRESHOLD, 4))
print(actual)

# COMMAND ----------

