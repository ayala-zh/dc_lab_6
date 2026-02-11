#!/usr/bin/env python3
"""
Customer Churn Prediction using Spark ML Pipeline
Distributed Computing Lab 6
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time

# Initialize Spark Session
print("=" * 60)
print("Initializing Spark Session...")
print("=" * 60)

spark = SparkSession.builder \
    .appName("CustomerChurnPipeline") \
    .getOrCreate()

# Record start time
start_time = time.time()

# Load Dataset from HDFS
print("\n" + "=" * 60)
print("Loading Dataset from HDFS...")
print("=" * 60)

data = spark.read.csv(
    "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
    header=True,
    inferSchema=True
)

# Display dataset info
print(f"\nTotal Records: {data.count()}")
print(f"Total Features: {len(data.columns)}")
print("\nSchema:")
data.printSchema()

print("\nFirst 5 rows:")
data.show(5)

# Check class distribution
print("\nChurn Distribution:")
data.groupBy("Exited").count().show()

# Split data into train and test
print("\n" + "=" * 60)
print("Splitting Data: 80% Train, 20% Test")
print("=" * 60)

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Training set: {train_data.count()} records")
print(f"Test set: {test_data.count()} records")

# ============================================================
# PIPELINE STAGE 1: Categorical Encoding
# ============================================================
print("\n" + "=" * 60)
print("Building ML Pipeline Stages...")
print("=" * 60)

# String Indexer for Geography
geo_indexer = StringIndexer(
    inputCol="Geography",
    outputCol="GeographyIndex",
    handleInvalid="keep"
)

# String Indexer for Gender
gender_indexer = StringIndexer(
    inputCol="Gender",
    outputCol="GenderIndex",
    handleInvalid="keep"
)

print("✓ Stage 1: Categorical Indexing (StringIndexer)")

# ============================================================
# PIPELINE STAGE 2: One-Hot Encoding
# ============================================================

encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec", "GenderVec"]
)

print("✓ Stage 2: One-Hot Encoding")

# ============================================================
# PIPELINE STAGE 3: Feature Vector Assembly
# ============================================================

# Numerical features
numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"
]

# Combine numerical and encoded categorical features
assembler = VectorAssembler(
    inputCols=[
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary"
    ],  # No GeographyVec, GenderVec
    outputCol="features"
)

print("✓ Stage 3: Feature Assembly")
print(f"   Features: {', '.join(numerical_features + ['Geography', 'Gender'])}")

# ============================================================
# PIPELINE STAGE 4: Feature Scaling
# ============================================================

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=False
)

print("✓ Stage 4: Standard Scaling")

# ============================================================
# PIPELINE STAGE 5: Model Training
# ============================================================

# Logistic Regression Model
lr = LogisticRegression(
    labelCol="Exited",
    featuresCol="scaledFeatures",
    maxIter=100,
    regParam=0.01
)

print("✓ Stage 5: Logistic Regression Model")

# ============================================================
# Build Complete Pipeline
# ============================================================

pipeline = Pipeline(stages=[
    assembler,  # Only assembler
    scaler,
    lr
])

print("\n" + "=" * 60)
print("Training ML Pipeline on Distributed Cluster...")
print("=" * 60)

# Train the model
train_start = time.time()
model = pipeline.fit(train_data)
train_end = time.time()

print(f"✓ Training completed in {train_end - train_start:.2f} seconds")

# ============================================================
# PIPELINE STAGE 6: Prediction
# ============================================================

print("\n" + "=" * 60)
print("Making Predictions on Test Set...")
print("=" * 60)

predictions = model.transform(test_data)

print("\nSample Predictions:")
predictions.select(
    "CustomerId",
    "Exited",
    "prediction",
    "probability"
).show(10, truncate=False)

# ============================================================
# PIPELINE STAGE 7: Evaluation
# ============================================================

print("\n" + "=" * 60)
print("Model Evaluation Metrics")
print("=" * 60)

# Accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = accuracy_evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Precision
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

precision = precision_evaluator.evaluate(predictions)
print(f"Precision: {precision:.4f}")

# Recall
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="weightedRecall"
)

recall = recall_evaluator.evaluate(predictions)
print(f"Recall: {recall:.4f}")

# F1 Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="f1"
)

f1 = f1_evaluator.evaluate(predictions)
print(f"F1 Score: {f1:.4f}")

# AUC-ROC
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="Exited",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = auc_evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
predictions.groupBy("Exited", "prediction").count().show()

# Total execution time
end_time = time.time()
total_time = end_time - start_time

print("\n" + "=" * 60)
print(f"Pipeline Execution Summary")
print("=" * 60)
print(f"Total Execution Time: {total_time:.2f} seconds")
print(f"Training Time: {train_end - train_start:.2f} seconds")
print(f"Records Processed: {data.count()}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print("=" * 60)

# Stop Spark session
spark.stop()
