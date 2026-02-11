# Bank Customer Churn Prediction with Spark ML on EMR

## Setup Instructions

### 1. Upload Data to HDFS
```bash
# Copy dataset to master node
scp -i C:\labsuser.pem Churn_Modelling.csv hadoop@ec2-3-93-147-36.compute-1.amazonaws.com:/home/hadoop/

# Upload to HDFS
hdfs dfs -put Churn_Modelling.csv /user/hadoop/
hdfs dfs -ls /user/hadoop/
```

### 2. Submit Spark Job
```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --num-executors 2 \
  --executor-memory 2G \
  --executor-cores 2 \
  churn_pipeline.py
```

## Quick Start
```python
# churn_pipeline.py - Full pipeline execution
python3 churn_pipeline.py
```

## Requirements
- EMR Cluster (1 master, 2 core nodes)
- Spark 2.4+/3.x
- Dataset: Churn_Modelling.csv (10K records)
- Python 3.6+
