# Databricks notebook source
# DBTITLE 1,Libraries
from databricks.feature_engineering import FeatureLookup
from databricks.feature_store import FeatureStoreClient
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from mlflow import log_param, log_metric, log_artifact, start_run
import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# COMMAND ----------

# DBTITLE 1,Variables
env = "dev"
catalog = f"mlops_{env}"
database = "feature_engineering"
table = "features"

# COMMAND ----------

# model_feature_lookups = [FeatureLookup(table_name=f"{catalog}.{database}.{table}", lookup_key="customerID")]
fs = FeatureStoreClient()
feature_table = f"{catalog}.{database}.{table}"
df_features = fs.read_table(feature_table).dropna()
df = df_features.drop("customerID")

# COMMAND ----------

input_cols = [col for col in df.columns if col != 'Churn_encoded']
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

data = assembler.transform(df)
lr = LogisticRegression(labelCol="Churn_encoded", featuresCol="features") 

lr_model = lr.fit(data)

# COMMAND ----------

predictions = lr_model.transform(data)
evaluator = BinaryClassificationEvaluator(labelCol='Churn_encoded')
accuracy = evaluator.evaluate(predictions)

# COMMAND ----------


# Define input schema for all feature columns
input_schema = Schema([
    ColSpec("long", "tenure"),
    ColSpec("double", "MonthlyCharges"),
    ColSpec("float", "TotalCharges"),
    ColSpec("double", "Partner_encoded"),
    ColSpec("double", "Dependents_encoded"),
    ColSpec("double", "PhoneService_encoded"),
    ColSpec("double", "MultipleLines_encoded"),
    ColSpec("double", "OnlineSecurity_encoded"),
    ColSpec("double", "OnlineBackup_encoded"),
    ColSpec("double", "DeviceProtection_encoded")
    ,
    ColSpec("double", "TechSupport_encoded"),
    ColSpec("double", "StreamingTV_encoded"),
    ColSpec("double", "StreamingMovies_encoded"),
    ColSpec("double", "PaperlessBilling_encoded"),
    ColSpec("double", "InternetService_encoded"),
])

# Define output schema for the label
output_schema = Schema([
    ColSpec("double", "Churn_encoded")
])

# Create a model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)


# COMMAND ----------

with mlflow.start_run() as run:
    log_param("maxIter", lr.getMaxIter())
    log_param("regParam", lr.getRegParam())
    log_metric("accuracy", accuracy)

    mlflow.spark.log_model(lr_model, "logistic_regression_model",signature=signature)

    model_uri = f"runs:/{run.info.run_id}/logistic_regression_model"

    registered_model  = mlflow.register_model(
        model_uri = model_uri, 
        name= "telco_churn_model"
    )

    version = registered_model.version

client = MlflowClient()

# Add a description to the registered model
client.update_model_version(
    name=registered_model.name,
    version = version,
    description="This linear regression model predicts customer churn using various customer features."
)

# COMMAND ----------

rf = RandomForestClassifier(labelCol="Churn_encoded", featuresCol="features", numTrees=20)
rf_model = rf.fit(data)

# COMMAND ----------

predictions = rf_model.transform(data)
evaluator = BinaryClassificationEvaluator(labelCol='Churn_encoded')
accuracy = evaluator.evaluate(predictions)

# COMMAND ----------

with mlflow.start_run() as run:
    log_param("maxIter", rf.getNumTrees())
    log_param("regParam", rf.getMaxDepth())
    log_metric("accuracy", accuracy)

    mlflow.spark.log_model(rf_model, "random_forest_model",signature=signature)

    model_uri = f"runs:/{run.info.run_id}/random_forest_model"

    registered_model = mlflow.register_model(
        model_uri = model_uri, 
        name= "telco_churn_model"
    )
    
    version = registered_model.version

client = MlflowClient()

# Add a description to the registered model
client.update_model_version(
    name=registered_model.name,
    version = version,
    description= "This random forest model predicts customer churn using various customer features."
)
