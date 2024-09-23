# Databricks notebook source
# DBTITLE 1,Libraries
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql.types import FloatType

# COMMAND ----------

# DBTITLE 1,Variables
env = "dev"
catalog = f"mlops_{env}"
database = "churn_data"
table = "telco_customer_churn"

# COMMAND ----------

data = spark.sql(f"SELECT * FROM {catalog}.{database}.{table}")

# COMMAND ----------

# DBTITLE 1,Label Encoding
columns_to_encode = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn','InternetService']

# Create a list of StringIndexer stages
indexers = [StringIndexer(inputCol=col, outputCol=col+"_encoded") for col in columns_to_encode]

# Build the pipeline with all the indexers
pipeline = Pipeline(stages=indexers)

# Fit the pipeline and transform the data
encoded_df = pipeline.fit(data).transform(data)
encoded_df = encoded_df.drop(*columns_to_encode)
encoded_df.display()

# COMMAND ----------

col_to_drop = ['gender','SeniorCitizen','Contract','PaymentMethod']
final_df = encoded_df.drop(*col_to_drop)
final_df.printSchema()

# COMMAND ----------

final_df = final_df.withColumn("TotalCharges", final_df["TotalCharges"].cast(FloatType()))

# COMMAND ----------

fe = FeatureEngineeringClient()

fe.create_table(
    name=f"{catalog}.feature_engineering.features",
    primary_keys=["customerID"],
    df=final_df,
    schema=final_df.schema,
    description="telco churn features"
)


# final_df.write.mode("overwrite").saveAsTable(f"{catalog}.{database}.")}")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE mlops_dev.feature_engineering.features
