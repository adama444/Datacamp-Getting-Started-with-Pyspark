# Module importation
from pyspark.sql.context import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Initialize the spark session
spark = (
    SparkSession.builder.appName("Datacamp Pyspark Tutorial")
    .config("spark.log.level", "ERROR")
    .getOrCreate()
)

# Load the data
df = spark.read.csv("datacamp_ecommerce.csv", header=True)

# Print 5 first rows
df.show(5, 0)

# Show number of rows
rows_number = df.count()
print("number of rows: ", rows_number)

# EXPLORATORY DATA ANALYSIS
# Show number of unique customers
unique_customers_number = df.select("CustomerID").distinct().count()
print("number of unique customers: ", unique_customers_number)

# The country which have the most customers
df.groupBy("Country").agg(countDistinct("CustomerID").alias("Nb_Customers")).orderBy(
    desc("Nb_Customers")
).show(5)

# The last and the first purchase
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df = df.withColumn("date", to_timestamp(df.InvoiceDate, "dd/MM/yy HH:mm"))
df.select(min("date").alias("First_Purchase")).show()
df.select(max("date").alias("Last_Purchase")).show()

# DATA PRE-PROCESSING
# Computation of recency
last_purchase_timestamp = (
    df.select(min("date").alias("First_Purchase"))
    .collect()[0]
    .asDict()
    .get("First_Purchase")
)

df = df.withColumn("from_date", lit(last_purchase_timestamp))
df.select('from_date').distinct().show()

df2 = df.withColumn("from_date", to_timestamp(col("from_date"))).withColumn(
    "recency", col("date").cast("long") - col("from_date").cast("long")
)
df2 = df2.join(
    df2.groupBy("CustomerID").agg(max("recency").alias("recency")),
    on="recency",
    how="leftsemi",
)
df2.show(5, 0)

# Computation of frequency
df_freq = df2.groupBy("CustomerID").agg(count("InvoiceDate").alias("frequency"))
df_freq.show(5, 0)

# Join the frequency data with the last dataframe
df3 = df2.join(df_freq, on="CustomerID", how="inner")
df3.show(5, 0)

# Computation of monetary value
m_val = df3.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))
m_val = m_val.groupBy("CustomerID").agg(sum("TotalAmount").alias("monetary_value"))

# Merge all data
finaldf = m_val.join(df3, on="CustomerID", how="inner")
finaldf = finaldf.select(
    ["recency", "frequency", "monetary_value", "CustomerID"]
).distinct()
finaldf.show(5, 0)

# Vectorize columns into features and Standardize the data
assembled_data = VectorAssembler(
    inputCols=["recency", "frequency", "monetary_value"], outputCol="features"
).transform(finaldf)

data_scale_output = (
    StandardScaler(inputCol="features", outputCol="standardized")
    .fit(assembled_data)
    .transform(assembled_data)
)
data_scale_output.select("standardized").show(2, truncate=False)

# Build the machine learning model
cost = np.zeros(10)

evaluator = ClusteringEvaluator(
    predictionCol="prediction",
    featuresCol="standardized",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean",
)

# Using Eblow method to find the best value of k
for i in range(2, 10):
    KMeans_algo = KMeans(featuresCol="standardized", k=i)
    KMeans_fit = KMeans_algo.fit(data_scale_output)
    output = KMeans_fit.transform(data_scale_output)
    cost[i] = KMeans_fit.summary.trainingCost

df_cost = pd.DataFrame(cost[2:])
df_cost.columns = ["cost"]

new_col = range(2, 10)
df_cost.insert(0, "cluster", new_col)

plt.plot(df_cost.cluster, df_cost.cost)
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.title("Elbow Curve")
plt.savefig("fig.png")

# Building k-means algorithm
KMeans_algo = KMeans(featuresCol="standardized", k=4)
KMeans_fit = KMeans_algo.fit(data_scale_output)

preds = KMeans_fit.transform(data_scale_output)
preds.show(5)

# Analysis of clusters
df_viz = preds.select(
    "recency", "frequency", 
    "monetary_value", "prediction"
).toPandas()
avg_df = df_viz.groupby(["prediction"], as_index=False).mean()

list1 = ["recency", "frequency", "monetary_value"]
for i in list1:
    plt.clf()
    sns.barplot(x="prediction", y=str(i), data=avg_df)
    plt.savefig(f"fig-{i}.png")
