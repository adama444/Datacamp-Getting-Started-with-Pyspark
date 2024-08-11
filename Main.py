# Module importation
from pyspark.sql.context import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np


# Initialize the spark session
spark = (
    SparkSession.builder.appName("Datacamp Pyspark Tutorial")
    .config("spark.log.level", 'ERROR')
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
df = df.withColumn("date", to_timestamp(df.InvoiceDate, 'dd/MM/yyyy HH:mm'))
df.select(min('date').alias('First_Purchase')).show()
df.select(max('date').alias('Last_Purchase')).show()

"""

df = df.withColumn("from_date", lit("12/1/10 08:26"))
df = df.withColumn('from_date',to_timestamp("from_date", 'yy/MM/dd HH:mm'))

df2=df.withColumn('from_date',to_timestamp(col('from_date'))).withColumn('recency',col("date").cast("long") - col('from_date').cast("long"))

df2 = df2.join(df2.groupBy('CustomerID').agg(max('recency').alias('recency')),on='recency',how='leftsemi')

df2.show(5,0)
df2.printSchema() # To show info of dataframe like info() in pandas

df_freq = df2.groupBy('CustomerID').agg(count('InvoiceDate').alias('frequency'))
df_freq.show(5,0)

df3 = df2.join(df_freq,on='CustomerID',how='inner')
df3.printSchema()

m_val = df3.withColumn('TotalAmount',col("Quantity") * col("UnitPrice"))

m_val = m_val.groupBy('CustomerID').agg(sum('TotalAmount').alias('monetary_value'))

finaldf = m_val.join(df3,on='CustomerID',how='inner')

finaldf = finaldf.select(['recency','frequency','monetary_value','CustomerID']).distinct()

assemble=VectorAssembler(inputCols=[
    'recency','frequency','monetary_value'
], outputCol='features')

assembled_data=assemble.transform(finaldf)

scale=StandardScaler(inputCol='features',outputCol='standardized')

data_scale=scale.fit(assembled_data)

data_scale_output=data_scale.transform(assembled_data)

data_scale_output.select('standardized').show(2,truncate=False)

cost = np.zeros(10)

evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2,10):
    KMeans_algo=KMeans(featuresCol='standardized', k=i)
    KMeans_fit=KMeans_algo.fit(data_scale_output)
    output=KMeans_fit.transform(data_scale_output)
    cost[i] = KMeans_fit.summary.trainingCost

"""
