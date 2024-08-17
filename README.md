# Datacamp-Getting-Started-with-Pyspark

## Why Pyspark ?

You need to learn a framework that allows you to manipulate datasets on top of a distributed processing  system, as most data-driven organizations will require you to do so. PySpark is a great place to get started, since its syntax is simple and can be picked up easily if you are already familiar with Python. it can handle larger amounts of data than frameworks like pandas.

## Purpose of the project

Using pyspark to execute an end-to-end customer segmentation project.

By the end of this tutorial, you will be familiar with the following concepts:

- Reading csv files with PySpark
- Exploratory Data Analysis with PySpark
- Grouping and sorting data
- Performing arithmetic operations
- Aggregating datasets
- Data Pre-Processing with PySpark
- Working with datetime values
- Type conversion
- Joining two dataframes
- The rank() function
- PySpark Machine Learning
- Creating a feature vector
- Standardizing data
- Building a K-Means clustering model
- Interpreting the model

## Cluster Analysis Results

- **Cluster 0**: Customers in this segment display high recency and frequency and monetary value. They are customers who are likely to be one of our best customers.
- **Cluster 1**: Users in this cluster display high recency but haven’t been seen spending much on the platform. They also don’t visit the site often. This indicates that they might be newer customers who have just started doing business with the company.
- **Cluster 2**: Customers in this segment display medium recency and frequency but don't spend much money on the platform. They are low potential customers.
- **Cluster 3**: The final segment comprises users who display high recency and are rarely on the platform. However, they spend much money on the platform, which might mean that they tend to select expensive items in each purchase.