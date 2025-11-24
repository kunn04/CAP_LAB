from pyspark.sql import SparkSession
import os

# Dirección del master del cluster Kubernetes
SPARK_URL = "spark://spark-master.spark.svc.cluster.local:7077"

IP = os.popen("hostname -I").read().strip().split()[0]

spark = (SparkSession.builder
         .appName("Spark Demo")
         .master(SPARK_URL)
         .config("spark.driver.host", IP)
         .config("spark.executor.memory", "512M")
         .config("spark.cores.max", 2)
         .config("spark.executor.instances", 2)
         .getOrCreate())

sc = spark.sparkContext

# Ejemplo
rdd = sc.parallelize([1, 2, 3, 4, 5])
result = rdd.map(lambda x: x * 2).collect()

print("Resultado:", result)

spark.stop()