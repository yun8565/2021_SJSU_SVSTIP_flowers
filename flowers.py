import sys

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import rand
from pyspark.ml.classification import RandomForestClassifier
from numpy import allclose

def main():
    print("Hello")

    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    flowers_df = spark.read.csv("iris.data", inferSchema="true", header="false")
    flowers_df = flowers_df.toDF(
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class" 
    )
    flowers_df.createOrReplaceTempView("flowers")
    flowers_df.persist(StorageLevel.DISK_ONLY)
    flowers_df.show()

    results_overall1 = spark.sql(
        """
        SELECT
                class
                , COUNT(*) AS cnt
                , AVG(sepal_length) AS sepal_length_avg
                , AVG(sepal_width) AS sepal_width_avg
                , AVG(petal_length) AS petal_length_avg
                , AVG(petal_width) AS petal_width_avg
            FROM flowers
            GROUP BY class
        """
    )
    results_overall1.show()

    results_overall2 = spark.sql(
        """
        SELECT
                COUNT(*) AS cnt
                , AVG(sepal_length) AS sepal_length_avg
                , AVG(sepal_width) AS sepal_width_avg
                , AVG(petal_length) AS petal_length_avg
                , AVG(petal_width) AS petal_width_avg
            FROM flowers
        """
    )
    results_overall2.show()


    results3 = spark.sql(
        """
        SELECT
                *
                , rand() AS random
            FROM flowers
        """
    )
    results3.show()

    results4 = flowers_df.withColumn('random', rand(seed=42)).orderBy("random")
    results4.show()

    # Build a model!!
    vecAssembler = VectorAssembler(outputCol="features")
    vecAssembler.setInputCols([
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ])
    flowers_df = vecAssembler.transform(flowers_df)
    flowers_df.show()

    # String Indexer
    stringIndexer = StringIndexer(inputCol="class", outputCol="label")
    stringIndexer_model = stringIndexer.fit(flowers_df)
    flowers_df = stringIndexer_model.transform(flowers_df)
    flowers_df.show()

    # Random Forest
    rf = RandomForestClassifier(numTrees=3, maxDepth=5, labelCol="label", seed=42)
    rf_model = rf.fit(flowers_df)
    flowers_df = rf_model.transform(flowers_df)
    flowers_df.show()

    # Accuracy
    flowers_df.createOrReplaceTempView("flowers")
    flowers_df = spark.sql(
        """
        SELECT
                AVG(CASE WHEN label = prediction THEN 1
                    ELSE 0 END) AS accuracy
            FROM flowers
        """
    )
    flowers_df.show()

    # Pipeline
    print("pipeline running")
    pipeline = Pipeline(
        stages=[vecAssembler, stringIndexer, rf]
    )
    model = pipeline.fit(flowers_df)
    flowers_new = model.transform(flowers_df)
    flowers_new.show()

if __name__ == "__main__":
    sys.exit(main())
