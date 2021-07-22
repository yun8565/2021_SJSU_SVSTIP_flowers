import sys
import tempfile

import requests
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


def main():
    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # Nice way to write a tmp file onto the system
    temp_csv_file = tempfile.mktemp()
    with open(temp_csv_file, mode="wb") as f:
        data_https = requests.get("https://teaching.mrsharky.com/data/iris.data")
        f.write(data_https.content)

    fisher_df_orig = spark.read.csv(temp_csv_file, inferSchema="true", header="false")
    fisher_df_orig.persist(StorageLevel.MEMORY_ONLY)
    fisher_df_orig.show()

    # Change column names
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]
    fisher_df_orig = fisher_df_orig.toDF(*column_names)

    # Randomize order of rows
    fisher_df_orig = fisher_df_orig.withColumn("random", rand()).orderBy("random")

    # Make a copy of the DataFrame (so we can start over)
    fisher_df = fisher_df_orig
    fisher_df.createOrReplaceTempView("fisher")
    print_heading("Original Dataset")
    fisher_df.show()

    # Get the average of all continuous fields
    print_heading("Population Average")
    fisher_avg_df = spark.sql(
        """
        SELECT
                AVG(sepal_length) AS avg_sepal_length
                , AVG(sepal_width) AS avg_sepal_width
                , AVG(petal_length) AS avg_petal_length
                , AVG(petal_width) AS avg_petal_width
            FROM fisher
        """
    )
    fisher_avg_df.show()

    # Get the average of all continuous fields by class
    print_heading("Average by class")
    fisher_avg_df = spark.sql(
        """
        SELECT
                class
                , AVG(sepal_length) AS avg_sepal_length
                , AVG(sepal_width) AS avg_sepal_width
                , AVG(petal_length) AS avg_petal_length
                , AVG(petal_width) AS avg_petal_width
            FROM fisher
            GROUP BY class
            ORDER BY class
        """
    )
    fisher_avg_df.show()

    # Build a features vector
    print_heading("VectorAssembler")
    vector_assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features",
    )
    fisher_df = vector_assembler.transform(fisher_df)
    fisher_df.show()

    # Label String Indexer
    print_heading("StringIndexer")
    label_indexer = StringIndexer(inputCol="class", outputCol="class_idx")
    label_indexer_model = label_indexer.fit(fisher_df)
    fisher_df = label_indexer_model.transform(fisher_df)
    fisher_df.show()

    # Random forest
    print_heading("RandomForestClassifier")
    random_forest = RandomForestClassifier(
        labelCol="class_idx",
        featuresCol="features",
    )
    random_forest_model = random_forest.fit(fisher_df)
    fisher_df_predicted = random_forest_model.transform(fisher_df)
    fisher_df_predicted.createOrReplaceTempView("predicted")
    fisher_df_predicted.show()

    print_heading("Accuracy")
    fisher_df_accuracy = spark.sql(
        """
        SELECT
                SUM(correct)/COUNT(*) AS accuracy
            FROM
                (SELECT
                        CASE WHEN prediction == class_idx THEN 1
                        ELSE 0 END AS correct
                    FROM predicted) AS TMP
              """
    )
    fisher_df_accuracy.show()

    # Pipeline
    print_heading("Pipeline")
    fisher_df = fisher_df_orig
    fisher_df.createOrReplaceTempView("fisher")
    pipeline = Pipeline(stages=[vector_assembler, label_indexer, random_forest])
    model = pipeline.fit(fisher_df)
    fisher_df_predicted = model.transform(fisher_df)
    fisher_df_predicted.show()
    return


if __name__ == "__main__":
    sys.exit(main())
