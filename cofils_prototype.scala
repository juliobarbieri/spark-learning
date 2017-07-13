import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import scala.io.Source._
import org.apache.spark.sql.SparkSession

import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, IndexedRow}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

case class DatasetRow(label:Option[Double],features:Option[Vector[Double]]) 
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._


object Test extends Serializable {
    def getIndex(array: Array[IndexedRow], index: Int) : org.apache.spark.mllib.linalg.Vector = {
        for (element <- array) {
            if (element.index == index) {
                return element.vector
            }
        }
        return Vectors.dense(0.0)
    }
    def calcuateAverage(dataset :Tuple, normalization :Int) = Array[Double] {
        val schema = StructType(
            Array(
            StructField("user", IntType, true),
            StructField("item", IntType, true),
            StructField("rating", DoubleType, true), 
            StructField("time", DoubleType, true) )
        val rdd = sc.parallelize(dataset).map (x => Row(x._1, x._2, x._3, x._4))
        val df = sqlContext.createDataFrame(rdd, schema)
        
        if (normalization == 1) {
            userDF = df.select("user", "rating").groupBy("user").agg("user", avg("rating"))
            return userDF.select("rating").rdd.map(r => r(0)).collect()
        }
        else {
            itemDF = df.select("item", "rating").groupBy("item").agg("user", avg("rating"))
            return userDF.select("rating").rdd.map(r => r(0)).collect()
        }

    }
    def main(args: Array[String]) {
        
        val filename = "/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/u.data"
        val f = sc.textFile(filename) 
        val dataset = f.map(_.split('\t') match {
                case Array(userId, itemId, rating, time) => (userId.toInt, itemId.toInt, rating.toDouble, time.toDouble)
            }
        )
        
        val coMatrix = new CoordinateMatrix(dataset.map {
            case (userId, itemId, rating, time) => MatrixEntry(userId, itemId, rating)
        })
        
        val mat: IndexedRowMatrix = coMatrix.toIndexedRowMatrix
        
        // Apply SVD
        // Compute the top 20 singular values and corresponding singular vectors.
        val svd: SingularValueDecomposition[IndexedRowMatrix, Matrix] = mat.computeSVD(8, computeU = true)
        val U: IndexedRowMatrix = svd.U // The U factor is a RowMatrix.
        val s: org.apache.spark.mllib.linalg.Vector = svd.s // The singular values are stored in a local dense vector.
        val V: Matrix = svd.V // The V factor is a local dense matrix.
        
        val Ui = U.rows.collect
        val Vi = Array() ++ V.rowIter
        
        
        var arrayData = dataset.map(line => Vectors.dense(line._3).toArray ++ getIndex(Ui, line._1).toArray ++ Vi(line._2).toArray)
        
        val rdd = arrayData.map(x => (x(0), org.apache.spark.ml .linalg.Vectors.dense(x.drop(1))))
        val df = rdd.toDF("label", "features")
        df.show(5, true)
        
        // Apply Random Forest
        // Load and parse the data file, converting it to a DataFrame.
        val data = df

        val numTrees = 50

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(data)

        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))

        // Train a RandomForest model.
        val rf = new RandomForestRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")
          .setNumTrees(numTrees)

        // Chain indexer and forest in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, rf))

        // Train model. This also runs the indexer.
        val model = pipeline.fit(trainingData)

        // Make predictions.
        val predictions = model.transform(testData)

        // Select example rows to display.
        predictions.select("prediction", "label", "features").show(5)

        // Select (prediction, true label) and compute test error.
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        val rmse = evaluator.evaluate(predictions)
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)

        //val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
        //println("Learned regression forest model:\n" + rfModel.toDebugString)
        
        //System.exit(0)
    }
}


