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

case class DatasetRow(user:Option[Int], item:Option[Int], label:Option[Double],features:Option[Vector[Double]]) 
//case class PredictRow(user:Option[Int], item:Option[Int], label:Option[Double],features:Option[Vector[Double]]) 
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
    def calculateAverage(dataset :RDD[(Int, Int, Double, Double)], normalization :Int) : Array[Double] = {
        val schema = StructType(
            Array(
            StructField("user", IntegerType, true),
            StructField("item", IntegerType, true),
            StructField("rating", DoubleType, true), 
            StructField("time", DoubleType, true) ))
        val rdd = dataset.map(x => Row(x._1, x._2, x._3, x._4))
        val df = sqlContext.createDataFrame(rdd, schema)
        //val globalAvg = df.select("rating").agg(avg("rating"))
        
        if (normalization == 1) {
            val userDF = df.select("user", "rating").orderBy("user").groupBy("user").agg($"user", avg("rating"))
            return userDF.select("avg(rating)").rdd.map(r => r(0).asInstanceOf[Double]).collect()
        }
        else {
            val itemDF = df.select("item", "rating").orderBy("item").groupBy("item").agg($"item", avg("rating"))
            return itemDF.select("avg(rating)").rdd.map(r => r(0).asInstanceOf[Double]).collect()
        }

    }
    def main(args: Array[String]) {
        
        // Load Train files
        val trainFilename = "/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/folds/u1.base"
        val fTrain = sc.textFile(trainFilename)
        val datasetTrain = fTrain.map(_.split('\t') match {
                case Array(userId, itemId, rating, time) => (userId.toInt, itemId.toInt, rating.toDouble, time.toDouble)
            }
        )
        
        // Load Test files
        val testFilename = "/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/folds/u1.test"
        val fTest = sc.textFile(testFilename)
        val datasetTest = fTest.map(_.split('\t') match {
                case Array(userId, itemId, rating, time) => (userId.toInt, itemId.toInt, rating.toDouble, time.toDouble)
            }
        )
        
        // Normalization
        val normalization = 1
        val avgs = calculateAverage(datasetTrain, normalization)
        val normalizedDataset = if (normalization == 1) datasetTrain.map(line => (line._1, line._2, line._3 - avgs(line._1 - 1), line._4)) else datasetTrain.map(line => (line._1, line._2, line._3 - avgs(line._2 - 1), line._4))
        // End
        
        val coMatrix = new CoordinateMatrix(datasetTrain.map {
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
        
        // Prepare Train Data
        var arrayDataTrain = normalizedDataset.map(line => Vectors.dense(line._1).toArray ++ Vectors.dense(line._2).toArray ++ Vectors.dense(line._3).toArray ++ getIndex(Ui, line._1).toArray ++ Vi(line._2).toArray)
        
        val rddTrain = arrayDataTrain.map(x => (x(0), x(1), x(2), org.apache.spark.ml .linalg.Vectors.dense(x.drop(1).drop(1).drop(1))))
        val train = rddTrain.toDF("user", "item", "label", "features")
        //train.show(5, false)
        
        // Prepare Test Data
        var arrayDataTest = datasetTest.map(line => Vectors.dense(line._1).toArray ++ Vectors.dense(line._2).toArray ++ Vectors.dense(line._3).toArray ++ getIndex(Ui, line._1).toArray ++ Vi(line._2).toArray)
        
        val rddTest = arrayDataTest.map(x => (x(0), x(1), x(2), org.apache.spark.ml .linalg.Vectors.dense(x.drop(1).drop(1).drop(1))))
        val test = rddTest.toDF("user", "item", "label", "features")
        
        val data = train.unionAll(test);
        // Apply Random Forest
        // Load and parse the data file, converting it to a DataFrame.
        //val data = train

        val numTrees = 50

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(data)

        // Split the data into training and test sets (30% held out for testing).
        //val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))

        // Train a RandomForest model.
        val rf = new RandomForestRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")
          .setNumTrees(numTrees)

        // Chain indexer and forest in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, rf))

        // Train model. This also runs the indexer.
        val model = pipeline.fit(train)

        // Make predictions.
        val predictions = model.transform(test)

        // Select example rows to display.
        //predictions.select("prediction", "label", "features").show(5)

        // DeNormalization
        //val rddPred = if (normalization == 1) predictions.select("user", "item", "label", "features", "indexedFeatures", "prediction").map(row => (row(0).asInstanceOf[String].toInt, row(1).asInstanceOf[String].toInt, row(2).asInstanceOf[String].toDouble, row(3), row(4), row(5).asInstanceOf[String].toDouble + avgs(row(0).asInstanceOf[String].toInt - 1))) else predictions.select("user", "item", "label", "features", "indexedFeatures", "prediction").map(row => (row(0).asInstanceOf[String].toInt, row(1).asInstanceOf[String].toInt, row(2).asInstanceOf[String].toDouble, row(3), row(4), row(5).asInstanceOf[String].toDouble + avgs(row(1).asInstanceOf[String].toInt - 1)))
        //val rddPred = if (normalization == 1) predictions.select("user", "item", "label", "prediction").map(row => (row(0).asInstanceOf[String].toInt, row(1).asInstanceOf[String].toInt, row(2).asInstanceOf[String].toDouble, row(5).asInstanceOf[String].toDouble + avgs(row(0).asInstanceOf[String].toInt - 1))) else predictions.select("user", "item", "label", "prediction").map(row => (row(0).asInstanceOf[String].toInt, row(1).asInstanceOf[String].toInt, row(2).asInstanceOf[String].toDouble, row(5).asInstanceOf[String].toDouble + avgs(row(1).asInstanceOf[String].toInt - 1)))
        /*
        val rddPred = if (normalization == 1) predictions.select("user", "item", "label", "prediction").map(row => (row(0).asInstanceOf[String], row(1).asInstanceOf[String], row(2).asInstanceOf[String].toDouble, row(5).asInstanceOf[String].toDouble + avgs(row(0).asInstanceOf[String].toInt - 1))) else predictions.select("user", "item", "label", "prediction").map(row => (row(0).asInstanceOf[String], row(1).asInstanceOf[String], row(2).asInstanceOf[String].toDouble, row(5).asInstanceOf[String].toDouble + avgs(row(1).asInstanceOf[String].toInt - 1)))
        val pred = rddPred.toDF("user", "item", "label", "prediction")
        */
        
        //val rddPred = if (normalization == 1) predictions.select("label", "prediction").map(row => (row(2).asInstanceOf[String].toDouble, row(5).asInstanceOf[String].toDouble + avgs(row(0).asInstanceOf[String].toInt - 1))) else predictions.select("user", "item", "label", "prediction").map(row => (row(2).asInstanceOf[String].toDouble, row(5).asInstanceOf[String].toDouble + avgs(row(1).asInstanceOf[String].toInt - 1)))
        
        val pred = predictions.withColumn("pred", predictions.select("user").rdd.map(x => avgs(x(0).asInstanceOf[Int])).toSeq)
        
        /*
        val schema = StructType(
            Array(
            StructField("user", IntegerType, true),
            StructField("item", IntegerType, true),
            StructField("label", DoubleType, true), 
            StructField("prediction", DoubleType, true) ))
        val rddPred = predictions.select("user", "item", "label", "prediction").map(row => Row(row(0).asInstanceOf[Int], row(1).asInstanceOf[Int], row(2).asInstanceOf[Double], row(3).asInstanceOf[Double] + avgs(row(0).asInstanceOf[Int] - 1)))
        val pred = sqlContext.createDataFrame(rddPred, schema)
        //val pred = rddPred.toDF("user", "item", "label", "prediction")
        */
        
        pred.show()
        
        
        
        
        /*
        val rdd4 = datasetTest.map(x => Row(x._1, x._2, x._3, x._4))
        val df = sqlContext.createDataFrame(rdd4, schema)
        df.withColumn("prediction", predictions("prediction")).show
        */
        //// End

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


