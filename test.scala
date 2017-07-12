import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import scala.io.Source._
import org.apache.spark.sql.SparkSession

import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

object Test {
    def main(args: Array[String]) {
        def matrixToRDD(m: Matrix): RDD[Vector] = {
            val columns = m.toArray.grouped(m.numRows)
            val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
            val vectors = rows.map(row => new DenseVector(row.toArray))
            sc.parallelize(vectors)
        }
        
        val filename = "/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/u.data"
        val lines = fromFile(filename).getLines
        
        var dataset: Array[Array[Int]] = fromFile(filename).getLines.map(_.split("\t").map(_.trim.toInt)).toArray
        
        var matrix = Array.ofDim[Float](943,1682)
        
        //def getCol(n: Int, x: Array[Array[Int]]) = x.map{_(n - 1)}
        //println(getCol(1, dataset))
        
        //val user: Int = (dataset.maxBy(tokens => tokens(0).toInt))(0)
        //val item: Int = (dataset.maxBy(tokens => tokens(1).toInt))(1)
        
        for (line <- lines) {
            val rating = line.split("\t").map(_.trim.toInt).toArray
            matrix(rating(0) - 1)(rating(1) - 1) = rating(2)
            //println(line.split("\t").map(_.trim.toInt).toArray)
        }
        
        val rows = matrixToRDD(matrix)
        val mat = new RowMatrix(rows)
        //val mat: RowMatrix = matrix

        // Apply SVD
        // Compute the top 20 singular values and corresponding singular vectors.
        //val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(20, computeU = true)
        //val U: RowMatrix = svd.U // The U factor is a RowMatrix.
        //val s: Vector = svd.s // The singular values are stored in a local dense vector.
        //val V: Matrix = svd.V // The V factor is a local dense matrix.
        
        // Apply Random Forest
        // Load and parse the data file, converting it to a DataFrame.
        //val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        //val featureIndexer = new VectorIndexer()
        //  .setInputCol("features")
        //  .setOutputCol("indexedFeatures")
        //  .setMaxCategories(4)
        //  .fit(data)

        // Split the data into training and test sets (30% held out for testing).
        //val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

        // Train a RandomForest model.
        //val rf = new RandomForestRegressor9()
        //  .setLabelCol("label")
        //  .setFeaturesCol("indexedFeatures")

        // Chain indexer and forest in a Pipeline.
        //val pipeline = new Pipeline()
        //  .setStages(Array(featureIndexer, rf))

        // Train model. This also runs the indexer.
        //val model = pipeline.fit(trainingData)

        // Make predictions.
        //val predictions = model.transform(testData)

        // Select example rows to display.
        //predictions.select("prediction", "label", "features").show(5)

        // Select (prediction, true label) and compute test error.
        //val evaluator = new RegressionEvaluator()
        //  .setLabelCol("label")
        //  .setPredictionCol("prediction")
        //  .setMetricName("rmse")
        //val rmse = evaluator.evaluate(predictions)
        //println("Root Mean Squared Error (RMSE) on test data = " + rmse)

        //val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
        //println("Learned regression forest model:\n" + rfModel.toDebugString)
        
        //kSystem.exit(0)
    }
}


