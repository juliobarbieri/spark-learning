import org.apache.spark.mllib.linalg.Matrix
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

object Test extends Serializable {
    def getIndex(array: Array[IndexedRow], index: Int) : Vector = {
        for (element <- array) {
            if (element.index == index) {
                return element.vector
            }
        }
        return Vectors.dense(0.0)
    }
    def matrixToRDD(m: Matrix): RDD[Vector] = {
        val columns = m.toArray.grouped(m.numRows)
        val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
        val vectors = rows.map(row => new DenseVector(row.toArray))
        sc.parallelize(vectors)
    }
    def main(args: Array[String]) {
        
        val filename = "/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/u.data"
        val f = sc.textFile(filename) 
        val dataset = f.map(_.split('\t') match {
                case Array(userId, itemId, rating, time) => (userId.toInt, itemId.toInt, rating.toDouble, time.toDouble)
            }
        )
        
        //for (line <- data) {
        //    println(line)
        //}
        
        val coMatrix = new CoordinateMatrix(dataset.map {
            case (userId, itemId, rating, time) => MatrixEntry(userId, itemId, rating)
        })
        
        val mat: IndexedRowMatrix = coMatrix.toIndexedRowMatrix
        
        // Apply SVD
        // Compute the top 20 singular values and corresponding singular vectors.
        val svd: SingularValueDecomposition[IndexedRowMatrix, Matrix] = mat.computeSVD(8, computeU = true)
        val U: IndexedRowMatrix = svd.U // The U factor is a RowMatrix.
        val s: Vector = svd.s // The singular values are stored in a local dense vector.
        val V: Matrix = svd.V // The V factor is a local dense matrix.
        
        val Ui = U.rows.collect//.map(x => x.toArray).collect
        val Vi = Array() ++ V.rowIter
        
        //val rows = matrixToRDD(V)
        //val Vi = new RowMatrix(rows)
        //println(Ui(30))
        //val Ud = DenseMatrix(Ui.map(_.toArray):_*)
        
        //val Ud = new BDM(U.numRows().toInt, U.numCols().toInt, Ui)
        //for (line <- U.rows) {
        //    println(line)
        //}
        
        var arrayData = Array[Array[Any]]()
        for (line <- dataset) {
            val userFeatures = getIndex(Ui, line._1)
            val itemFeatures = Vi(line._2)
            //val size = userFeatures.size + itemFeatures.size
            //val maxIndex = userFeatures.size
            //val indices = userFeatures.indices ++ itemFeatures.indices.map(e => e + maxIndex)
            //val values = userFeatures ++ itemFeatures
            val features = Vectors.dense(line._3).toArray ++ userFeatures.toArray ++ itemFeatures.toArray
            
            arrayData ++= features
            //println(features.mkString(" "))
            //println(features)
            //println(line._1) //User
            //println(line._2) //Item
            //println(line._3) //Rating
        }
        
        //println(arrayData) 
        //val lol :Nothing = arrayData
        
        val schemaString = "y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16"

        // Generate the schema based on the string of schema
        val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = true))
        val schema = StructType(fields)
        
        val rdd = sc.parallelize(arrayData).map(attributes => Row(attributes))
        val df = spark.createDataFrame(rdd, schema)
        //val df = rdd.toDF("y","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16")
        df.show
        //val rdd = sc.parallelize(arrayData).toDF() //.map(Row.fromSeq(_))
        //val data = sqlContext.createDataFrame(rdd, schema)
        
        //println(rdd.show)
        //println(U)
        //println(V)
        
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


