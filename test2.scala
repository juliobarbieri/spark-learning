import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.VectorUDT
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

//val sqlContext = new org.apache.spark.sql.SQLContext(sc)
//import sqlContext.implicits._

object Test extends Serializable {
    def getIndex(array: Array[IndexedRow], index: Int) : Vector = {
        for (element <- array) {
            if (element.index == index) {
                return element.vector
            }
        }
        return Vectors.dense(0.0)
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
        
        /*
        var arrayData = Array[Double]()
        dataset.foreach(line => arrayData ++= Vectors.dense(line._3).toArray ++ getIndex(Ui, line._1).toArray ++ Vi(line._2).toArray)
        println(arrayData.length)
        */
        
        
        
        var arrayData = dataset.map(line => Vectors.dense(line._3).toArray ++ getIndex(Ui, line._1).toArray ++ Vi(line._2).toArray)//.collect
        //udf(arrayData.map(x => Vectors.dense(x)), VectorUDT())
        //println(arrayData.length)
        //val vectorData = Vectors.dense(arrayData.length, arrayData)
        
        //println(arrayData) 
        //val lol :Nothing = arrayData
        
        /*
        val schemaString = "label features"

        // Generate the schema based on the string of schema
        val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, DoubleType, nullable = false))
        val schema = StructType(fields)
        */
        
        val schema = StructType(
            Array(
            StructField("label", DoubleType, true), 
            StructField("features", VectorType, true) )
        )
        
        //Vectors.dense(length(x.drop(1)), x.drop(1)))
        //val rdd = sc.parallelize(arrayData).map(attributes => Row(attributes))
        //val rdd = sc.parallelize(arrayData).map(x => (x(0), Row.fromSeq(x.drop(1))))
        val rdd = arrayData.map(x => Row.fromSeq(Seq(x(0), Vectors.dense(x.drop(1)))))
        val df = spark.createDataFrame(rdd, schema) //rdd.toDF("label, features")rdd.toDF("label, features")
        df.show
        
        
        //println(rdd.show)
        //println(U)
        //println(V)
        
        
        /*
        // Apply Random Forest
        // Load and parse the data file, converting it to a DataFrame.
        val data = df //spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(data)

        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

        // Train a RandomForest model.
        val rf = new RandomForestRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")

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

        val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
        println("Learned regression forest model:\n" + rfModel.toDebugString)
        */
        //System.exit(0)
    }
}


