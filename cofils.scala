import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.ml.PipelineModel
import scala.io.Source._
import org.apache.spark.sql.SparkSession

import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, IndexedRow}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.LinearRegression

import org.apache.spark.sql._
import org.apache.spark.sql.types._
import scala.collection.Map

case class UntrainedDataset(user:Double, item:Double, label:Double, features:org.apache.spark.ml.linalg.Vector)
case class TrainedDataset(user:Double, item:Double, label:Double, prediction:Double)
case class customException(smth:String)  extends Exception
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

def createDataFrame(dataset :RDD[(Int, Int, Double, Double)]) : DataFrame = {
    val schema = StructType(
        Array(
        StructField("user", IntegerType, true),
        StructField("item", IntegerType, true),
        StructField("rating", DoubleType, true), 
        StructField("time", DoubleType, true) ))
    val rdd = dataset.map(x => Row(x._1, x._2, x._3, x._4))
    val df = sqlContext.createDataFrame(rdd, schema)
    return df
}

def getIndex(array: Array[IndexedRow], index: Int) : org.apache.spark.mllib.linalg.Vector = {
    return array.filter(x => x.index == index)(0).vector
}

def remapIds(dataset :RDD[(Int, Int, Double, Double)], normalization :Int) : Map[Int,Int] = {
    val df = createDataFrame(dataset)
    
    if (normalization == 1) {
        val userDF = df.select("user").orderBy("user").distinct
        val schema = userDF.schema
        val rows = userDF.rdd.zipWithIndex.map{
            case (r: Row, id: Long) => Row.fromSeq(id +: r.toSeq)}
        val dfWithPK = sqlContext.createDataFrame(
            rows, StructType(StructField("id", LongType, false) +: schema.fields))
        var dicIds: Map[Int,Int] = dfWithPK.rdd.map(x => (x(1).asInstanceOf[Int] -> x(0).asInstanceOf[Long].toInt)).collectAsMap
        return dicIds
    }
    else {
        val itemDF = df.select("item").orderBy("item").distinct
        val schema = itemDF.schema
        val rows = itemDF.rdd.zipWithIndex.map{
            case (r: Row, id: Long) => Row.fromSeq(id +: r.toSeq)}
        val dfWithPK = sqlContext.createDataFrame(
            rows, StructType(StructField("id", LongType, false) +: schema.fields))
        var dicIds: Map[Int,Int] = dfWithPK.rdd.map(x => (x(1).asInstanceOf[Int] -> x(0).asInstanceOf[Long].toInt)).collectAsMap
        return dicIds
    }
}

def calculateAverage(dataset :RDD[(Int, Int, Double, Double)], normalization :Int) : Array[Double] = {
    val df = createDataFrame(dataset)
    
    if (normalization == 1) {
        val userDF = df.select("user", "rating").orderBy("user").groupBy("user").agg($"user", avg("rating"))
        return userDF.select("avg(rating)").rdd.map(r => r(0).asInstanceOf[Double]).collect()
    }
    else {
        val itemDF = df.select("item", "rating").orderBy("item").groupBy("item").agg($"item", avg("rating"))
        return itemDF.select("avg(rating)").rdd.map(r => r(0).asInstanceOf[Double]).collect()
    }

}

def prepareData(U: IndexedRowMatrix, V: Matrix, dataset :RDD[(Int, Int, Double, Double)]) : DataFrame = {
    val Ui = U.rows.collect
    val Vi = Array() ++ V.rowIter
    
    // Prepare Train Data
    var arrayData = dataset.map(line => Vectors.dense(line._1).toArray ++ Vectors.dense(line._2).toArray ++ Vectors.dense(line._3).toArray ++ getIndex(Ui, line._1).toArray ++ Vi(line._2).toArray)
    
    val rdd = arrayData.map(x => UntrainedDataset(x(0), x(1), x(2), org.apache.spark.ml.linalg.Vectors.dense(x.drop(1).drop(1).drop(1))))
    val data = rdd.toDF("user", "item", "label", "features")
    return data
}

def fitModel(modelType: String, train: DataFrame, parameters: Array[Double]) : PipelineModel = {
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(train)
    
    if (modelType != "RandomForest" && modelType != "GBTrees" && modelType != "DecisionTree" && modelType != "LinearRegression") {
        throw new IllegalArgumentException
    }
    
    if (modelType == "RandomForest" && parameters.size != 2) {
        throw new IllegalArgumentException
    }
    
    if (modelType == "GBTrees" && parameters.size != 2) {
        throw new IllegalArgumentException
    }
    
    if (modelType == "DecisionTree" && parameters.size != 1) {
        throw new IllegalArgumentException
    }
    
    if (modelType == "LinearRegression" && parameters.size != 3) {
        throw new IllegalArgumentException
    }
    
    val sl = if (modelType == "RandomForest") new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setNumTrees(parameters(0).asInstanceOf[Int]).setMaxDepth(parameters(1).asInstanceOf[Int]) else if (modelType == "GBTrees") new GBTRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxIter(parameters(0).asInstanceOf[Int]).setMaxDepth(parameters(1).asInstanceOf[Int]) else if (modelType == "DecisionTree") new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxDepth(parameters(0).asInstanceOf[Int]) else new LinearRegression().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxIter(parameters(0).asInstanceOf[Int]).setRegParam(parameters(1)).setElasticNetParam(parameters(2))
    
    val pipeline = new Pipeline().setStages(Array(featureIndexer, sl))
    val model = pipeline.fit(train)
    return model
}

// Parameters
val slParameters :Array[Double] = Array(50, 20)

val factors         = 8
val normalization   = 2 
val modelType       = "RandomForest"

val trainFilename = "/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/folds/u1.base"
val testFilename = "/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/folds/u1.test"

// Load Train files
val fTrain = sc.textFile(trainFilename)
val datasetTrain = fTrain.map(_.split('\t') match {
        case Array(userId, itemId, rating, time) => (userId.toInt, itemId.toInt, rating.toDouble, time.toDouble)
    }
)

// Normalization
var dicIds: Map[Int,Int] = remapIds(datasetTrain, normalization)
val avgs = calculateAverage(datasetTrain, normalization)

val normalizedDataset = if (normalization == 1) datasetTrain.map(line => (dicIds(line._1) + 1, line._2, line._3 - avgs(dicIds(line._1)), line._4)) else datasetTrain.map(line => (line._1, dicIds(line._2) + 1, line._3 - avgs(dicIds(line._2)), line._4))

// Load Test files
val fTest = sc.textFile(testFilename)
val datasetTest = fTest.map(_.split('\t') match {
        case Array(userId, itemId, rating, time) => (userId.toInt, itemId.toInt, rating.toDouble, time.toDouble)
    }
)

// Remove Cold Start
val preFilteredDatasetTest = if (normalization == 1) datasetTest.filter(line => dicIds.keys.toList.contains(line._1)) else datasetTest.filter(line => dicIds.keys.toList.contains(line._2))
val filteredDatasetTest = if (normalization == 1) preFilteredDatasetTest.map(line => (dicIds(line._1) + 1, line._2, line._3, line._4)) else preFilteredDatasetTest.map(line => (line._1, dicIds(line._2) + 1, line._3, line._4))

// Prepare Matrix
val coMatrix = if (normalization == 1) new CoordinateMatrix(datasetTrain.map {
    case (userId, itemId, rating, time) => MatrixEntry(dicIds(userId) + 1, itemId, rating)
}) else new CoordinateMatrix(datasetTrain.map {
    case (userId, itemId, rating, time) => MatrixEntry(userId, dicIds(itemId) + 1, rating)
})

val mat: IndexedRowMatrix = coMatrix.toIndexedRowMatrix

// Apply Latent Variable Extraction Technique
val svd: SingularValueDecomposition[IndexedRowMatrix, Matrix] = mat.computeSVD(factors, computeU = true)
val U: IndexedRowMatrix = svd.U
val V: Matrix = svd.V

// Prepare Train Data
val train = prepareData(U, V, normalizedDataset)

// Prepare Test Data
val test = prepareData(U, V, filteredDatasetTest)

// Train a Supervised Learning model.
val model = fitModel(modelType, train, slParameters)
var predictions = model.transform(test)

// DeNormalization
predictions = if (normalization == 1) predictions.rdd.map(x => TrainedDataset(x(0).asInstanceOf[Double], x(1).asInstanceOf[Double], x(2).asInstanceOf[Double], x(5).asInstanceOf[Double] + avgs(x(0).asInstanceOf[Double].toInt - 1))).toDF("user", "item", "label", "prediction") else predictions.rdd.map(x => TrainedDataset(x(0).asInstanceOf[Double], x(1).asInstanceOf[Double], x(2).asInstanceOf[Double], x(5).asInstanceOf[Double] + avgs(x(1).asInstanceOf[Double].toInt - 1))).toDF("user", "item", "label", "prediction")

//predictions = predictions.withColumn("prediction", when(col("prediction").lt(1.0), 1.0)).withColumn("prediction", when(col("prediction").gt(5.0), 5.0))

//// End

// Select (prediction, true label) and compute test error.
val evaluatorRmse = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluatorRmse.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val evaluatorMae = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")
val mae = evaluatorMae.evaluate(predictions)
println("Mean Absolute Error (MAE) on test data = " + mae)

System.exit(0)

