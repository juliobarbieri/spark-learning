import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
def parseRating(str: String): Rating = {
  val fields = str.split("\t")
  assert(fields.size == 4)
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
}

val nFolds   = 1

val maxIter  = 14
val factors  = 7
val reg      = 0.1

var MAEs     = Array.fill[Double](nFolds)(0)
var RMSEs    = Array.fill[Double](nFolds)(0)

var avgMAE   = 0.0
var avgRMSE  = 0.0

var stdMAE   = 0.0
var stdRMSE  = 0.0

for (i <- 1 to nFolds) {
    //val currentDirectory = new java.io.File(".").getCanonicalPath
    //println(currentDirectory)

    val training = spark.read.textFile(s"/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/folds/u$i.base").map(parseRating).toDF()
    val test = spark.read.textFile(s"/home/juliobarbieri/Desenvolvimento/spark_codes/data/ml-100k/folds/u$i.test").map(parseRating).toDF()

    // Build the recommendation model using ALS on the training data
    val als = new ALS().setMaxIter(maxIter).setRank(factors).setRegParam(reg).setUserCol("userId").setItemCol("movieId").setRatingCol("rating").setColdStartStrategy("drop")
    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    val predictions = model.transform(test)

    val evaluatorRMSE = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
    val rmse = evaluatorRMSE.evaluate(predictions)
    println(s"Fold $i - Root-mean-square error = $rmse")

    val evaluatorMAE = new RegressionEvaluator().setMetricName("mae").setLabelCol("rating").setPredictionCol("prediction")
    val mae = evaluatorMAE.evaluate(predictions)
    println(s"Fold $i - Mean absolute error = $mae")

    //avgMAE  += mae
    //avgRMSE += rmse
    
    MAEs(i-1)  = mae
    RMSEs(i-1) = rmse
}

//avgMAE  /= nFolds
avgMAE = MAEs.sum / nFolds
//avgRMSE /= nFolds
avgRMSE = RMSEs.sum / nFolds

val devsMAE = MAEs.map(mae => (mae - avgMAE) * (mae - avgMAE))
stdMAE = Math.sqrt(devsMAE.sum / (nFolds - 1))

val devsRMSE = RMSEs.map(rmse => (rmse - avgRMSE) * (rmse - avgRMSE))
stdRMSE = Math.sqrt(devsRMSE.sum / (nFolds - 1))

println(s"Average Root-mean-square error = $avgRMSE")
println(s"Average Mean absolute error = $avgMAE")

println(s"Std Root-mean-square error = $stdRMSE")
println(s"Std Mean absolute error = $stdMAE")


System.exit(0)
