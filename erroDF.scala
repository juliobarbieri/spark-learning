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
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

def main(args: Array[String]) {
    //val conf = new SparkConf().setMaster("local").setAppName("test")
    //val sc = new SparkContext(conf)
    //require spark sql environment
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val df1 = sc.makeRDD(1 to 5).map(i => (i, i * 2)).toDF("single", "double")
    sc.stop()
}
