import Titanic.{accuracyScore, evaluateResult, handleCategorical}
import com.microsoft.ml.spark.{LightGBMClassificationModel, LightGBMClassifier, LightGBMRegressionModel, LightGBMRegressor}
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineStage, PredictionModel}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object LightGbmClassifierDemo {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[2]")
    implicit val sparkSession = SparkSession.builder.config(sparkConf).getOrCreate()

    //    val df = sparkSession.read
    //      .option("header", "true")
    //      .option("inferSchema", "true")
    //      //      .csv("src/main/resources/air_small.csv")
    //      .csv("src/main/resources/2008_14col.csv")
    //
    //    df.show()
    //
    //    val Array(trainDf, testDf) = df
    //      .withColumnRenamed("ArrDelay", "label")
    //      //      .withColumn("label", when(col("ArrDelay") < 0, 0).otherwise(1))
    //      .randomSplit(Array(0.001, 0.001))
    //
    //
    //    val categoricalFeaturesStages = handleCategorical(Array("UniqueCarrier", "Origin", "Dest"))
    //
    //    //    val cols = Array("Year","Month","DayofMonth","DayofWeek","CRSDepTime","CRSArrTime","FlightNum","ActualElapsedTime","Distance","Diverted")
    //    val cols = Array("Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime", "CRSArrTime", "UniqueCarrier_index",
    //      "FlightNum", "ActualElapsedTime", "Origin_index", "Dest_index", "Distance", "Diverted")
    //    val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    //
    //    val preProcessStages = categoricalFeaturesStages ++ Array(vectorAssembler)
    //
    //    val trainData = new Pipeline().setStages(preProcessStages).fit(trainDf).transform(trainDf)
    //    val testData = new Pipeline().setStages(preProcessStages).fit(testDf).transform(trainDf)
    //    trainData.write.parquet("src/main/resources/2008_14col_train")

    //    val trainData = sparkSession.read.parquet("/Users/katerinaglushchenko/Downloads/HIGGS_train")//.coalesce(1)
    val trainData = sparkSession.read.parquet("src/main/resources/2008_14col_train") //.coalesce(1)
    println("TOTAL: " + trainData.count())
    println("PARTITIONS : " + trainData.rdd.partitions.length)

    val lgbmModel = new LightGBMRegressor()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setApplication("regression_l2")
      .fit(trainData)

    evaluateResult[LightGBMRegressionModel]("lightGbm", lgbmModel, trainData)
  }


  def evaluateResult[M <: PredictionModel[_, M]](label: String, model: M, train: DataFrame, test: Option[DataFrame] = None) = {
    println(s"$label train accuracy with pipeline " + accuracyScore(model.transform(train)))
    test.foreach(t => println(s"$label test accuуracy with pipeline " + accuracyScore(model.transform(t))))
  }

  def handleCategorical(columns: Array[String]): Array[StringIndexer] = {
    val stringIndexer = columns.map { column =>
      new StringIndexer().setInputCol(column)
        .setOutputCol(s"${column}_index")
        .setHandleInvalid("skip")
    }

    //    val oneHot = new OneHotEncoderEstimator()
    //      .setInputCols(columns.map(column => s"${column}_index")).setOutputCols(columns.map(column => s"${column}_onehot"))
    stringIndexer //:+ oneHot
  }
}
