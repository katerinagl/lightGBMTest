import com.microsoft.ml.spark.{LightGBMRegressionModel, LightGBMRegressor}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PredictionModel}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, when}

object LightGbmClassifierDemo {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[2]")
    implicit val sparkSession = SparkSession.builder.config(sparkConf).getOrCreate()

    val trainData = sparkSession.read.parquet("src/main/resources/2008_14col_train")
    println("TOTAL: " + trainData.count())
    println("PARTITIONS : " + trainData.rdd.partitions.length)

    val lgbmModel = new LightGBMRegressor()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
//      .setApplication("regression_l2")
      .fit(trainData)

    evaluateResult[LightGBMRegressionModel]("lightGbm", lgbmModel, trainData)
  }


  def prepareData(implicit sparkSession: SparkSession) = {
    val df = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/2008_14col.csv")

    df.show()

    val Array(trainDf, testDf) = df
      .withColumn("label", when(col("ArrDelay") < 0, 0).otherwise(1))
      .randomSplit(Array(0.001, 0.001))


    val categoricalFeaturesStages = handleCategorical(Array("UniqueCarrier", "Origin", "Dest"))

    val cols = Array("Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime", "CRSArrTime", "UniqueCarrier_index",
      "FlightNum", "ActualElapsedTime", "Origin_index", "Dest_index", "Distance", "Diverted")
    val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")

    val preProcessStages = categoricalFeaturesStages ++ Array(vectorAssembler)

    val trainData = new Pipeline().setStages(preProcessStages).fit(trainDf).transform(trainDf)
    val testData = new Pipeline().setStages(preProcessStages).fit(testDf).transform(trainDf)
    trainData.write.parquet("src/main/resources/2008_14col_train")
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

    stringIndexer
  }

  def accuracyScore(df: DataFrame) = {
    val dfR = df.select("label", "prediction")
    dfR.show()
    val rdd = dfR.rdd.map(row ⇒ (row.getInt(0).toDouble, row.getDouble(1)))
    new RegressionMetrics(rdd).rootMeanSquaredError
  }
}
