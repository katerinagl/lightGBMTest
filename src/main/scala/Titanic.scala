import com.microsoft.ml.spark.{LightGBMClassificationModel, LightGBMClassifier, LightGBMRegressionModel, LightGBMRegressor}
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, PredictionModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics}
import org.apache.spark.sql.functions.{col, mean}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by katerinaglushchenko on 8/7/18.
  */
object Titanic {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[*]")
    implicit val sparkSession = SparkSession.builder.config(sparkConf).getOrCreate()

    val df = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/train.csv")

    val meanValue = df.agg(mean(df("Age"))).first.getDouble(0)
    val fixedDf = df.na.fill(meanValue, Array("Age"))

    val Array(trainDf, testDf) = fixedDf.randomSplit(Array(0.7, 0.3)).map(_.withColumnRenamed("Survived", "label"))

    val categoricalFeaturesStages = handleCategorical(Array("Sex", "Embarked", "Pclass"))

    val cols = Array("Sex_onehot", "Embarked_onehot", "Pclass_onehot", "SibSp", "Parch", "Age", "Fare")
    val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")

    val preProcessStages = categoricalFeaturesStages ++ Array(vectorAssembler)

    val trainData = new Pipeline().setStages(preProcessStages).fit(trainDf).transform(trainDf)
    val testData = new Pipeline().setStages(preProcessStages).fit(testDf).transform(trainDf)

//    val randomForestModel = new RandomForestClassifier().setFeaturesCol("features").fit(trainData)
//
//    val xgbModel = XGBoost.trainWithDataFrame(trainData, Map("objective" -> "binary:logistic"), 100, 4)
//    xgbModel.save("")

    val lgbmModel = new LightGBMRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setNumIterations(100)
      .fit(trainData)

    ////////

//    val allResults = randomForestModel.transform(trainData).union(randomForestModel.transform(testData))
//    allResults.createOrReplaceTempView("result")
//
//    val q = sparkSession.sql("select Name, prediction from result where Name = 'Abbott, Mrs. Stanton (Rosa Hunt)'")
//    q.show(false)
//
//    allResults.select("Name", "prediction").where(col("Name") === "Abbott, Mrs. Stanton (Rosa Hunt)").show(false)


//    evaluateResult[RandomForestClassificationModel]("random forest ", randomForestModel, trainData, testData)
//    evaluateResult[XGBoostModel]("xgboost ", xgbModel, trainData, testData)
    evaluateResult[LightGBMRegressionModel]("lightGbm", lgbmModel, trainData, testData)
  }

  def showResults(df: DataFrame) = {
    df.select("Name", "prediction").filter(col("Name") === "Abbott, Mrs. Stanton (Rosa Hunt)").show(false)
  }

  def evaluateResult[M <: PredictionModel[_, M]](label: String, model: M, train: DataFrame, test: DataFrame) = {
    println(s"$label train accuracy with pipeline " + accuracyScore(model.transform(train)))
    println(s"$label test accuracy with pipeline " + accuracyScore(model.transform(test)))
  }

  def handleCategorical(columns: Array[String]): Array[PipelineStage] = {
    val stringIndexer = columns.map { column =>
      new StringIndexer().setInputCol(column)
        .setOutputCol(s"${column}_index")
        .setHandleInvalid("skip")
    }

    val oneHot = new OneHotEncoderEstimator()
      .setInputCols(columns.map(column => s"${column}_index")).setOutputCols(columns.map(column => s"${column}_onehot"))
    stringIndexer :+ oneHot
  }

  def accuracyScore(df: DataFrame) = {
    val dfR = df.select("label", "prediction")
    dfR.show()
//    val rdd = dfR.rdd.map(row ⇒ (row.getInt(0).toDouble, row.getFloat(1).toDouble))
    val rdd = dfR.rdd.map(row ⇒ (row.getInt(0).toDouble, row.getDouble(1)))
    new RegressionMetrics(rdd).rootMeanSquaredError
//    new MulticlassMetrics(rdd).accuracy
  }
}