name := "SparkMLDemo"

version := "1.0"

scalaVersion := "2.11.2"

resolvers += "MMLSpark Repo" at "https://mmlspark.azureedge.net/maven"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.0",
  "org.apache.spark" %% "spark-sql" % "2.3.0",
  "org.apache.spark" %% "spark-mllib" % "2.3.0",
  "ml.dmlc" % "xgboost4j-spark" % "0.72",
  "com.microsoft.ml.spark" %% "mmlspark" % "0.14"
)