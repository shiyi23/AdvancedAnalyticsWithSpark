
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession._


val spark = SparkSession
  .builder()
  .appName("AdvancedAnalyticsWithSpark")
  .master("")
  .getOrCreate()
import spark.implicits._

val dataWithoutHeader = spark.read
  .option("inferSchema", true)
  .option("header", false) //设置表格的第一行不解析为标题行
  .csv("/FileStore/tables/covtype.data")

//在处理 DataFrame 之前，将列名添加到其中,使用++操作符连接两个集合
val colNames = Seq(
  "Elevation", "Aspect", "Slope",
  "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
  "Horizontal_Distance_To_Roadways",
  "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
  "Horizontal_Distance_To_Fire_Points"
) ++ (
  (0 until 4).map(i => s"Wilderness_Area_$i")
  ) ++ (
  (0 until 40).map(i => s"Soil_Type_$i")
  ) ++ Seq("Cover_Type")

val data = dataWithoutHeader.toDF(colNames: _*)
  .withColumn("Cover_Type", $"Cover_Type".cast("double"))

val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1)) //在这里对数据集做分类

trainData.cache() //及时缓存
testData.cache()

  import org.apache.spark.ml.feature.VectorAssembler //用于将要输入Spark的ml的分类器的、包含
  // 许多列的DataFrame合并成一列，即一个向量，且这个向量是双精度的浮点数

val inputCols = trainData.columns.filter(_ != "Cover_Type")

val assembler = new VectorAssembler()
  .setInputCols(inputCols)
  .setOutputCol("featureVector")

val assembledTrainData = assembler.transform(trainData)

assembledTrainData.select("featureVector").show(truncate = false)

import org.apache.spark.ml.classification.DecisionTreeClassifier
import scala.util.Random

val classifier = new DecisionTreeClassifier() //分类器的主要配置
  .setSeed(Random.nextLong()) //使用随机种子
  .setLabelCol("Cover_Type")
  .setFeaturesCol("featureVector")
  .setPredictionCol("prediction")

val model = classifier.fit(assembledTrainData)

//println(model.toDebugString)

//在构造决策树的过程中，决策树能够评估输入特征的重要性，
// 也就是说它们可以评估每个输入特征对做出正确的预测的贡献值

model.featureImportances.toArray.zip(inputCols).sorted.reverse.foreach(println)

val predictions = model.transform(assembledTrainData)

//predictions.select("Cover_type", "prediction", "probability")
//  .show(truncate = false)

/**
 * MulticlassClassificationEvaluator 能计算准确率和其他评估模型预测质量的指标。 Spark
 * MLlib 中评估器的作用就是以某种方式评估输出 DataFrame 的质量， MulticlassClassification
 * Evaluator 就是评估器的一个例子。
 */

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("Cover_Type")
  .setPredictionCol("prediction")

evaluator.setMetricName("accuracy").evaluate(predictions)
evaluator.setMetricName("f1").evaluate(predictions)



































