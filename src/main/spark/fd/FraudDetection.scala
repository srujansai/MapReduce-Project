package fd

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.log4j.Level
import org.apache.log4j.LogManager
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Bucketizer, Normalizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel

import scala.util.Try

object FraudDetectionMain {
  
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val conf = new SparkConf().setAppName("Fraud Detection")
    val sc = new SparkContext(conf)
    val spark = SparkSession
      .builder()
      .appName("Fraud Detection")
      .getOrCreate()
      
    import spark.implicits._

    /* read input file and convert values to double using createLabeledPoints and store as a DF, 
     * using case class a custom schema is inferred */
    var inputDF = createLabeledPoints(sc.textFile(args(0)))
                              .map(inline => Transaction(inline(0), inline(1), inline(2), inline(3),
                                                         inline(4), inline(5), inline(6), inline(7),
                                                         inline(8), inline(9), inline(10))).toDF().cache()

                                                            
    
     /* Generating a feature vector, which contains all features/columns used to predict the
     *  isFraud (label) column */
                                                           
    val featureNames = Array("step","transType","amt","nameSender",
                                "preBalSen","newBalSen","nameReciever",
                                "preBalRecv","newBalRecv","isFlagged")

    val featureVector = new VectorAssembler().setInputCols(featureNames).setOutputCol("features")
    
    //A data frame created containing feature tag using the above feature vector format
    
    val modfDF = featureVector.transform(inputDF)

    // To be predicted column  isFraud is set as the label    
    val labelCol = new StringIndexer().setInputCol("isFraud").setOutputCol("label")
    
    // A new DF which contains both features & label tags
    val labeledDF = labelCol.fit(modfDF).transform(modfDF)
    
    
    
    

    /* Split the data into train and test data in a 70:30 ratio, both test and train data  
    	have fraud as well as non fraud records, first split based on isFraud and then 
    	convert to train test ratio mentioned above. */
    
    
    val nonFraudRecords = labeledDF.filter(x => x.getDouble(9) == 0.0).randomSplit(Array(0.7, 0.3), seed = 11L)
    val fraudRecords = labeledDF.filter(x => x.getDouble(9) == 1.0).randomSplit(Array(0.7, 0.3), seed = 11L)
    
    //Combine both fraud and non-fraud records to get training and testing data
    var training = nonFraudRecords(0).union(fraudRecords(0)).toDF()
    var testing = nonFraudRecords(1).union(fraudRecords(1)).toDF()
    
    
    
    /* Random Forest Classifier
  
     A Random forest classifier, set with appropriate parameters for training */
    
    
    val randomFrClassifier = new RandomForestClassifier()
                              .setImpurity("gini")
                              .setMaxDepth(10)
                              .setNumTrees(250)
                              .setFeatureSubsetStrategy("auto")
                              .setSeed(9000)

    // train the model
    val randomFrModel = randomFrClassifier.fit(training)
    

    /* Get predictions using the test data 
    		find accuracy of predictions using the binary classification evaluator */
    val fraudPrediction = randomFrModel.transform(testing)
    val binaryEvaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val randomFrAccuracy = binaryEvaluator.evaluate(fraudPrediction)
    
    logger.info("Random Forest accuracy : "+ randomFrAccuracy)

    
     
     
     /* Converting the training and testing data into an RDD of LabeledPoints
     */
    val trainLabeled = training.rdd.map(row =>
      LabeledPoint(row.getAs("label"),
        org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs("features"))))
    val testLabeled = testing.rdd.map(row =>
      LabeledPoint(row.getAs("label"),
        org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs("features"))))
    
    
    
    /* Decision Tree 

     * A Decision Tree  classifier, set with appropriate parameters for training a binary classifier  */  
        
    val numClasses = 2 
    val categoricalFeatures = Map[Int,Int]((1,5),(3,2),(6,2))
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 15
    
    
    
    // train the model
    val decisionTrModel = DecisionTree.trainClassifier(trainLabeled, numClasses, categoricalFeatures, impurity, maxDepth, maxBins)
    
    
    /* Get predictions using the test data 
    		 using the binary classification evaluator */
    var dtFraudPrediction = testLabeled.map{ testRecord => val prediction = decisionTrModel.predict(testRecord.features)
                                                                   (testRecord.label, prediction)}
    
    
    
    /* find accuracy of predictions by comparing the predicted label with the actual label within 
     * testing data and count matches and then divide by total testing data rows  */
    
    val decisionTrAccuracy = dtFraudPrediction.filter(row => row._1 == row._2).count().toDouble / testing.count()
    logger.info("Decision Tree Accuracy  : "+ decisionTrAccuracy);
    
    // Save the accuracy of the models to an RDD and an output file
    val accuracyRDD = sc.parallelize(List(Row("Decision Tree Accuracy",decisionTrAccuracy),
                                          Row("Random Forest Accuracy",randomFrAccuracy)))
                                               

                                               
    accuracyRDD.saveAsTextFile(args(1))
    
  }
  
  /* Helper methods and a implicit conversion class  */
  
  // A schema for each input record/row to be stored in inputDF 
  case class Transaction(
    step: Double,
    transType: Double,
    amt: Double,
    nameSender: Double,
    preBalSen: Double,
    newBalSen: Double,
    nameReciever: Double,
    preBalRecv: Double,
    newBalRecv: Double,
    isFraud: Double,
    isFlagged: Double)
    
    
    
    
  /* Convert the RDD of Strings to RDD of Double (to create labeled points as input to models)  */
  
  def createLabeledPoints(inRDD : RDD[String]): RDD[Array[Double]] = {

    val convDoubleRDD = inRDD.map(_.split(","))
                              .map{case Array(step,transType,amt,nameSender,preBalSen,newBalSen,nameReciever, preBalRecv,newBalRecv,isFraud,isFlagged)
                                                    => var modTranstype = transTypeConversion(transType)
                                                       var modNameSender = binaryConversion(nameSender)
                                                       var modNameReciever = binaryConversion(nameReciever)
                                                       Array(step.toDouble, modTranstype, amt.toDouble, modNameSender,
                                                             preBalSen.toDouble, newBalSen.toDouble, modNameReciever, preBalRecv.toDouble,
                                                             newBalRecv.toDouble, isFraud.toDouble, isFlagged.toDouble)}

  return convDoubleRDD;
}

  /* Convert nameSender and nameReciever to binary double value, each of those start with either 'C' or 'M'
    If 'C' it is convert to 1.0 else if 'M' it is 0.0 */
  def binaryConversion(nameConv : String): Double = {
    var binaryVal:Double = 0.0
    
    if (nameConv(0).equals('C')){
        binaryVal = 1.0
    }
    
    return binaryVal;

}
  
  /* Convert transType to double values, "CASH_IN" => 0.0, "CASH_OUT" => 1.0, "DEBIT" => 2.0,
    "PAYMENT" => 3.0 and "TRANSFER" => 4.0 */
 
  def transTypeConversion(transactionType : String): Double = {
    var transactionVal:Double = 0.0
    
    if (transactionType.equals("CASH_OUT")){
        transactionVal = 1.0
    }
    else if (transactionType.equals("DEBIT")){
        transactionVal = 2.0
    }
    else if (transactionType.equals("PAYMENT")){
        transactionVal = 3.0
    }
    else if (transactionType.equals("TRANSFER")){
        transactionVal = 4.0
    }
    
    return transactionVal;

}
  
}