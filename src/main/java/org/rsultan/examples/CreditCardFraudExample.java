package org.rsultan.examples;

import java.io.File;
import java.io.IOException;
import java.util.UUID;
import org.rsultan.core.Models;
import org.rsultan.core.clustering.ensemble.evaluation.TPRThresholdEvaluator;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.TrainTestDataframe;

public class CreditCardFraudExample {

  // Dataset present in resources/credit-card/creditcard.csv

  // threshold = 0.50999999999999996
  // =========================================
  //                TPR ║                  FPR
  // =========================================
  //  0.816993464052276 ║ 0.034716848399577914
  // =========================================

  public static final String TEMP_DIR = System.getProperty("java.io.tmpdir");

  public static void main(String[] args) throws IOException, ClassNotFoundException {
    var df = Dataframes.csv(args[0], ",", "\"", true).mapWithout("id");
    File file = new File(TEMP_DIR + File.separator
        + "credit-card-fraud-detection" + UUID.randomUUID() + ".gz");
    String pathName = file.toPath().toString();
    TrainTestDataframe dataframe = Dataframes.trainTest(df.getColumns()).setSplitValue(0.7);
    if
     (!file.exists()) {
      int nbTrees = 200;
      IsolationForest model = new IsolationForest(nbTrees)
          .train(df.mapWithout("Class"));
      var evaluator = new TPRThresholdEvaluator("Class", "anomalies")
          .setDesiredTPR(0.7)
          .setLearningRate(0.1);
      Double threshold = evaluator.evaluate(model, dataframe);
      System.out.println("threshold = " + threshold);
      evaluator.showMetrics();
      Models.write(pathName, model);
    } else {
      IsolationForest isolationForest = Models.read(pathName);
      isolationForest.predict(df).select("Class", "anomalies").show(1000000);
    }
  }
}
