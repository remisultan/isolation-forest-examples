package org.rsultan.examples;

import java.io.IOException;
import org.rsultan.core.clustering.ensemble.evaluation.TPRThresholdEvaluator;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.TrainTestDataframe;

public class HttpDatasetExample {

  // Dataset present in resources/http/

  // PCA dataset of 3 features
  // Dataset Split of 0.7
  // threshold = 0.6000000000000001
  // ========================================
  //                TPR ║                 FPR
  // ========================================
  // 0.90633608815427 ║ 0.03318607908630535
  // ========================================

  // PCA dataset of 10 features
  // Dataset Split of 0.7
  // threshold = 0.5199999999999996
  // ========================================
  //                TPR ║                 FPR
  // ========================================
  // 0.8472972972972973 ║ 0.03666163467759327
  // ========================================

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csvTrainTest(args[0], ",", "\"", false);

    TrainTestDataframe dfTrainTest = Dataframes.trainTest(df.getColumns()).setSplitValue(0.7);
    IsolationForest model = new IsolationForest(10);
    var evaluator = new TPRThresholdEvaluator("c3", "anomalies")
        .setDesiredTPR(0.9)
        .setLearningRate(0.001);
    Double threshold = evaluator.evaluate(model, dfTrainTest);
    System.out.println("threshold = " + threshold);
    evaluator.showMetrics();
  }
}
