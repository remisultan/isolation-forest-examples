package org.rsultan.examples;

import org.rsultan.core.clustering.ensemble.evaluation.TPRThresholdEvaluator;
import org.rsultan.core.clustering.ensemble.isolationforest.ExtendedIsolationForest;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.TrainTestDataframe;

import java.io.IOException;

public class ExtendedHttpDatasetExample {

  // Dataset present in resources/http/

  // PCA dataset of 3 features
  // Dataset Split of 0.7
  // Desired TPR 0.9
  // threshold = 0.6329999999999997
  // ========================================
  //                TPR ║                 FPR
  // ========================================
  // 0.9116409537166901 ║ 0.02363391655450875
  // ========================================

  // PCA dataset of 10 features
  // Dataset Split of 0.7
  // Desired TPR 0.9
  // threshold = 0.6159999999999997
  // ========================================
  //                TPR ║                 FPR
  // ========================================
  // 0.9092261904761905 ║ 0.03470133218736571
  // ========================================


  public static void main(String[] args) throws IOException {
    var df = Dataframes.csvTrainTest(args[0], ",", "\"", false);

    TrainTestDataframe dfTrainTest = Dataframes.trainTest(df.getColumns()).setSplitValue(0.7);
    IsolationForest model = new ExtendedIsolationForest(200, 9);
    var evaluator = new TPRThresholdEvaluator("c10", "anomalies")
        .setDesiredTPR(0.9)
        .setLearningRate(0.001);
    Double threshold = evaluator.evaluate(model, dfTrainTest);
    System.out.println("threshold = " + threshold);
    evaluator.showMetrics();
  }
}
