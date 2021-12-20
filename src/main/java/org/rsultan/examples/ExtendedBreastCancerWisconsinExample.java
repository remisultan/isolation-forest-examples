package org.rsultan.examples;

import org.rsultan.core.clustering.ensemble.evaluation.TPRThresholdEvaluator;
import org.rsultan.core.clustering.ensemble.isolationforest.ExtendedIsolationForest;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.core.dimred.PrincipalComponentAnalysis;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.TrainTestDataframe;

import java.io.IOException;
import java.util.Objects;

public class ExtendedBreastCancerWisconsinExample {

    // Dataset present in resources/breast_cancer/wdbc-file.data

    //  threshold = 0.40000000000000013
    //  =======================================
    //                 TPR ║                FPR
    //  =======================================
    //  0.9838709677419355 ║ 0.6330275229357798
    //  =======================================

  public static void main(String[] args) throws IOException {
    var dataframe = Dataframes.csv(args[0], ",", "\"", false)
        .mapWithout("c0")
        .map("response", (String s) -> Objects.equals(s, "M") ? 1L : 0L, "c1")
        .mapWithout("c1");

    // We run the data through a PCA to extract only the principal components of the data
    var PCA = new PrincipalComponentAnalysis(10).setResponseVariable("response");
    var df = PCA.train(dataframe).predict(dataframe);

    TrainTestDataframe dfTrainTest = Dataframes.trainTest(df.getColumns()).setSplitValue(0.7);
    IsolationForest model = new ExtendedIsolationForest(500, 9);
    var evaluator = new TPRThresholdEvaluator("response", "anomalies")
        .setDesiredTPR(0.7)
        .setLearningRate(0.1);
    Double threshold = evaluator.evaluate(model, dfTrainTest);
    System.out.println("threshold = " + threshold);
    evaluator.showMetrics();
  }
}
