package org.rsultan.etl;

import org.rsultan.core.dimred.PrincipalComponentAnalysis;
import org.rsultan.dataframe.Dataframes;

import java.io.IOException;

public class PcaCreditCard {

  // To work with the dataset again
  // http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
  // with the kddcup.data_10_percent.gz

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", true);


    var PCA = new PrincipalComponentAnalysis(3).setResponseVariable("Class").train(df);

    PCA.predict(df).write(args[1], ",", "\"");
  }

}
