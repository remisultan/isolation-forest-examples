package org.rsultan.etl;

import java.io.IOException;
import org.rsultan.core.dimred.PrincipalComponentAnalysis;
import org.rsultan.dataframe.Dataframes;

public class PcaKDDCUP99 {

  // To work with the dataset again
  // http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
  // with the kddcup.data_10_percent.gz

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", false)
        .filter("c2", "http"::equals)
        .mapWithout("c0", "c1", "c2", "c3")
        .map("response", (String s) -> s.contains("normal") ? 0L : 1L, "c41")
        .mapWithout("c41");


    var PCA = new PrincipalComponentAnalysis(10).setResponseVariable("response")
        .train(df);

    PCA.predict(df).write(args[0], ",", "\"");
  }

}
