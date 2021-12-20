# Isolation forest example

This repository intends to show the performance of the Isolation Forest method on a few datasets
based on my [java-ml](https://github.com/remisultan/java-ml)

## Requirements

- JDK17+

## Getting started

Clone the repo and run

```bash
 $ ./mvnw clean install
```

Once you are good with this you can run the different main classes:

- `BreastCancerWisconsinExample.class`
  
  ```
    // threshold = 0.40000000000000013
    // =======================================
    //                TPR ║                FPR
    // =======================================
    // 0.8269230769230769 ║ 0.3445378151260504
    // =======================================
  
  The optimal threshold is 0.4 with a desired TPR of at least 0.8.
  The issue here is that we have to work with a very high FPR which is not good.
  The reasons are mainly due to the data not being suited for anomaly detection.
  There are too many malignent vs benign cases which makes it difficult to isolate
  anomalies.
  ```
  
- `CreditCardFraudExample.class`
To retry make sure the credit card dataset is rebuilt by merging all the parts from 1 to 4
```
  // threshold = 0.50999999999999996
  // =========================================
  //                TPR ║                  FPR
  // =========================================
  //  0.816993464052276 ║ 0.034716848399577914
  // =========================================

  The optimal threshold is 0.51 with a desired TPR of 0.8
  We get very good results here, we have a very low FPR score while our TPR
  respects the requirements.
``` 

- `HttpDatasetExample.class`

```
  // PCA dataset of 3 features
  // threshold = 0.6000000000000001
  // ========================================
  //                TPR ║                 FPR
  // ========================================
  // 0.90633608815427 ║ 0.03318607908630535
  // ========================================

  // PCA dataset of 10 features
  // threshold = 0.5199999999999996
  // ========================================
  //                TPR ║                 FPR
  // ========================================
  // 0.8472972972972973 ║ 0.03666163467759327
  // ========================================
  
  Here we worked with the kddcup dataset and applied a PCA from 40 features to 
  10 and 3. (See PcaKDDCUP99.class)
  We get very good results on the 3-feature reduced dataset with a threshold of 0.6
  and a desired TPR of at least 0.8 while keeping our FPR low.
```

- `ExtendedHttpDatasetExample.class`

```
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

  If we compare the with regular IsolationForests, we see that the FPR does not change much whereas we have better
  TPR. 

```
