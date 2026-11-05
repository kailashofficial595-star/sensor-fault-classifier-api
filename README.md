# sensor-fault-classifier-api
Machine learning model that predicts industrial equipment faults from sensor data, served as a containerized FastAPI service with tests and live deployment.
Cross-validation table:-
Best threshold: 0.567
              precision    recall  f1-score   support

           0      0.993     0.997     0.995      1932
           1      0.900     0.794     0.844        68

    accuracy                          0.990      2000
   macro avg      0.946     0.896     0.919      2000
weighted avg      0.990     0.990     0.990      2000

[[1926    6]
 [  14   54]]

the dataset is synthetic and that the engineered features reflect its documented failure rules.
the threshold was chosen on training data only, and the test set was used once.