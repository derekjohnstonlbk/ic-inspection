import numpy as np
import pandas as pd 

from sklearn.model_selection    import train_test_split
from sklearn.linear_model       import LogisticRegression
from imblearn.over_sampling     import SMOTE

devices = [
    "lm555cmx",
    "lmc555im",
    "lmc555imx",
    "msp430g2210id",
    "msp430g2210idr",
    "msp430g2230id",
    "msp430g2230idr"
]

for device_a in devices:
    for device_b in devices:
        scores = []
        for n in range(1000):
            print("Performing analysis for " + device_a + " vs. " + device_b + " trial " + str(n+1) + " of 1000.")
            data_a = pd.read_csv("data_rmse/" + device_a + ".csv")
            data_b = pd.read_csv("data_rmse/" + device_b + ".csv")
            data_a["y"] = 1
            data_b["y"] = 0
            data = pd.concat([data_a, data_b]).sample(frac = 1)
            X = data.loc[:, data.columns != "y"]
            y = data.loc[:, data.columns == "y"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            columns = X_train.columns 
            os_data_X, os_data_y = SMOTE().fit_sample(X_train, np.asarray(y_train).ravel())
            os_data_X = pd.DataFrame(data = os_data_X, columns = columns)
            os_data_y = pd.DataFrame(data = os_data_y, columns = ["y"]) 
            model = LogisticRegression(solver="liblinear")
            model.fit(os_data_X, np.asarray(os_data_y.values).ravel())
            score = model.score(X_test, y_test)
            scores.append(score)
        
        scores = np.array(scores)
        mean = np.mean(scores)
        median = np.median(scores)
        std = np.std(scores)

        file = open("results/logreg_smote.csv", mode="a", newline="")
        file.write(device_a + "," + device_b + "," + str(mean) + "," + str(median) + "," + str(std) + "\n")
        file.close()