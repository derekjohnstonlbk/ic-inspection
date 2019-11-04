#=============================================================================#
#   logreg_store_synth1000.py
#   Developed By: Derek Johnston
#
#   For each combinational pair of devices, train a logistic regression
#   classifier to classify between two types by looking at the s11 datasets
#   and generating a synthetic training set of N = 1000. Store the 
#   coefficients so that they may be reused for scanning and classification.
#
#=============================================================================#
import numpy    as np
import pandas   as pd
import pickle   as pk

from sklearn.linear_model   import LogisticRegression

def generate_synth_data(device_name, n):
    data = pd.read_csv("data_rmse/" + device_name + ".csv")
    synth_data = pd.DataFrame()

    for label in data.columns:
        col = np.array(data[label].tolist())
        avg = np.mean(col)
        std = np.std(col)

        synth_col = np.random.normal(loc=avg, scale=std, size=n)
        synth_data[label] = synth_col

    return synth_data

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
        print("Generating and storing models for " + device_a + " and " + device_b + ".")
        data_a = generate_synth_data(device_a, 1000)
        data_b = generate_synth_data(device_b, 1000)
        data_a["y"] = 1
        data_b["y"] = 0
        data = pd.concat([data_a, data_b]).sample(frac = 1)
        X = data.loc[:, data.columns != "y"]
        y = data.loc[:, data.columns == "y"]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = LogisticRegression(solver = 'liblinear')
        model.fit(X, np.asarray(y).ravel())
        with open("pickles/model_" + device_a + "_" + device_b + ".pkl", "wb") as file:
            pk.dump(model, file)