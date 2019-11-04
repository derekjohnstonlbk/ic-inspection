import numpy    as np
import pandas   as pd

from sklearn.model_selection    import train_test_split
from sklearn.linear_model       import LogisticRegression


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
        scores = []
        for n in range(10):
            print("Performing synthetic analysis for " + device_a + " vs. " + device_b + " trial " + str(n+1) + " of 10.")
            data_a = generate_synth_data(device_a, 1000)
            data_b = generate_synth_data(device_b, 1000)
            data_a["y"] = 1
            data_b["y"] = 0
            data = pd.concat([data_a, data_b]).sample(frac = 1)
            X = data.loc[:, data.columns != "y"]
            y = data.loc[:, data.columns == "y"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model = LogisticRegression(solver = 'liblinear')
            model.fit(X_train, np.asarray(y_train).ravel())
            score = model.score(X_test, y_test)
            scores.append(score)
        
        scores = np.array(scores)
        mean = np.mean(scores)
        median = np.median(scores)
        std = np.std(scores)

        file = open("results/logreg_synthetic_1000.csv", mode="a", newline="")
        file.write(device_a + "," + device_b + "," + str(mean) + "," + str(median) + "," + str(std) + "\n")
        file.close()  
        