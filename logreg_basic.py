import numpy    as np
import pandas   as pd

from sklearn.model_selection    import train_test_split
from sklearn.linear_model       import LogisticRegression

def logreg(part_number_a, part_number_b):
    # Import the datasets from the repository as Pandas dataframes.
    data_a = pd.read_csv("data_rmse/" + part_number_a + ".csv")
    data_b = pd.read_csv("data_rmse/" + part_number_b + ".csv")

    # Apply labels to the datasets.
    data_a["y"] = 1
    data_b["y"] = 0

    # Combine the datasets and randomize them.
    data = pd.concat([data_a, data_b]).sample(frac = 1)

    # Generate testing and training sets from the data.
    X = data.loc[:, data.columns != "y"]
    y = data.loc[:, data.columns == "y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Train the logistic regression model.
    model = LogisticRegression(solver = 'liblinear')
    model.fit(X_train, np.asarray(y_train).ravel())

    return model.score(X_test, y_test)

devices = ["lm555cmx", "lmc555im", "lmc555imx", "msp430g2210id", "msp430g2210idr", "msp430g2230id", "msp430g2230idr"]

for a in devices:
    for b in devices:

        scores = []


        for n in range(1000):
            print("Performing analysis " + str(n + 1) + " of 1000.")
            scores.append(logreg(a, b))

        scores = np.array(scores)
        mean = np.mean(scores)
        median = np.median(scores)
        std = np.std(scores)

        print("Mean = " + str(mean))
        print("Median = " + str(median))
        print("STD = " + str(std))

        file = open("results/logreg_basic.csv", mode="a", newline="")
        file.write(a + "," + b + "," + str(mean) + "," + str(median) + "," + str(std) + "\n")
        file.close()