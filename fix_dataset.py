from csv import reader, writer
from math import sqrt

dut = "msp430g2230idr"
dut_long = "msp430g2230idr_dont_"


baseline = []
with open("data_raw/empty_socket.csv") as file:
    data = reader(file)
    for idx, row in enumerate(data):
        baseline.append([])
        for element in row:
            baseline[idx].append(float(element))

for i in range(32):
    num = str(i)
    if i < 10:
        num = "0" + str(i)
    
    sample = []
    with open("data_raw/" + dut_long + num + ".csv") as file:
        data = reader(file)
        for idx, row in enumerate(data):
            sample.append([])
            for element in row:
                sample[idx].append(float(element))
    
    # Compute the Root Mean Squared Error (RMSE) for the sample.
    rmse = []
    for idx, _ in enumerate(sample):
        test    = sample[idx]
        control = baseline[idx]

        # Calculate the squared error
        squared_error = 0
        for jdx, _ in enumerate(test):
            squared_error += (test[jdx] - control[jdx]) ** 2

        # Divide by the length of the row to get the "mean" squared error.
        mean_se = squared_error / len(test)

        # Take the square root of the mean squared error.
        rmse.append(sqrt(mean_se))

    # Append this data to the dataset for the indicated device.
    with open("data_rmse/" + dut + ".csv", mode = "a", newline = "") as file:
        writer(file).writerow(rmse)