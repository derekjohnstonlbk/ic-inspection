#=============================================================================#
#   RF-based inspection via Logistic Regression
#   Developed By: Derek Johnston
#
#   The goal of this project is to develop a technique for classifying 
#   integrated circuits by performing S11 measurements and using a
#   Logistic Regression Algorithm
#
#=============================================================================#
# Import the required dependencies
import numpy    as np
import pandas   as pd

from csv                        import reader, writer
from datetime                   import datetime
from math                       import sqrt
from nidaqmx                    import Task
from nidaqmx.constants          import LineGrouping
from pyvisa                     import ResourceManager
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import classification_report
from sklearn.model_selection    import train_test_split
from time                       import sleep, time
#=============================================================================#
#   collect_s11_sample
#
#   Collect S11 measurement data for a single device under test.
#
#   @returns:
#       - sample (List): The measurements taken for the sample
#
#=============================================================================#
def collect_s11_sample():
    # Prompt the user for the name of the sample.
    sample_id = input("Enter a name for the sample: ")

    # Set the NI MyDAQ to output 3.3 VDC on analog channel 0.
    with Task() as task:
        task.ao_channels.add_ao_voltage_chan("myDAQ1/ao0")
        task.write([3.3], auto_start=True)
    
    # Prepare an empty list to contain the data.
    sample = []

    # Perform an S11 measurement on every pin permutation (256)
    for permutation in range(256):
        print("Performing S11 measurement on permutation " + str(permutation) + " of 256.")

        # Add an empty list to the sample list.
        sample.append([])

        # Write to the SIPO registers on the PCB to set the relays.
        with Task() as task:
            task.do_channels.add_do_chan("myDAQ1/port0/line0:2", line_grouping = LineGrouping.CHAN_FOR_ALL_LINES)

            # Convert the permutation into an 8-bit binary string.
            binary = format(permutation, "08b")

            # Zero-out the channels to start.
            task.write(0, 2)

            # Write each bit to the shift register.
            for bit in binary:
                if bit == "1":
                    task.write(1, 2)
                    task.write(3, 2)
                else:
                    task.write(0, 2)
                    task.write(2, 2)

            # Cycle the register clock to load the parallel register.
            task.write(4, 2)
            task.write(0, 2)
        
        # Delay for 700 ms to account for relay bounce.
        sleep(0.70)

        # Utilize the Keysight E5063A ENA to collect an S11 measurement.
        resource_manager = ResourceManager()
        ENA5063 = resource_manager.open_resource("ENA5063")
        ENA5063.write(':INITiate1:CONTinuous %d' % (1))
        ENA5063.write(':CALCulate1:PARameter1:DEFine %s' % ('S11'))
        ENA5063.write(':CALCulate1:PARameter1:SELect')
        ENA5063.write(':TRIGger:SEQuence:SOURce %s' % ('MANual'))
        ENA5063.write(':TRIGger:SEQuence:SINGle')
        ENA5063.write('*OPC')
        ENA5063.write(':INITiate1:CONTinuous %d' % (0))
        ENA5063.write(':CALCulate1:SELected:FORMat %s' % ('MLOGarithmic'))
        ENA5063.write(':FORMat:DATA %s' % ('REAL'))
        ENA5063.write(':FORMat:BORDer %s' % ('SWAPped'))
        measurement = ENA5063.query_binary_values(':CALCulate1:SELected:DATA:FDATa?','d',False)
        ENA5063.close()
        resource_manager.close()

        # The measurement data is returned with 0.0 in odd-indexed elements.
        # We need to iterate over the array and remove these elements before
        # adding them to the data list.
        for j in range(len(measurement)): 
            if j % 2 == 0: sample[permutation].append(measurement[j])

    # Record the data by writing it to a .csv file and storing it in data_raw
    with open("data_raw/" + sample_id + ".csv", mode = "w", newline = "") as file:
        writer(file).writerows(sample)

    # Prompt the user to give the part number for the sample so that it may
    # be added to that device's RMSE dataset.
    device = input("Enter the part number for the device: ")

    baseline = []
    # Use the empty_socket.csv dataset as the baseline for the RMSE calculation.
    with open("data_raw/empty_socket.csv") as file:
        data = reader(file)
        
        for idx, row in enumerate(data):
            baseline.append([])
            for element in row:
                baseline[idx].append(float(element))

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
    with open("data_rmse/" + device + ".csv", mode = "a", newline = "") as file:
        writer(file).writerow(rmse)

#=============================================================================#
#   logreg_analysis
#
#   For a pair of device datasets, perform a logistic regression analysis
#   and record the results.
#
#=============================================================================#
def logreg_analysis():
    # Select the datasets to perform the analysis on.
    part_number_a = input("Enter the part number for class A: ")
    part_number_b = input("Enter the part number for class B: ")

    # Import the datasets from the repository as Pandas dataframes.
    print("Importing datasets for class A (" + part_number_a + ") and class B (" + part_number_b + ").\n")
    data_a = pd.read_csv("data_rmse/" + part_number_a + ".csv")
    data_b = pd.read_csv("data_rmse/" + part_number_b + ".csv")

    # Apply labels to the datasets.
    print("Applying labels to the dataset (A = 1, B = 0).\n")
    data_a["y"] = 1
    data_b["y"] = 0

    # Combine the datasets and randomize them.
    print("Combining and randomizing the datasets.\n")
    data = pd.concat([data_a, data_b]).sample(frac = 1)

    # Generate testing and training sets from the data.
    print("Splitting the dataset into training (7/10) and testing (3/10) datasets.\n")
    X = data.loc[:, data.columns != "y"]
    y = data.loc[:, data.columns == "y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # Train the logistic regression model.
    print("Training the logistic regression model.\n")
    model = LogisticRegression(solver = 'liblinear')
    model.fit(X_train, np.asarray(y_train).ravel())

    # Test the logistic regression model.
    print("Testing the logistic regression model.\n")
    y_pred = model.predict(X_test)

    # Generate the classification report.
    report = classification_report(y_test, y_pred)
    print("*** CLASSIFICATION REPORT ***")
    print(report + "\n")

    # Generate a report of the test results.
    results = "Report of Results for IC Inspection Analysis\n\n"
    results += "Timestamp: " + datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S' + "\n\n")
    results += "Device Class A: " + part_number_a + " (N = " + str(data_a.shape[0]) + ")\n"
    results += "Device Class B: " + part_number_b + " (N = " + str(data_b.shape[0]) + ")\n\n"
    results += "*** Classification Report ***\n\n"
    results += report
    results += "\n\n*** Model Coefficients ***\n\n"
    results += str(model.coef_)

    # Save the results to a textfile.
    print("Storing results in a textfile.\n")
    file = open("results/results_" + part_number_a + "_" + part_number_b + ".txt", mode = "w")
    file.write(results)
    file.close()

    # End of program
    print("Analysis is complete.\n")