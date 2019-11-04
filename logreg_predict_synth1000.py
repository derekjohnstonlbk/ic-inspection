#=============================================================================#
#   logreg_predict_synth1000.py
#   Developed By: Derek Johnston
#   
#   Use the stored logistic regression models to scan and predict the 
#   part number of a device from an S11 scan.
#=============================================================================#
import numpy    as np
import pandas   as pd
import pickle   as pk

from csv                    import reader
from math                   import sqrt
from nidaqmx                import Task
from nidaqmx.constants      import LineGrouping
from pyvisa                 import ResourceManager
from sklearn.linear_model   import LogisticRegression
from time                   import sleep, time

input("Ensure that all of the instruments are powered on and configured properly.")
input("Ensure that the DUT is seated firmly in the test socket.")

# Scan the device using the S11 method.
with Task() as task:
    task.ao_channels.add_ao_voltage_chan("myDAQ1/ao0")
    task.write([3.3], auto_start=True)

sample = []

for permutation in range(256):
    print("Performing S11 measurement on permutation " + str(permutation + 1) + " of 256.")

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

# Compute the RMSE value for the sample 
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

devices = [
    "lm555cmx",
    "lmc555im",
    "lmc555imx",
    "msp430g2210id",
    "msp430g2210idr",
    "msp430g2230id",
    "msp430g2230idr"
]

predictions = {}
for device in devices:
    predictions[device] = 0

for device_a in devices:
    for device_b in devices:
        with open("pickles/model_" + device_a + "_" + device_b + ".pkl", "rb") as file:
            model = pk.load(file)
            label = model.predict([rmse])
            print("Prediction for device " + device_a + " vs. " + device_b + " = " + str(label[0]))
            if label[0] == 1:
                predictions[device_a] += 1
            else:
                predictions[device_b] += 1

print(predictions)

pred = ""
votes = 0
for key in predictions:
    vote = predictions[key]
    if vote > votes:
        votes = vote
        pred = key

print("This device most likely has the part number: " + pred)

