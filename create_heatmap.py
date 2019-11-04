import pandas               as pd
import seaborn              as sns
import matplotlib.pyplot    as plt 

devices = [
    "lm555cmx",
    "lmc555im",
    "lmc555imx",
    "msp430g2210id",
    "msp430g2210idr",
    "msp430g2230id",
    "msp430g2230idr"
]

data = pd.read_csv("results/logreg_synthetic_1000.csv")
counter = 0
counter_long = 0
matrix = []
for idx, value in enumerate(data.values):
    if counter == 0:
        print("1")
        matrix.append([])
        matrix[counter_long].append(value[2])
        counter += 1
    elif counter < 6:
        print("2")
        matrix[counter_long].append(value[2])
        counter += 1
    else:
        print("3")
        matrix[counter_long].append(value[2])
        counter = 0
        counter_long += 1

print(matrix)

ax = sns.heatmap(matrix, xticklabels=devices, yticklabels=devices, annot=True, cmap="Greys")
ax.figure.tight_layout()
plt.title("Logistic Regression Scores for Synthetic S11 (N = 1000).")
plt.show()