# Alfie Godley
# Generating histograms of the theta estimates

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from math import sqrt
from pathlib import Path    # For directories

# Finds directory for the project
adaptiveMeasurementSimulation = (Path.cwd()).parents[2]
data = adaptiveMeasurementSimulation.joinpath('data').joinpath('Model2')

# Imports data
estimates = pd.read_csv(data.joinpath('thetas_adaptive.csv'), header=None, skiprows=6, nrows=1)
# F = np.delete(data.iloc[0].to_numpy(), [0])

print(estimates.iloc[0, 0:])
fig, ax = plt.subplots()
ax.hist(estimates.iloc[0, 0:], bins=30)
ax.set_title('Adaptive with n=500')
ax.set_xlabel('Estimate of $\\theta$', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()