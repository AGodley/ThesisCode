# Alfie Godley
# Histograms of the counting measurements data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Finds directory for the project
adaptiveMeasurementSimulation = (Path.cwd()).parents[2]
data = adaptiveMeasurementSimulation.joinpath('data').joinpath('Model2')

# Imports data
# N=100
estimates100 = pd.read_csv(data.joinpath('counting_thetas.csv'), header=None, skiprows=8, nrows=1)
# N=1000
estimates1000 = pd.read_csv(data.joinpath('thetas_adaptive.csv'), header=None, skiprows=10, nrows=1)

# N=100 Plot
fig = plt.figure()
ax = plt.subplot(111)
ax.hist(estimates100.iloc[0, 0:], bins=75)
ax.set_title(f'Counting Measurements: (N=100, n=300)')
ax.set_xlabel(f'$\\theta$')
ax.set_ylabel(f'Frequency')
plt.show()

# N=1000 Plot
fig2 = plt.figure()
ax2 = plt.subplot(111)
ax2.hist(estimates1000.iloc[0, 0:], bins=75)
ax2.set_title(f'Counting Measurements: (N=1000, n=300)')
ax2.set_xlabel(f'$\\theta$')
ax2.set_ylabel(f'Frequency')
plt.show()