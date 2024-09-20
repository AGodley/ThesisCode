# Alfie Godley
# 21/11/2022
# Generating the first figure for our new model

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
full_adaptive = pd.read_csv(data.joinpath('full_adaptive.csv'))
full_adaptive_no_init = pd.read_csv(data.joinpath('full_adaptive_no_initial_est.csv'))
full_simple = pd.read_csv(data.joinpath('full_adaptive_simple_meas.csv'))
Q = 11.06782379410951    # QFI per step

# Figure setup
fig1, ax1 = plt.subplots(1, 1)  # Creates figure and axes
# plt.rcParams['text.usetex'] = True  # Activates Tex
# Labels
ax1.set_xlabel('Trajectory length, n', fontsize=14)
ax1.set_ylabel('Fisher information', fontsize=14)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Plots
# Full adaptive case
plt.plot(full_adaptive.loc[25:28, 'n'], full_adaptive.loc[25:28, 'I_sample'])
# Full adaptive case derivative method
plt.plot(full_adaptive.loc[25:28, 'n'], full_adaptive.loc[25:28, 'I_dm'])
# Perfect adaptive case - not included
# plt.plot(full_adaptive_no_init.loc[12:, 'n'], full_adaptive_no_init.loc[12:, 'I_dm'])
# Simple measurements
plt.plot(full_simple.loc[22:, 'n'], full_simple.loc[22:, 'I_sample'])
# Simple measurements derivative method
plt.plot(full_simple.loc[22:, 'n'], full_simple.loc[22:, 'I_dm'])
# QFI - Imported from the terminal readout
plt.plot(full_adaptive.loc[25:28, 'n'], full_adaptive.loc[25:28, 'n']*Q)
# Legend
# plt.legend(['Adaptive', 'Adaptive - DM', 'Simple', 'Simple - DM', 'Q'])
# Shows the figure
plt.show()

# Figure for sampling FI
# Figure setup
fig2, ax2 = plt.subplots(1, 1)  # Creates figure and axes
# plt.rcParams['text.usetex'] = True  # Activates Tex
# Labels
ax2.set_xlabel('Trajectory length, n', fontsize=14)
ax2.set_ylabel('Fisher information', fontsize=14)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Sampling FI data
plt.plot(full_adaptive_no_init.loc[18:, 'n'], full_adaptive_no_init.loc[18:, 'I'])
# Derivative method FI data
plt.plot(full_adaptive_no_init.loc[18:, 'n'], full_adaptive_no_init.loc[18:, 'I_dm'])
# QFI FI data
plt.plot(full_adaptive_no_init.loc[18:, 'n'], full_adaptive_no_init.loc[18:, 'n']*Q)
plt.legend(['Sampling', 'Derivative method', 'Q'])
plt.show()
