# Alfie Godley
# 09/02/2022
# Generating the first figure

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from math import sqrt

# Sets up figure
fig1, ax1 = plt.subplots(1, 1)

data = pd.read_csv('../../../data/dataArchive/fig1_2_adaptive_measure_system.csv')
I_adapt_measure = data['I'].to_numpy()
n_vals = data['n'].to_numpy()

# Defines function for exact QFI
lmbd = 0.1
QFI_exact = np.zeros(len(n_vals))
for i in np.arange(len(n_vals)):
    n = int(n_vals[i])
    QFI_exact[i] = (1-(sqrt(1-lmbd))**n)**2/(1-sqrt(1-lmbd))**2 \
                + (1-lmbd)**(n-1)\
                + (n-1)*lmbd/(1-sqrt(1-lmbd))**2 \
                + (1-lmbd - (1-lmbd)**n)/(1-sqrt(1-lmbd))**2 \
                - 2*lmbd*(sqrt(1-lmbd)-(sqrt(1-lmbd))**n)/((1-sqrt(1-lmbd))**3) \
                + (1- (1-lmbd)**(n-1))/lmbd \
                -(1-lmbd-(1-lmbd)**n)/lmbd - 1 + n*(1-lmbd)**(n-1)\
                +n*(1- (1-lmbd)**(n-1))

data2 = pd.read_csv('../../../data/dataArchive/fig1_2_adaptive.csv')
I_adapt = data2['I'].to_numpy()
n_vals2 = data2['n'].to_numpy()

data3 = pd.read_csv('../../../data/dataArchive/fig1_2_regular.csv')
I_reg = data3['I'].to_numpy()
n_vals3 = data3['n'].to_numpy()

data4 = pd.read_csv('../../../data/dataArchive/fig1_2_regular_system_measure.csv')
I_reg_syst = data4['I'].to_numpy()
n_vals4 = data4['n'].to_numpy()

# Making the figure look better
# Labels
ax1.set_xlabel('Trajectory length, n', fontsize=14)
ax1.set_ylabel('Fisher information', fontsize=14)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Plots the exact QFI
ax1.plot(n_vals, 4*QFI_exact, linewidth=2)
# Plots the adaptive w system measurements FI
ax1.plot(n_vals, I_adapt_measure, linewidth=2)
# Plots the adaptive FI
ax1.plot(n_vals2, I_adapt, linewidth=2)
# Plots the regular FI
ax1.plot(n_vals3, I_reg, linewidth=2)
#
# ax1.plot(n_vals4, I_reg_syst)
# Legend
ax1.legend(['QFI', 'Adaptive w/ system measurement', 'Adaptive', 'Regular'])
plt.show()