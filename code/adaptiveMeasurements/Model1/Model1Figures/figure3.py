import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from math import sqrt

# Sets up figure
fig1, ax1 = plt.subplots(1, 1)

# Adaptive data
# Extracts I_20
data_20 = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=1, nrows=1)
I_20 = (data_20.iloc[0].to_numpy())[0]
# Extracts I_30
data_30 = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=4, nrows=1)
I_30 = (data_30.iloc[0].to_numpy())[0]
# Extracts I_40
data_40 = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=7, nrows=1)
I_40 = (data_40.iloc[0].to_numpy())[0]
# Extracts I_50
data_50 = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=10, nrows=1)
I_50 = (data_50.iloc[0].to_numpy())[0]
# Extracts I_100
data_100 = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=13, nrows=1)
I_100 = (data_100.iloc[0].to_numpy())[0]
# Extracts I_200
data_200 = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=16, nrows=1)
I_200 = (data_200.iloc[0].to_numpy())[0]
# Extracts I_300
data_300 = pd.read_csv('../../data/Model1/fig3_adaptive.csv', header=None, skiprows=19, nrows=1)
I_300 = (data_300.iloc[0].to_numpy())[0]
# All together
I = [I_20, I_30, I_40, I_50, I_100, I_200, I_300]
n_I = [20, 30, 40, 50, 100, 200, 300]

# Regular data
# Extracts I_20
data_20 = pd.read_csv('../../../data/Model1/fig3_regular_non_opt_outlier.csv', header=None, skiprows=1, nrows=1)
Ir_20 = (data_20.iloc[0].to_numpy())[0]
# Extracts I_30
data_30 = pd.read_csv('../../../data/Model1/fig3_regular_non_opt_outlier.csv', header=None, skiprows=4, nrows=1)
Ir_30 = (data_30.iloc[0].to_numpy())[0]
# Extracts I_40
data_40 = pd.read_csv('../../../data/Model1/fig3_regular_non_opt_outlier.csv', header=None, skiprows=7, nrows=1)
Ir_40 = (data_40.iloc[0].to_numpy())[0]
# Extracts I_50
data_50 = pd.read_csv('../../../data/Model1/fig3_regular_non_opt_outlier.csv', header=None, skiprows=10, nrows=1)
Ir_50 = (data_50.iloc[0].to_numpy())[0]
# Extracts I_100
data_100 = pd.read_csv('../../../data/Model1/fig3_regular_non_opt_outlier.csv', header=None, skiprows=13, nrows=1)
Ir_100 = (data_100.iloc[0].to_numpy())[0]
# Extracts I_200
data_200 = pd.read_csv('../../../data/Model1/fig3_regular_non_opt_outlier.csv', header=None, skiprows=16, nrows=1)
Ir_200 = (data_200.iloc[0].to_numpy())[0]
# Extracts I_300
data_300 = pd.read_csv('../../data/Model1/fig3_regular_non_opt.csv', header=None, skiprows=19, nrows=1)
Ir_300 = (data_300.iloc[0].to_numpy())[0]
# All together
Ir = [Ir_20, Ir_30, Ir_40, Ir_50, Ir_100, Ir_200, Ir_300]
n_Ir = [20, 30, 40, 50, 100, 200, 300]

# Exact QFI
# Defines function for exact QFI
lmbd = 0.8
n_vals = [20, 30, 40, 50, 100, 200, 300]
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

# Calculated adaptive
data2 = pd.read_csv('../../../data/Model1/fig1_adaptive.csv')
I_adapt = data2['I'].to_numpy()
n_vals2 = data2['n'].to_numpy()

# Calculated regular
data3 = pd.read_csv('../../../data/Model1/fig1_regular_non_opt.csv')
I_reg = data3['I'].to_numpy()
n_vals3 = data3['n'].to_numpy()

# Plots the exact QFI
ax1.plot(n_vals, 4*QFI_exact, linewidth=2)
# Plots adaptive w system measurements
ax1.plot(n_I, I, linewidth=2)
# Calculated adaptive
ax1.plot(n_vals2[2:], I_adapt[2:], linewidth=2)
# Regular
ax1.plot(n_Ir, Ir, linewidth=2)
# Calculated regular
ax1.plot(n_vals3, I_reg, linewidth=2)
# Legend
# ax1.legend(['QFI', 'Adaptive', 'Calculated adaptive', 'Regular', 'Calculated regular'], fontsize=12)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Labels
ax1.set_xlabel('Trajectory length, n', fontsize=14)
ax1.set_ylabel('Fisher information', fontsize=14)
plt.show()

# Ratio plot Calculated adaptive / QFI
fig2, ax2 = plt.subplots()
ax2.plot(n_vals, I/QFI_exact, linewidth=2)
ax2.set_xlabel('n', fontsize=14)
ax2.set_ylabel('Orange/Blue', fontsize=14)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Ratio plot QFI / Calculated adaptive
fig3, ax3 = plt.subplots()
ax3.plot(n_I, QFI_exact/I)
ax3.set_xlabel('n', fontsize=14)
ax3.set_ylabel('Blue/Orange', fontsize=14)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()