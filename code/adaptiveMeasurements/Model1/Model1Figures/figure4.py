# Alfie Godley
# 14/02/2022
# Generating the second figure

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from math import sqrt, pi

data = pd.read_csv('../../../data/Model1/fig4.csv', header=None)
# Extracts FI data removing label from row
F = np.delete(data.iloc[2].to_numpy(), [0])
# Extracts angles data removing label from row
angs = np.delete(data.iloc[3].to_numpy(), [0])

# For QFI
n = 20
lmbd = 0.8
# calculate exact QFI
QFI_exact = (1-(sqrt(1-lmbd))**n)**2/(1-sqrt(1-lmbd))**2 \
            + (1-lmbd)**(n-1)\
            + (n-1)*lmbd/(1-sqrt(1-lmbd))**2 \
            + (1-lmbd - (1-lmbd)**n)/(1-sqrt(1-lmbd))**2 \
            - 2*lmbd*(sqrt(1-lmbd)-(sqrt(1-lmbd))**n)/((1-sqrt(1-lmbd))**3) \
            + (1- (1-lmbd)**(n-1))/lmbd \
            -(1-lmbd-(1-lmbd)**n)/lmbd - 1 + n*(1-lmbd)**(n-1)\
            +n*(1- (1-lmbd)**(n-1))

fig, ax = plt.subplots()
ax.plot(angs, F, linewidth=2)
ax.plot(np.linspace(0, pi, 50), np.ones(50)*4*QFI_exact, linewidth=2)
ax.legend(['FI', 'QFI'], fontsize=12)
ax.set_xlabel('Measurement angle', fontsize=14)
ax.set_ylabel('Fisher information', fontsize=14)
# Changing ticks to degrees
ax.set_xticks([0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi])
labels = [str(i)+'$^\circ$' for i in [0, 30, 60, 90, 120, 150, 180]]
ax.set_xticklabels(labels, fontsize=12)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

