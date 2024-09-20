# Alfie Godley
# 14/02/2022
# Generating the second figure

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from math import sqrt, pi

data = pd.read_csv('../../../data/Model1/fig4_2.csv', header=None)
# Extracts FI data removing label from row
F = np.delete(data.iloc[0].to_numpy(), [0])
# Extracts angles data removing label from row
angs = np.delete(data.iloc[1].to_numpy(), [0])

# For QFI
n = 20
lmbd = 0.1
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
ax.plot(angs, F)
ax.plot(np.linspace(0, pi, 50), np.ones(50)*4*QFI_exact)
ax.legend(['FI', 'QFI'])
ax.set_xlabel('Measurement angle')
plt.show()

