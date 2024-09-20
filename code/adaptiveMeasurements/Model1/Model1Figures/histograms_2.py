# Alfred Godley

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from math import sqrt, pi

# (N=10,000, n=20) Bounded
data = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=2, nrows=1)
F = np.delete(data.iloc[0].to_numpy(), [0])
# (N=10,000, n=30) Bounded
data = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=5, nrows=1)
F2 = np.delete(data.iloc[0].to_numpy(), [0])
# (N=1000, n=20) Bounded
data = pd.read_csv('../../../data/Model1/fig3_adaptive.csv', header=None, skiprows=17, nrows=1)
F6 = np.delete(data.iloc[0].to_numpy(), [0])


fig, ax = plt.subplots()
ax.hist(F6, bins=30)
# ax.set_title('(N=20000, n=50) - Bounded')
ax.set_xlabel('Estimate of $\\theta$', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Show
plt.show()

