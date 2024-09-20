# Alfred Godley

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from math import sqrt, pi

#
data = pd.read_csv('../../../data/Model1/fig3_regular_non_opt.csv', header=None, skiprows=2, nrows=1)
F = np.delete(data.iloc[0].to_numpy(), [0])

fig, ax = plt.subplots()
ax.hist(F)
ax.set_title('')
ax.set_xlabel('Estimate of $\\theta$', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()