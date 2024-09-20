import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path   # For file directories
from scipy.stats import poisson, norm


# # Poisson distribution
# def poisson(lmbda, k):
#     return lmbda**k * np.exp(-lmbda) / np.math.factorial(k)


# Finds directory for the data
adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
counting = adaptiveMeasurementSimulation.joinpath('data').joinpath('countingMeasurements')

# Reads the data
data = pd.read_csv(counting.joinpath('counting_Markov_excitations_full_analysis.csv'))

# Reads the expected counts
expected = pd.read_csv(counting.joinpath('counting_Markov_excitations_full_expected.csv'))

# Isolated ones
ones = data['1']
# Finds the max and min values
max_ones = max(ones)
min_ones = min(ones)

# 11
pat_11 = data['11']
# Finds the max and min values
max_11 = max(pat_11)
min_11 = min(pat_11)

# 111
pat_111 = data['111']
# Finds the max and min values
max_111 = max(pat_111)
min_111 = min(pat_111)

# 1111
pat_1111 = data['1111']
# Finds the max and min values
max_1111 = max(pat_1111)
min_1111 = min(pat_1111)

# 101
pat_101 = data['101']
# Finds the max and min values
max_101 = max(pat_101)
min_101 = min(pat_101)

# Creates the figure
fig = plt.figure()
# Gridspec
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
# Title for figure
fig.suptitle('Counts of various patterns')
# Axes labels
fig.supxlabel('Counts')
fig.supylabel('Frequency')
# Creates the axes
(ax1, ax2), (ax3, ax4) = gs.subplots()

# Plots the histogram
x = np.arange(min_ones, max_ones+1)
ax1.bar(x, [np.sum(ones == i)/len(ones) for i in x], width=1)
# Axes labels
ax1.set_title('1')
# Range of k values from max to min for plotting Poisson distribution
ks = np.arange(start=min_ones, stop=max_ones+1)
ax1.step(ks-0.5, [poisson.pmf(k, expected['1']) for k in ks], 'tab:orange', where='post')

# Plots the histogram
x = np.arange(min_11, max_11+1)
ax2.bar(x, [np.sum(pat_11 == i)/len(pat_11) for i in x], width=1)
# Axes labels
ax2.set_title('11')
# Range of k values from max to min for plotting Poisson distribution
ks = np.arange(start=min_11, stop=max_11+1)
ax2.step(ks-0.5, [poisson.pmf(k, expected['11']) for k in ks], 'tab:orange', where='post')

# Plots the histogram
x = np.arange(min_111, max_111+1)
ax3.bar(x, [np.sum(pat_111 == i)/len(pat_111) for i in x], width=1)
# Axes labels
ax3.set_title('111')
# Range of k values from max to min for plotting Poisson distribution
ks = np.arange(start=min_111, stop=max_111+1)
ax3.step(ks-0.5, [poisson.pmf(k, expected['111']) for k in ks], 'tab:orange', where='post')

# Plots the histogram
x = np.arange(min_101, max_101+1)
ax4.bar(x, [np.sum(pat_101 == i)/len(pat_101) for i in x], width=1)
# Axes labels
ax4.set_title('101')
# Range of k values from max to min for plotting Poisson distribution
ks = np.arange(start=min_101, stop=max_101+1)
ax4.step(ks-0.5, [poisson.pmf(k, expected['101']) for k in ks], 'tab:orange', where='post')

plt.show()

# Madalin's estimator
QFI = 11.067842379410951
absorber = data['absorber offset'].iloc[0]
n = data['n'].iloc[0]
N = len(data['1'])
FI = 13.120604030877187
# Stores the estimators
estimators = np.zeros(N)
# Calculates the estimator for all trajectories
for i in range(N):
    sum_counts = 0
    # Sums over all observed counts
    for j in range(len(data.iloc[0, :])-4):
        sum_counts += data.iloc[i, j+4]
    estimators[i] = (2 / (FI * absorber * n) * sum_counts - absorber / 2) * np.sqrt(n)

# Creates a second figure
fig2 = plt.figure()
ax = plt.subplot(111)
# Plots a histogram of the estimators
counts, bins, ignored = ax.hist(estimators, bins=10, density=True)
# Plots the gaussian
x = np.linspace(min(estimators), max(estimators))
width = bins[1] - bins[0]
ax.plot(x, norm.pdf(x*np.sqrt(FI)) * (np.sqrt(FI)))
# Axes
ax.set_xlabel('Estimates of $\\theta$')
ax.set_ylabel('Density')
plt.show()
