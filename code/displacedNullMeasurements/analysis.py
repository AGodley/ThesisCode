import math
import sys

import matplotlib.pyplot as plt
import matplotlib   # For changing tick sizes
import numpy as np
import pandas as pd
from pathlib import Path   # For file directories
from scipy.stats import poisson, norm
from qfi import qfi_calc

# Tick fontsize
matplotlib.rc('xtick', labelsize=11)
matplotlib.rc('ytick', labelsize=11)

# # Poisson distribution
# def poisson(lmbda, k):
#     return lmbda**k * np.exp(-lmbda) / np.math.factorial(k)


def u_method(i, num, traj_data, traj_mus, QFI, tol, approx_tol):
    # Calculates estimates u for the local parameter with a method specified by num
    # num=1: Includes all patterns in estimator
    # num=2: Includes specified patterns in estimator
    # num=3: Filters patterns based on ratio with pattern mu_1; i.e.- |mu_alpha|^2 / |mu_1|^2 < tol
    # num=4: Filters patterns based on Gaussian approximation of Poisson dist; i.e.- require
    # |mu_alpha|^2.offset^2 > approx_tol
    # num=5: New estimator based on expected total counts

    # i - index of trajectory being considered
    # num - selects method used
    # traj_data - data of trajectory
    # traj_mus - intensities |mu_alpha|^2 of patterns
    # tol - tolerance for method 3
    # approx_tol - tolerance for method 4

    # Absorber offset, multiply by sqrt(n) to get local offset
    offset = traj_data['offset'].iloc[i] * np.sqrt(n)

    # Finds where pattern 1 is in header
    headers = data.columns.values
    start = 0
    for head in range(len(headers)):
        if headers[head] == '1':
            start = head

    # New estimator based on total intensity
    if num == 5:
        #  Finds Total no. of counts
        sum_counts = 0
        # Their contribution to the FI
        F_restricted = 0
        for j in range(len(traj_data.iloc[i, :]) - start):
            sum_counts += traj_data.iloc[i, j + start]
            F_restricted += 4 * traj_mus.iloc[i, j + 11]
        # Final estimator
        # No initial estimation
        if math.isnan(gamma):
            # Estimator for local parameter
            u = np.sqrt(4 * sum_counts / F_restricted) - offset
        # Initial estimation
        else:
            # Estimator for local parameter; includes factor n_final / n
            u = np.sqrt(4 * sum_counts / (F_restricted * (1 - n ** (-eps)) ** (-1) ) ) - offset
    # Handles most methods
    else:
        # Method 1: Includes all patterns
        if num == 1:
            # Sums over all observed counts
            sum_counts = 0
            # Their contribution to the FI
            F_restricted = 0
            for j in range(len(traj_data.iloc[i, :]) - start):
                sum_counts += traj_data.iloc[i, j + start]
                F_restricted += 4 * traj_mus.iloc[i, j + start]
            # F_restricted = QFI  # QFI calculated at theta=theta_0

        # Method 2: Restricts to only using some patterns
        elif num == 2:
            # The patterns we're restricted to
            pat = ['1', '11', '111', '101', '1111', '11111']

            # Sums over restricted patterns
            sum_counts = 0
            # Their contribution to the FI
            F_restricted = 0

            # Calculated the FI contribution from these patterns
            for p in pat:
                mu_squared = traj_mus[p].iloc[i]
                F_restricted += 4 * mu_squared

            # Sums over counts in desired patterns
            for p in pat:
                sum_counts += traj_data[p].iloc[i]

        # Method 3: Filters patterns based on ratio with pattern mu_1; Rejects pattern when |mu_alpha|^2/|mu_1|^2 < tol
        elif num == 3:
            # pat has elements removed, so need two copies
            pats = [p for p in traj_data.columns[start:]]
            pat = pats.copy()

            # Filters patterns based on the criteria
            for j in range(len(pats)):
                if traj_mus[pats[j]].iloc[i] / traj_mus['1'].iloc[i] < tol:
                    pat.remove(pats[j])

            # Sums over restricted patterns
            sum_counts = 0
            # Their contribution to the FI
            F_restricted = 0

            # Calculated the FI contribution from these patterns
            for p in pat:
                mu_squared = traj_mus[p].iloc[i]
                F_restricted += 4 * mu_squared
            # print(F_restricted)

            # Sums over counts in desired patterns
            for p in pat:
                sum_counts += traj_data[p].iloc[i]

        # Method 4: Filters based on Guassian approximation wrt tolerance approx_tol
        elif num == 4:
            # pat has elements removed, so need two copies
            pats = [p for p in traj_data.columns[start:]]
            pat = pats.copy()

            # Filters patterns based on the criteria
            for j in range(len(pats)):
                if traj_mus[pats[j]].iloc[i] * offset**2 < approx_tol:
                    # print(pats[j], traj_mus[pats[j]].iloc[i] * offset**2)
                    pat.remove(pats[j])
                else:
                    # print(pats[j])
                    pass

            # Sums over restricted patterns
            sum_counts = 0
            # Their contribution to the FI
            F_restricted = 0

            # Calculated the FI contribution from these patterns
            for p in pat:
                mu_squared = traj_mus[p].iloc[i]
                F_restricted += 4 * mu_squared
            # print(F_restricted)

            # Sums over counts in desired patterns
            for p in pat:
                sum_counts += traj_data[p].iloc[i]

        # Handles when an incorrect method is supplied
        else:
            print('Error in estimator method function')
            sys.exit()

        # Final estimator
        # No initial estimation
        if math.isnan(gamma):
            # Estimator for local parameter
            u = (2 / (F_restricted * offset)) * (-1) * sum_counts + offset / 2
        # Initial estimation
        else:
            # Estimator for local parameter; includes factor n_final / n
            u = (2 / (F_restricted * offset)) * (1 - n ** (-eps)) ** (-1) * (-1) * sum_counts + offset / 2
    return u, F_restricted


# Finds directory for the data
adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
Model2 = adaptiveMeasurementSimulation.joinpath('data').joinpath('displaced_null_markov')

#########
# Start #
#########
# Allows us to bypass these old plots
if False:
    # Fixed theta_0
    # Reads the data
    data = pd.read_csv(Model2.joinpath('fixed').joinpath('trajectories.csv'))
    # Reads the expected counts
    expected = pd.read_csv(Model2.joinpath('fixed').joinpath('expected.csv'))

    # Creates the figure
    fig = plt.figure()
    # Gridspec
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
    # Title for figure
    fig.suptitle('Counts of various patterns', fontsize=16)
    # Axes labels
    # fig.supxlabel('Counts')
    # fig.supylabel('Density')
    # Creates the axes
    (ax1, ax2), (ax3, ax4) = gs.subplots()

    # 1
    pat_1 = data['1']
    # Finds the max and min values
    max_1 = max(pat_1)
    min_1 = min(pat_1)
    # Finds the values to plot
    x = np.arange(min_1, max_1+1)
    ax1.bar(x, [np.sum(pat_1 == i)/len(pat_1) for i in x], width=1)
    # Axes labels
    ax1.set_title('1')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_1, stop=max_1+1)
    ax1.step(ks-0.5, [poisson.pmf(k, expected['1']) for k in ks], 'tab:orange', where='post')

    # 11
    pat_11 = data['11']
    # Finds the max and min values
    max_11 = max(pat_11)
    min_11 = min(pat_11)
    # Finds the values to plot
    x = np.arange(min_11, max_11+1)
    ax2.bar(x, [np.sum(pat_11 == i)/len(pat_11) for i in x], width=1)
    # Axes labels
    ax2.set_title('11')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_11, stop=max_11+1)
    ax2.step(ks-0.5, [poisson.pmf(k, expected['11']) for k in ks], 'tab:orange', where='post')

    # 101
    pat_101 = data['101']
    # Finds the max and min values
    max_101 = max(pat_101)
    min_101 = min(pat_101)
    # Finds the values to plot
    x = np.arange(min_101, max_101+1)
    ax3.bar(x, [np.sum(pat_101 == i)/len(pat_101) for i in x], width=1)
    # Axes labels
    ax3.set_title('101')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_101, stop=max_101+1)
    ax3.step(ks-0.5, [poisson.pmf(k, expected['101']) for k in ks], 'tab:orange', where='post')

    # 111
    pat_111 = data['111']
    # Finds the max and min values
    max_111 = max(pat_111)
    min_111 = min(pat_111)
    # Finds the values to plot
    x = np.arange(min_111, max_111+1)
    ax4.bar(x, [np.sum(pat_111 == i)/len(pat_111) for i in x], width=1)
    # Axes labels
    ax4.set_title('111')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_111, stop=max_111+1)
    ax4.step(ks-0.5, [poisson.pmf(k, expected['111']) for k in ks], 'tab:orange', where='post')

    plt.show()

    # Creates a second figure for additonal patterns
    fig2 = plt.figure()
    # Gridspec
    gs2 = fig2.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
    # Title for figure
    fig2.suptitle('Counts of various patterns', fontsize=16)
    # Axes labels
    # fig2.supxlabel('Counts')
    # fig2.supylabel('Density')
    # Creates the axes
    (ax1, ax2), (ax3, ax4) = gs2.subplots()

    # 1111
    pat_1111 = data['1111']
    # Finds the max and min values
    max_1111 = max(pat_1111)
    min_1111 = min(pat_1111)
    # Finds the values to plot
    x = np.arange(min_1111, max_1111+1)
    ax1.bar(x, [np.sum(pat_1111 == i)/len(pat_1111) for i in x], width=1)
    # Axes labels
    ax1.set_title('1111')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_1111, stop=max_1111+1)
    ax1.step(ks-0.5, [poisson.pmf(k, expected['1111']) for k in ks], 'tab:orange', where='post')

    # 1101
    pat_1101 = data['1101']
    # Finds the max and min values
    max_1101 = max(pat_1101)
    min_1101 = min(pat_1101)
    # Finds the values to plot
    x = np.arange(min_1101, max_1101+1)
    ax2.bar(x, [np.sum(pat_1101 == i)/len(pat_1101) for i in x], width=1)
    # Axes labels
    ax2.set_title('1101')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_1101, stop=max_1101+1)
    ax2.step(ks-0.5, [poisson.pmf(k, expected['1101']) for k in ks], 'tab:orange', where='post')

    # 11111
    pat_11111 = data['11111']
    # Finds the max and min values
    max_11111 = max(pat_11111)
    min_11111 = min(pat_11111)
    # Finds the values to plot
    x = np.arange(min_11111, max_11111+1)
    ax3.bar(x, [np.sum(pat_11111 == i)/len(pat_11111) for i in x], width=1)
    # Axes labels
    ax3.set_title('11111')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_11111, stop=max_11111+1)
    ax3.step(ks-0.5, [poisson.pmf(k, expected['11111']) for k in ks], 'tab:orange', where='post')

    # 11011
    pat_11011 = data['11011']
    # Finds the max and min values
    max_11011 = max(pat_11011)
    min_11011 = min(pat_11011)
    # Finds the values to plot
    x = np.arange(min_11011, max_11011+1)
    ax4.bar(x, [np.sum(pat_11011 == i)/len(pat_11011) for i in x], width=1)
    # Axes labels
    ax4.set_title('11011')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_11011, stop=max_11011+1)
    ax4.step(ks-0.5, [poisson.pmf(k, expected['11011']) for k in ks], 'tab:orange', where='post')

    plt.show()

##################
# Full procedure #
##################
if True:
    # Reads the data
    data = pd.read_csv(Model2.joinpath('varying').joinpath('trajectories_varying.csv'))
    # Reads the expected counts
    expected = pd.read_csv(Model2.joinpath('varying').joinpath('expected_mus_varying.csv'))

    # Madalin's estimator
    QFI = np.real_if_close(qfi_calc(0.2, 0.8, np.pi / 4))
    n = data['n'].iloc[0]
    N = len(data['1'])
    eps = data['eps'].iloc[0]
    gamma = data['gamma'].iloc[0]
    # FI calculated from expected counts
    FI = (1/N) * np.sum(expected['calculated_FI'])
    # Absorber offset, multiply by sqrt(n) to get local offset
    offset = abs(data['offset'].iloc[0]) * np.sqrt(n)
    # Summarizes parameters
    print(f'n: {n}\nN: {N}\neps: {eps}\nn^-eps: {n**(-eps)}\ngamma: {gamma}\nQFI: {QFI}\nFI: {FI}\nOffset: {offset}')

    # Expected numbers of patterns should be mu^2.local_u^2.n
    pat = ['1', '11', '101', '111']
    if N < 5:
        for i in range(N):
            expected_numbers = {}
            for p in pat:
                mu_squared_pat = expected[p].iloc[i]
                local_u = (data['theta_rough'].iloc[i] + data['offset'].iloc[i] - 0.2) ** 2 * n
                expected_numbers[p] = mu_squared_pat * local_u
            print(expected_numbers)
    else:
        for i in range(2):
            expected_numbers = {}
            for p in pat:
                mu_squared_pat = expected[p].iloc[i]
                local_u = (data['theta_rough'].iloc[i] + data['offset'].iloc[i] - 0.2) ** 2 * n
                expected_numbers[p] = mu_squared_pat * local_u
            print(expected_numbers)

    # Initial estimator sample mean
    sample_mean_0 = 1/N * np.sum(data['theta_rough'])
    # Initial estimator sample variance
    sample_variance_0 = 1/(N-1) * np.sum((data['theta_rough'] - sample_mean_0)**2)
    print('Initial estimator')
    print(f'Sample mean: {sample_mean_0}\nSample variance: {sample_variance_0}')
    print(f'Local parameter (mean-theta): {1/N * np.sqrt(n) * np.sum(abs(data["theta_rough"] - 0.2))}')
    # Initial estimatior sample variance using true value of theta
    MSE_0 = 1/N * np.sum((data['theta_rough'] - 0.2)**2)
    print(f'[n^(1-eps).MSE_0]^-1: {1/(n**(1-eps) * MSE_0)}')
    # Initial estimation FI, uses true value of theta
    FI_0 = 1 / (MSE_0 * n**(1-eps))
    print(f'Initial estimation FI: {FI_0}')
    print(f'[n_init.FI_0]^-1: {1 / (n**(1-eps) * FI_0)}')

    # Stores the estimators
    estimators = np.zeros(N)
    # Average restricted FI
    avg_F_restricted = 0
    # Calculates the estimator for all trajectories
    for i in range(N):
        # Rough estimate
        theta_0 = data['theta_rough'].iloc[i]

        # Calculates estimators with method specified by a number 1 to 5
        qfi = qfi_calc(theta_0, 0.8, np.pi / 4)
        method = 1
        tolerance = 10**(-2)
        approximation_tolerance = 5
        u, avg_cont = u_method(i, method, data, expected, qfi, tolerance, approximation_tolerance)

        # Final estimates
        estimators[i] = theta_0 + u / np.sqrt(n)

        # Adds contribution to average
        avg_F_restricted += 1/N * avg_cont
    # Sample mean
    sample_mean = (1/N) * np.sum(estimators)
    # Sample variance
    sample_variance = 1 / (N-1) * np.sum((estimators - sample_mean)**2)
    # Sample variance using true value of theta
    MSE = 1 / N * np.sum((estimators - 0.2)**2)
    print('Final estimator')
    print(f'Sample mean: {sample_mean}\nSample variance: {sample_variance}')
    print(f'[n.sample_variance]^-1: {1 / (n * sample_variance)}')
    print(f'[n*QFI]^-1: {1 / (n * QFI)}')
    print(f'Average restricted FI: {avg_F_restricted}')
    print(f'[n.MSE]^-1: {1/(n*MSE)}')

    # Creates a second figure
    fig2 = plt.figure()
    ax = plt.subplot(111)
    # Finds the optimal binning
    bin_edges = np.histogram_bin_edges(estimators, 'doane')
    # Plots a histogram of the estimators
    counts, bins, ignored = ax.hist(estimators, bins=bin_edges, density=True)
    # Points for the pdf
    pdf_points = np.linspace(min(bins), max(bins))
    # Plots the gaussian
    pdf = norm.pdf(pdf_points, loc=0.2, scale=np.sqrt(1 / (n * QFI)))
    ax.plot(pdf_points, pdf, linewidth=2.5)
    # Axes
    # ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1990, 0.1995, 0.2, 0.2005, 0.201]))
    ax.set_xlabel('$\\hat{\\theta}$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    plt.show()

    # Creates a third figure
    fig3 = plt.figure()
    ax = plt.subplot(111)
    # If true plots the local params for initial estimation instead of final estimates
    us = True
    if us:
        # Plots a histogram of the local us
        counts, bins, ignored = ax.hist((data['theta_rough'] - 0.2)*np.sqrt(n), bins='auto', density=True)
        ax.set_xlabel('$u$', fontsize=12)
    else:
        # Plots a histogram of the estimators
        counts, bins, ignored = ax.hist(data['theta_rough'], bins='auto', density=True)
        # Plots the gaussian
        pdf2 = norm.pdf(bins, loc=sample_mean_0, scale=np.sqrt(1 / (n**(1-eps) * QFI)))
        ax.plot(bins, pdf2, linewidth=2)
        ax.set_xlabel('$\\tilde{\\theta}$', fontsize=12)
    # Axes
    ax.set_ylabel('Density', fontsize=12)
    plt.show()

##################
# Pattern counts #
##################
# Reads the data
data = pd.read_csv(Model2.joinpath('varying').joinpath('trajectories_varying_2024_03_20.csv'))
# Reads the expected counts
expected = pd.read_csv(Model2.joinpath('varying').joinpath('expected_mus_varying_2024_03_20.csv'))

if False:
    # Finds the offset and n
    offset = data['offset'].iloc[0]
    if np.isnan(data['gamma'].iloc[0]):
        # No initial estimation
        n = data['n'].iloc[0]
    else:
        # For with initial estimation
        n = data['n'].iloc[0] ** (1-data['eps'].iloc[0])    # n used in final estimation

    # Various patterns figure
    # Creates the figure
    fig4 = plt.figure()
    # Gridspec
    gs4 = fig4.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
    # Title for figure
    # fig4.suptitle('Counts of various patterns')
    # Axes labels
    # fig4.supxlabel('Counts')
    # fig4.supylabel('Frequency')
    # Creates the axes
    (ax1, ax2), (ax3, ax4) = gs4.subplots()

    #
    pat = ['1', '11', '111', '101']
    counts = {}
    for p in pat:
        counts[p] = expected[p].iloc[0] * offset**2 * n
    print(f'Expected counts:\n{counts}')

    # 1
    pat_1 = data['1']
    # Finds the max and min values
    max_1 = max(pat_1)
    min_1 = min(pat_1)
    # Finds the values to plot
    x = np.arange(min_1, max_1+1)
    ax1.bar(x, [np.sum(pat_1 == i)/len(pat_1) for i in x], width=1)
    # Axes labels
    ax1.set_title('1')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_1, stop=max_1+2)
    ax1.step(ks-0.5, [poisson.pmf(k, expected['1'].iloc[0] * offset**2 * n) for k in ks], 'tab:orange', where='post')

    # 11
    pat_11 = data['11']
    # Finds the max and min values
    max_11 = max(pat_11)
    min_11 = min(pat_11)
    # Finds the values to plot
    x = np.arange(min_11, max_11+1)
    ax2.bar(x, [np.sum(pat_11 == i)/len(pat_11) for i in x], width=1)
    # Axes labels
    ax2.set_title('11')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_11, stop=max_11+2)
    ax2.step(ks-0.5, [poisson.pmf(k, expected['11'].iloc[0] * offset**2 * n) for k in ks], 'tab:orange', where='post')

    # 101
    pat_101 = data['101']
    # Finds the max and min values
    max_101 = max(pat_101)
    min_101 = min(pat_101)
    # Finds the values to plot
    x = np.arange(min_101, max_101+1)
    ax3.bar(x, [np.sum(pat_101 == i)/len(pat_101) for i in x], width=1)
    # Axes labels
    ax3.set_title('101')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_101, stop=max_101+2)
    ax3.step(ks-0.5, [poisson.pmf(k, expected['101'].iloc[0] * offset**2 * n) for k in ks], 'tab:orange', where='post')

    # 111
    pat_111 = data['111']
    # Finds the max and min values
    max_111 = max(pat_111)
    min_111 = min(pat_111)
    # Finds the values to plot
    x = np.arange(min_111, max_111+1)
    ax4.bar(x, [np.sum(pat_111 == i)/len(pat_111) for i in x], width=1)
    # Axes labels
    ax4.set_title('111')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min_111, stop=max_111+2)
    ax4.step(ks-0.5, [poisson.pmf(k, expected['111'].iloc[0] * (offset)**2 * n) for k in ks], 'tab:orange', where='post')

    plt.show()

# Total counts figure
if False:
    fig5, ax5 = plt.subplots()
    # Finds total counts of each trajectory
    N_traj = np.zeros(N)
    for i in range(N):
        N_traj[i] = sum(data.iloc[i, 10:])
    # Finds the values to plot
    x = np.arange(min(N_traj), max(N_traj)+1)
    ax5.bar(x, [np.sum(N_traj == i)/len(N_traj) for i in x], width=1)
    # Axes labels
    ax5.set_title('111')
    # Range of k values from max to min for plotting Poisson distribution
    ks = np.arange(start=min(N_traj), stop=max(N_traj)+2)
    ax5.step(ks-0.5, [poisson.pmf(k,  (2+offset*np.sqrt(n))**2 * FI / 4) for k in ks], 'tab:orange', where='post')
    plt.show()

##########################
# Empirical mus comparison
##########################
# Calculates an empirical mu_alpha^2 from average counts in trajectories
alpha = ['1', '11', '101', '111']
for p in alpha:
    # Empirical mu squared for pattern
    mu_alpha = 0
    # In case initial estimation is included
    for i in range(len(data[p])):
        n = data['n'].iloc[i]
        eps = data['eps'].iloc[i]
        n_final = n - n**(1 - eps)
        counts_pat = np.sum(data[p].iloc[i])
        local_u = (data['theta_rough'].iloc[i] + data['offset'].iloc[i] - 0.2)
        mu_alpha += 1/len(data[p]) * counts_pat / (local_u ** 2 * n_final)
    print(f'{p}')
    print(f'(Expected mu)^2 / (empirical mu)^2 : {expected[p].iloc[0] / mu_alpha}')

#######################
# Repeated trajectory #
#######################
if False:
    # Reads the data
    data = pd.read_csv(Model2.joinpath('varying').joinpath('repeated_traj_2024_04_16.csv'))
    # Reads the expected counts
    expected = pd.read_csv(Model2.joinpath('varying').joinpath('repeated_exp_mus_2024_04_16.csv'))

    # Madalin's estimator
    QFI = np.real_if_close(qfi_calc(0.2, 0.8, np.pi / 4))
    n = data['n'].iloc[0]
    k = data['k'].iloc[0]
    N = len(data['1'])
    eps = data['eps'].iloc[0]
    n_init = int(np.floor(n**(1-eps)))
    n_final = (n - n_init) * k
    n_total = n_init + n_final
    gamma = data['gamma'].iloc[0]
    # FI calculated from expected counts
    FI = (1 / N) * np.sum(expected['calculated_FI'])
    # Absorber offset, multiply by sqrt(n) to get local offset
    offset = abs(data['offset'].iloc[0]) * np.sqrt(n)
    # Summarizes parameters
    print(f'n: {n}\nN: {N}\nk: {k}\neps: {eps}\ngamma: {gamma}\nQFI: {QFI}\nFI: {FI}\nOffset: {offset}')

    # Expected numbers of patterns should be mu^2.local_u^2.n
    pat = ['1', '11', '101', '111']
    if N < 3:
        for i in range(N):
            expected_numbers = {}
            for p in pat:
                mu_squared_pat = expected[p].iloc[i]
                local_u = (data['theta_rough'].iloc[i] + data['offset'].iloc[i] - 0.2) ** 2 * n
                expected_numbers[p] = mu_squared_pat * local_u
            print(expected_numbers)
    else:
        for i in range(2):
            expected_numbers = {}
            for p in pat:
                mu_squared_pat = expected[p].iloc[i]
                local_u = (data['theta_rough'].iloc[i] + data['offset'].iloc[i] - 0.2) ** 2 * n
                expected_numbers[p] = mu_squared_pat * local_u
            print(expected_numbers)

    # Initial estimator sample mean
    sample_mean_0 = 1 / N * np.sum(data['theta_rough'])
    # Initial estimator sample variance
    sample_variance_0 = 1 / (N - 1) * np.sum((data['theta_rough'] - sample_mean_0) ** 2)
    print('Initial estimator')
    print(f'Sample mean: {sample_mean_0}\nSample variance: {sample_variance_0}')
    print(f'Local parameter (mean-theta): {1 / N * np.sqrt(n_total) * np.sum(abs(data["theta_rough"] - 0.2))}')
    # Initial estimatior sample variance using true value of theta
    MSE_0 = 1 / N * np.sum((data['theta_rough'] - 0.2) ** 2)
    print(f'[n^(1-eps).MSE_0]^-1: {1 / (n ** (1 - eps) * MSE_0)}')
    # Initial estimation FI, uses true value of theta
    FI_0 = 1 / (MSE_0 * n ** (1 - eps))
    print(f'Initial estimation FI: {FI_0}')
    print(f'[n_init.FI_0]^-1: {1 / (n ** (1 - eps) * FI_0)}')

    # Stores the estimators
    estimators = np.zeros(N)
    # Average restricted FI
    avg_F_restricted = 0
    # Calculates the estimator for all trajectories
    for i in range(N):
        # Rough estimate
        theta_0 = data['theta_rough'].iloc[i]

        # Calculates estimators with method specified by a number 1 to 5
        qfi = qfi_calc(theta_0, 0.8, np.pi / 4)
        method = 1
        tolerance = 10 ** (-3)
        approximation_tolerance = 5
        u, avg_cont = u_method(i, method, data, expected, qfi, tolerance, approximation_tolerance)

        # Final estimates
        estimators[i] = theta_0 + u / np.sqrt(n)

        # Adds contribution to average
        avg_F_restricted += 1 / N * avg_cont
    # Sample mean
    sample_mean = (1 / N) * np.sum(estimators)
    # Sample variance
    sample_variance = 1 / (N - 1) * np.sum((estimators - sample_mean) ** 2)
    # Sample variance using true value of theta
    MSE = 1 / N * np.sum((estimators - 0.2) ** 2)
    print('Final estimator')
    print(f'Sample mean: {sample_mean}\nSample variance: {sample_variance}')
    print(f'[n_total.sample_variance]^-1: {1 / (n_total * sample_variance)}')
    print(f'[n_total*QFI]^-1: {1 / (n_total * QFI)}')
    print(f'Average restricted FI: {avg_F_restricted}')
    print(f'[n_total.MSE]^-1: {1 / (n_total * MSE)}')

    # Creates figure
    fig6 = plt.figure()
    ax = plt.subplot(111)
    # Finds the optimal binning
    bin_edges = np.histogram_bin_edges(estimators, 'doane')
    # Plots a histogram of the estimators
    counts, bins, ignored = ax.hist(estimators, bins=bin_edges, density=True)
    # Points for the pdf
    pdf_points = np.linspace(min(bins), max(bins))
    # Plots the gaussian
    pdf = norm.pdf(pdf_points, loc=0.2, scale=np.sqrt(1 / (n_total * QFI)))
    ax.plot(pdf_points, pdf, linewidth=2)
    # Axes
    # ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1990, 0.1995, 0.2, 0.2005, 0.201]))
    ax.set_xlabel('$\\hat{\\theta}$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    plt.show()

    # Creates a third figure
    fig7 = plt.figure()
    ax = plt.subplot(111)
    # If true plots the local params for initial estimation instead of final estimates
    us = True
    if us:
        # Plots a histogram of the local us
        counts, bins, ignored = ax.hist((data['theta_rough'] - 0.2) * np.sqrt(n), bins='auto', density=True)
        ax.set_xlabel('$u$', fontsize=12)
    else:
        # Plots a histogram of the estimators
        counts, bins, ignored = ax.hist(data['theta_rough'], bins='auto', density=True)
        # Plots the gaussian
        pdf2 = norm.pdf(bins, loc=sample_mean_0, scale=np.sqrt(1 / (n ** (1 - eps) * QFI)))
        ax.plot(bins, pdf2, linewidth=2)
        ax.set_xlabel('$\\tilde{\\theta}$', fontsize=12)
    # Axes
    ax.set_ylabel('Density', fontsize=12)
    plt.show()

###########
# Varying local u: FI plot #
###########
if False:
    # Reads the data
    data = pd.read_csv(Model2.joinpath('varying').joinpath('trajectories_varying_2024_04_05.csv'))
    # Reads the expected counts
    expected = pd.read_csv(Model2.joinpath('varying').joinpath('expected_mus_varying_2024_04_05.csv'))

    # Madalin's estimator
    QFI = np.real_if_close(qfi_calc(0.2, 0.8, np.pi / 4))
    n = data['n'].iloc[0]
    N = len(data['1'])
    gamma = data['gamma'].iloc[0]

    # Stores the estimators
    estimators = np.zeros(N)
    # Calculates the estimator for all trajectories
    for i in range(N):
        # Rough estimate
        theta_0 = data['theta_rough'].iloc[i]

        # Calculates estimators with method specified by a number 1 to 5
        qfi = qfi_calc(theta_0, 0.8, np.pi / 4)
        method = 1
        tolerance = 10 ** (-3)
        approximation_tolerance = 5
        u, avg_cont = u_method(i, method, data, expected, qfi, tolerance, approximation_tolerance)

        # Final estimates
        estimators[i] = theta_0 + u / np.sqrt(n)

    # No. of samples for each value of u
    batches = 100
    # No. of batches
    num_batch = int(len(data.iloc[:, 0]) / batches)
    # Stores FI of each batch
    FIs = np.zeros(num_batch)
    # Stores u of each batch
    us = np.zeros(num_batch)
    # Calculates the MSE of each batch
    for l in range(num_batch):
        print(estimators[l * batches:(l+1) * batches])
        MSE = 1 / batches * np.sum((estimators[l * batches:(l+1) * batches] - 0.2) ** 2)
        print((n*MSE)**(-1))
        us[l] = (data['theta_rough'].iloc[l*batches] - 0.2) * np.sqrt(n)
        FIs[l] = (n * MSE)**(-1)
    print('FIs:')
    print(FIs)
    print('us:')
    print(us)