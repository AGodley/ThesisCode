# Alfred Godley
# Counting measurements project:

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from numpy import pi, sqrt, cos, sin
from scipy.special import binom     # Binomial coefficients
import scipy.optimize as opt   # function that implements the mle
import time     # for timing the code
from multiprocessing import Pool    # For parallelising the trajectory generation
from itertools import repeat    # For use with starmap
from pathlib import Path   # For file directories
from scipy.signal import find_peaks
import csv


# First estimation stage
def stage1_mle(tau, angle, s1):
    # Initialises the log-likelihood
    log_l = 0

    # Prob. of measuring |0/1> at angle theta
    p0 = (1/2) * (1+cos(tau-angle))
    p1 = (1/2) * (1-cos(tau-angle))
    ps = [p0, p1]

    # Calculates the ml of a trajectory
    for i in s1:
        # log(0) raises a ZeroDivision error; this occurs at theta=0 or 1, so we ignore it here
        with np.errstate(divide='ignore'):
            log_l += np.log(ps[i])
    return -log_l


# Second estimation stage
def stage2_mle(tau, angle, theta_0, s1, s2):
    # Initialises the log-likelihood
    log_l = 0

    # Takes into account stage 1
    log_l -= stage1_mle(tau, angle, s1)

    # Prob. of measuring |0/1> at angle theta after shift
    p0 = (1 / 2) * (1 + cos(tau-theta_0))
    p1 = (1 / 2) * (1 - cos(tau-theta_0))
    ps = [p0, p1]

    # Calculates the ml of the s2 trajectory
    for i in s2:
        # log(0) raises a ZeroDivision error; this occurs at theta=0 or 1, so we ignore it here
        with np.errstate(divide='ignore'):
            log_l += np.log(ps[i])
    return -log_l


# Third estimation stage
def stage3_mle(tau, angle, theta_0, pick_theta, s1, s2, s3):
    # Initialises the log-likelihood
    log_l = 0

    # Takes into account stage 1 and 2
    log_l -= stage2_mle(tau, angle, theta_0, s1, s2)

    # Prob. of measuring |0/1> at angle theta after shift
    p0 = (1 / 2) * (1 + cos(tau - pick_theta))
    p1 = (1 / 2) * (1 - cos(tau - pick_theta))
    ps = [p0, p1]

    # Calculates the ml of the s3 trajectory
    for i in s3:
        # log(0) raises a ZeroDivision error; this occurs at theta=0 or 1, so we ignore it here
        with np.errstate(divide='ignore'):
            log_l += np.log(ps[i])
    return -log_l


# Trajectory
def Trajectory(id, setup):
    ## Sample generation and argmax ##
    # Starts loop timer
    t0 = time.time()

    # Unpacks the necessary components of setup
    n = setup['n']
    theta = setup['theta']
    eps = setup['eps']

    # Stage 1 trajectory
    n1 = int(np.floor(n*eps))     # No. of qubits to use in stage 1 estimation
    angle = 0.1
    p0 = (1/2) * (1 + cos(theta - angle))   # Prob. of measuring a qubit in the |0> state
    p1 = (1/2) * (1 - cos(theta - angle))   # Prob. of measuring a qubit in the |1> state
    s1 = np.random.choice([0, 1], size=n1, p=[p0, p1])    # Outcomes of measuring first n_tilde qubits
    k = len(s1) - np.sum(s1)  # No. of times |0> was measured
    # First estimate of theta
    theta_0 = np.arccos(2*(k/n1) - 1)

    # Want to measure away from theta-theta_0
    # theta_to_s2 = np.random.choice([-2*theta_0, 2*theta_0])
    theta_to_s2 = -0.3

    # ML Plot  for stage 1
    plot = False
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Grid of points to plot
        thetas = np.linspace(-pi, pi, 1000)
        # ML at these points
        mle = stage1_mle(thetas, s1)
        ax.plot(thetas, mle)
        ax.set_title('log-likelihood of stage 1')
        ax.set_xlabel(r'$\theta$')
        # xticks
        step = pi/2     # Step between x axes ticks
        ticks = np.arange(min(thetas), max(thetas)+step, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'${i/pi:.2g}\pi$' for i in ticks])
        # Displays the figure
        plt.show()

    # Stage 2 trajectory
    n2 = int(np.ceil(n*(1-eps)))    # int(np.floor(n23/2))   # No. of qubits to use in stage 2
    p0 = (1/2) * (1 + cos(theta-theta_to_s2))   # New prob. of measuring a qubit in the |0> state
    p1 = (1/2) * (1 - cos(theta-theta_to_s2))   # New prob. of measuring a qubit in the |1> state
    s2 = np.random.choice([0, 1], size=n2, p=[p0, p1])  # Outcomes of measuring qubits for second stage

    # ML Plot for stage 2
    thetas = np.linspace(-pi/2, pi/2, 1000)     # Grid of points to plot
    plot = False
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ML at these points
        mle = stage2_mle(thetas, theta_0, s1, s2)
        ax.plot(thetas, mle)
        ax.set_title('log-likelihood of stage 2')
        ax.set_xlabel(r'$\tau$')
        # xticks
        step = pi / 2  # Step between x axes ticks
        ticks = np.arange(min(thetas), max(thetas) + step, step)
        ax.set_xticks(ticks)
        # ax.set_xticklabels([f'${i/pi:.2g}\pi$' for i in ticks])
        # Displays the figure
        plt.show()

    # # Finds the peak in the stage 2 graph
    # ML_2 = [stage2_mle(val, theta_0, s1, s2) for val in thetas]
    # peak_indices = find_peaks(ML_2)[0]
    # # Uses that the peak we're interested in should be at theta_0 in stage 2
    # midpoint = thetas[peak_indices[0]]
    # for i in peak_indices:
    #     if abs(thetas[i] - theta_0) < abs(midpoint - theta_0):
    #         midpoint = thetas[i]
    #
    # # Finds theta_1 and theta_2
    # # theta1
    # lbup = 0.5
    # bnds1 = opt.Bounds(lb=(midpoint-lbup), ub=midpoint)
    # theta_1 = opt.minimize(stage2_mle, args=(theta_0, s1, s2), bounds=bnds1, method='Nelder-Mead',
    #                        x0=((2*midpoint - lbup)/2)).x[0]
    # # theta_2
    # bnds2 = opt.Bounds(lb=midpoint, ub=(midpoint+lbup))
    # theta_2 = opt.minimize(stage2_mle, args=(theta_0, s1, s2), bounds=bnds2, method='Nelder-Mead',
    #                        x0=((2*midpoint + lbup) / 2)).x[0]
    #
    # # Finds which was closer to theta_0
    # if stage2_mle(theta_1, theta_0, s1, s2) < stage2_mle(theta_2, theta_0, s1, s2):
    #     theta_out = theta_1
    # else:
    #     theta_out = theta_2

    # theta from brute force
    lbub = 0.5
    theta_b = opt.brute(stage2_mle, args=(angle, theta_to_s2, s1, s2), ranges=[(-lbub, lbub)], Ns=1000)
    theta_out = theta_b[0]

    # Ends loop timer
    t1 = time.time()
    total_time = t1 - t0  # In seconds
    print(f'Sample {id} took {total_time:.1f} seconds and resulted in theta_out={theta_out:.3f}')
    return theta_out


#==========
## Setup ##
#==========
# Global parameter values
setup = {
    'N': 1000,  # No. of samples to generate
    'n': 10000,  # No. of ancillary systems
    'theta': 0.1,  # True value of theta for trajectory generation
    'eps': 0.5  # Proportion of qubits to use in the first estimation stage
}
#==========
## Setup ##
#==========

if __name__ == '__main__':
    # Process ids
    ids = np.arange(setup['N'])
    # Pool object to run the multiprocessing
    pool = Pool()
    results = pool.starmap(Trajectory, zip(ids, repeat(setup)))
    pool.close()
    pool.join()
    print('Samples successfully generated')

    # Unpacks results
    N = setup['N']
    theta = setup['theta']
    estimates = np.zeros(setup['N'])
    for i in np.arange(len(results)):
        estimates[i] = results[i]

    print(theta)
    print(estimates)
    # Calculates the sample variance
    sampleVar = (1 / N) * np.sum((estimates - theta) ** 2)
    sampleFI = 1 / sampleVar
    print("The sample variance is {} and the empirical FI is {}".format(sampleVar, sampleFI))

    # ===================
    ## Saving utility ##
    # ===================
    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
    countingMeasurements = adaptiveMeasurementSimulation.joinpath('data').joinpath('countingMeasurements')

    # Saves the data
    save = True
    print('Saving FI: {}'.format(save))
    # Toggles whether to save
    if save:
        # Opens the file in append mode
        with open(countingMeasurements.joinpath('counting.csv'), 'a', newline='') as file:
            data = [setup['n'], sampleFI, theta, setup['N'], setup['n'], setup['eps']]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)