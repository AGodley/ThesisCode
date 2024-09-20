# Alfred Godley
# Counting measurements project:
# Estimating a parameter theta from iid qubits
# Assumes theta has been localised through some preliminary estimation to a neighbourhood of size d=n^(-1/2 + eps)
# Then measures at angle 2d on half the n qubits and at angle -2d on the other half

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


# Complete (first + second) estimation
def mle(tau, angle, angle2, s1, s2):
    # Initialises the log-likelihood
    log_l = 0

    # Prob. of measuring |0/1> at angle theta after shift
    p0 = (1 / 2) * (1 + cos(tau - angle))
    p1 = (1 / 2) * (1 - cos(tau - angle))
    ps = [p0, p1]

    # Calculates the ml of the s1 trajectory
    for i in s1:
        # log(0) raises a ZeroDivision error; this occurs at theta=0 or 1, so we ignore it here
        with np.errstate(divide='ignore'):
            log_l += np.log(ps[i])

    # Prob. of measuring |0/1> at angle theta after shift
    p0 = (1 / 2) * (1 + cos(tau - angle2))
    p1 = (1 / 2) * (1 - cos(tau - angle2))
    ps = [p0, p1]

    # Calculates the ml of the s2 trajectory
    for i in s2:
        # log(0) raises a ZeroDivision error; this occurs at theta=0 or 1, so we ignore it here
        with np.errstate(divide='ignore'):
            log_l += np.log(ps[i])

    return -log_l


# First estimation stage
def mle_stage1(tau, angle, angle2, s1, s2):
    # Initialises the log-likelihood
    log_l = 0

    # Prob. of measuring |0/1> at angle theta after shift
    p0 = (1 / 2) * (1 + cos(tau - angle))
    p1 = (1 / 2) * (1 - cos(tau - angle))
    ps = [p0, p1]

    # Calculates the ml of the s1 trajectory
    for i in s1:
        # log(0) raises a ZeroDivision error; this occurs at theta=0 or 1, so we ignore it here
        with np.errstate(divide='ignore'):
            log_l += np.log(ps[i])

    # Prob. of measuring |0/1> at angle theta after shift
    p0 = (1 / 2) * (1 + cos(tau - angle2))
    p1 = (1 / 2) * (1 - cos(tau - angle2))
    ps = [p0, p1]

    # # Calculates the ml of the s2 trajectory
    # for i in s2:
    #     # log(0) raises a ZeroDivision error; this occurs at theta=0 or 1, so we ignore it here
    #     with np.errstate(divide='ignore'):
    #         log_l += np.log(ps[i])

    return -log_l


# Second estimation stage
def mle_stage2(tau, angle, angle2, s1, s2):
    # Initialises the log-likelihood
    log_l = 0

    # Prob. of measuring |0/1> at angle theta after shift
    p0 = (1 / 2) * (1 + cos(tau - angle))
    p1 = (1 / 2) * (1 - cos(tau - angle))
    ps = [p0, p1]

    # # Calculates the ml of the s1 trajectory
    # for i in s1:
    #     # log(0) raises a ZeroDivision error; this occurs at theta=0 or 1, so we ignore it here
    #     with np.errstate(divide='ignore'):
    #         log_l += np.log(ps[i])

    # Prob. of measuring |0/1> at angle theta after shift
    p0 = (1 / 2) * (1 + cos(tau - angle2))
    p1 = (1 / 2) * (1 - cos(tau - angle2))
    ps = [p0, p1]

    # Calculates the ml of the s2 trajectory
    for i in s2:
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
    eps = setup['eps']

    # Assumes we've localized theta already
    theta = n**(-1/2 + eps)

    # Stage 1 trajectory
    n1 = int(np.floor(n/2))     # No. of qubits to use in stage 1 estimation
    angle = 3 * n**(-1/2 + 2*eps)  # Measures at twice the localised region
    p0 = (1/2) * (1 + cos(theta - angle))   # Prob. of measuring a qubit in the |0> state
    p1 = (1/2) * (1 - cos(theta - angle))   # Prob. of measuring a qubit in the |1> state
    s1 = np.random.choice([0, 1], size=n1, p=[p0, p1])    # Outcomes of measuring first n_tilde qubits

    # Stage 2 trajectory
    n2 = int(np.ceil(n/2))    # int(np.floor(n23/2))   # No. of qubits to use in stage 2
    angle2 = -3 * n**(-1/2 + 2*eps)   # Measures at minus twice the localised region
    p0 = (1/2) * (1 + cos(theta - angle2))   # New prob. of measuring a qubit in the |0> state
    p1 = (1/2) * (1 - cos(theta - angle2))   # New prob. of measuring a qubit in the |1> state
    s2 = np.random.choice([0, 1], size=n2, p=[p0, p1])  # Outcomes of measuring qubits for second stage

    # ML Plot for stage 1 and 2 side-by-side
    plot = True
    if plot:
        # Grid of points to plot
        thetas = np.linspace(-15 * theta, 15 * theta, 100)

        # Creates the figure
        fig = plt.figure()
        # Adds the axes for the first plot
        ax = fig.add_subplot(121)
        # ML at these points
        maxlikelihood = mle_stage1(thetas, angle, angle2, s1, s2)
        # Plots the ML
        ax.plot(thetas, maxlikelihood)
        # Plots a horizontal line at theta and angle
        ax.vlines(theta, min(maxlikelihood), max(maxlikelihood), 'r')
        ax.annotate(r'$\theta$', xy=(theta, max(maxlikelihood) / 2))
        ax.vlines(angle, min(maxlikelihood), max(maxlikelihood), 'y')
        ax.annotate(r'$angle$', xy=(angle, max(maxlikelihood) / 2))
        # Axes and title
        ax.set_title('log-likelihood of stage 1')
        ax.set_xlabel(r'$\tau$')

        # Second stage
        # Adds the axes for the second plot
        ax2 = fig.add_subplot(122)
        # ML for the second stage
        maxlikelihood = mle_stage2(thetas, angle, angle2, s1, s2)
        # Plots the ML
        ax2.plot(thetas, maxlikelihood)
        # Plots a horizontal line at theta and angle2
        ax2.vlines(theta, min(maxlikelihood), max(maxlikelihood), 'r')
        ax2.annotate(r'$\theta$', xy=(theta, max(maxlikelihood) / 2))
        ax2.vlines(angle2, min(maxlikelihood), max(maxlikelihood), 'y')
        ax2.annotate(r'$angle2$', xy=(angle2, max(maxlikelihood) / 2))
        # Axes and title
        ax2.set_title('log-likelihood of stage 2')
        ax2.set_xlabel(r'$\tau$')

        # Displays the figure
        plt.show()

    # theta from brute force for stage 1
    # We expect two peaks centred around angle
    theta_b = opt.brute(mle_stage1, args=(angle, angle2, s1, s2), ranges=[(-angle/2, angle/2)], Ns=500)
    theta_1 = theta_b[0]
    diff = abs(angle - theta_1)
    theta_1_alt = angle + diff

    # theta from brute force for stage 2
    # We expect two peaks centred around angle2=-angle
    theta_b = opt.brute(mle_stage2, args=(angle, angle2, s1, s2), ranges=[(angle2/2, -angle2/2)], Ns=500)
    theta_2 = theta_b[0]
    diff = abs(angle2 - theta_2)
    theta_2_alt = angle2 - diff

    # Takes the two closest
    theta_out = 1/2 * (theta_1 + theta_2)

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
    'n': 10,  # No. of ancillary systems
    'eps': 0.1  # Proportion of qubits to use in the first estimation stage
}
#==========
## Setup ##
#==========

if __name__ == '__main__':
    # Process ids
    ids = np.arange(setup['N'])
    # Pool object to run the multiprocessing
    pool = Pool(10)
    results = pool.starmap(Trajectory, zip(ids, repeat(setup)))
    pool.close()
    pool.join()
    print('Samples successfully generated')

    # Unpacks the necessary components of setup
    n = setup['n']
    eps = setup['eps']
    N = setup['N']
    theta = n**(-1/2 + eps)    # Assumes we've localized theta already

    # Unpacks results
    estimates = np.zeros(setup['N'])
    for i in np.arange(len(results)):
        estimates[i] = results[i]

    print(estimates)
    print(f'Theta was {theta}')
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