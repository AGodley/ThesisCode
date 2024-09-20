# Alfred Godley
# Parallelizing this to run some longer trajectories

from math import pi, sqrt, sin  # For general maths
import numpy as np      # For arrays and matrices
from scipy.optimize import *     # For argmax
import matplotlib.pyplot as plt    # For plotting graphs
from qutip import *     # For quantum objects
import csv   # Comma separated values
from multiprocessing import Pool    # For parallelization
from itertools import repeat    # For inputs to pool
import time     # For timing the trajectory generation loop
from pathlib import Path   # For file directories

# Creates a function for calculating Kraus operators
# Also returns the corresponding pseudo-unitary
# Kraus operators are used to update the state, unitary is used to calculate next measurement
# Inputs are respectively, the param of interest, lambda, phi and a list containing the measurements
# adaptive_v3: added functionality to send 1 measurement for use when calculating classical FI
def kraus(theta_u, lmbd_u, phi_u, meas_U):
    # Not a true unitary as only it's action on |00> and |10> state are important
    unitry = Qobj([[np.cos(theta_u) * sqrt(1 - theta_u ** 2), 0, 1j*np.sin(theta_u) * sqrt(1 - lmbd_u), 0],
                   [0, 0, sqrt(lmbd_u) * np.exp(1j * phi_u), 0],
                   [1j*np.sin(theta_u) * sqrt(1 - theta_u**2), 0, np.cos(theta_u) * sqrt(1 - lmbd_u), 0],
                   [theta_u, 0, 0, 0]],
              dims=[[2, 2], [2, 2]])
    # Checks whether the function was sent 2 measurements or not
    if len(meas_U) == 2:
        # Calculates Kraus operators as Tr_e(Io|0><meas|*U)
        K_0 = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry).ptrace(0)
        K_1 = (tensor(qeye(2), fock(2, 0)*meas_U[1].dag()) * unitry).ptrace(0)
        # Kraus operators are returned in a list
        K = [K_0, K_1]
        # For checking they're proper Kraus operators
        # print(K[0].dag()*K[0]+K[1].dag()*K[1])
    else:
        # Used in fischer_cont() where measurement choice is already known
        K = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry).ptrace(0)
    return K, unitry


# Creates a function for the derivatives of the Kraus operators and unitary
# Only important at theta=0
# Needed for calculation of the classical FI and A_1
# adaptive_v3: added functionality to send 1 measurement and calc. the corresponding Kraus for calc. FI
def kraus_dot(lmbd_U, meas_U):
    unitry_dot = Qobj([[0, 0, 1j*sqrt(1-lmbd_U), 0],
                       [0, 0, 0, 0],
                       [1j, 0, 0, 0],
                       [1, 0, 0, 0]],
                      dims=[[2, 2], [2, 2]])
    # Checks whether the function was sent 2 measurements or not
    if len(meas_U) == 2:
        # Calculates K_dot operators as Tr_e(Io|0><meas|*U_dot)
        K_0_dot = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry_dot).ptrace(0)
        K_1_dot = (tensor(qeye(2), fock(2, 0)*meas_U[1].dag()) * unitry_dot).ptrace(0)
        # Once again returned in a list
        K_dot = [K_0_dot, K_1_dot]
    else:
        # Used in fischer_cont() where measurement choice is already known
        K_dot = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry_dot).ptrace(0)
    return K_dot, unitry_dot


# Generates next choice of measurement
# strat_m decides between adaptive and regular measurements
def measurement_choice(pi_mc, U_m, U_dot_m, A_1_m, j_m, strat_m, opt_m):
    # Regular +/- measurements
    if strat_m == 'reg':
        meas = [(1 / sqrt(2)) * (basis(2, 0) + np.exp(1j*opt_m)*basis(2, 1)), (1 / sqrt(2)) * (basis(2, 0) - np.exp(1j*opt_m)*basis(2, 1))]

    # Adaptive measurements
    elif strat_m == 'adapt':
        # The pi_m the function receives is the <e|A|e> from the last step
        # Begins by tensoring this with a new |0><0|
        A = tensor(pi_mc, fock_dm(2, 0))
        # A has contribution from A_1 for j>0, which this statement adds
        if j_m == 0:
            # A_1 calculated outside this function as it doesn't vary
            A = A_1_m
        else:
            # Choice of renormalisation multiplies A by 2 at each stage
            # This removes (1/2)^n term that's normally in front of A_1 term
            A = 2 * U_m * A * U_m.dag()
            A = A_1_m + A
            # A_norm = A / (-(A*A).tr()/2) # Alternative renormalisation

        # Traces out the system of interest producing B
        B = A.ptrace(1)
        print('z')
        print((B*sigmaz()).tr())
        print('I')
        print(B.tr())
        # Checks whether B=0, if so returns +/- basis measurements
        # Otherwise, calculates components of B's bloch vec. and uses these to determine an orthogonal measurement
        if np.sum(abs(B.data)) < 1e-8:

            meas = [(1 / sqrt(2)) * (basis(2, 0) + basis(2, 1)), (1 / sqrt(2)) * (basis(2, 0) - basis(2, 1))]
            # print('Measurement choice')
            # print(meas)
        else:
            # A/B are anti-hermitian, so the imaginary part of B.sigx gives rx
            rx = np.imag((B * sigmax()).tr())
            # print('rx')
            # print(rx)
            ry = np.imag((B * sigmay()).tr())
            # print('ry')
            # print(ry)
            # ~Possible change here. Madalin agreed that this is working, but he suggested a better way that calculates
            # the projections directly
            C = (1 / 2) * (qeye(2) + (rx * sigmay() - ry * sigmax()) / sqrt(rx ** 2 + ry ** 2))
            ev, meas = C.eigenstates()
        # function returns two measurements
        # loop then chooses one of these and sends <e|pi_m|e> to this function to determine the next measurement
        pi_mc = A
    return meas, pi_mc


# Calculates measurement choice for measuring the system of interest at the end
def system_Measurement(pi_n, meas_syst_s_m, strat_s_m, opt_s_m):
    if strat_s_m == 'reg' and meas_syst_s_m:
        meas = [(fock(2, 0) + np.exp(1j*opt_s_m)*fock(2, 1)).unit(), (fock(2, 0) - np.exp(1j*opt_s_m)*fock(2, 1)).unit()]
    else:
        # Can use the same method as above
        rx = np.imag((pi_n * sigmax()).tr())
        ry = np.imag((pi_n * sigmay()).tr())
        C = (1 / 2) * (qeye(2) + (rx * sigmay() - ry * sigmax()) / sqrt(rx ** 2 + ry ** 2))
        ev, meas = C.eigenstates()
    return meas


# Creates a function for ML estimation
# Essentially the same process as sample generation, but theta may now be non-zero
# args = (theta, lmbd, phi, x[i]-a sample, rho_0)
# measure_system_mle = [measure_system, system_outcome]
def mle(tau, lmbd_val, phi_val, measurements, rho_init, strat_mle, measure_system_mle, opt_mle):
    # Initiates MLE probability and state
    log_p_mle = 0
    rho_mle = rho_init

    for i_mle in np.arange(len(measurements)):
        # Finds adaptive choice of next measurement
        meas_mle = measurements[i_mle]
        # Calculates the corresponding Kraus operators
        K_mle, U_mle = kraus(tau, lmbd_val, phi_val, [meas_mle])

        # Updates the state based on which Kraus occurred in the sample
        rho_mle = K_mle * rho_mle * K_mle.dag()

        # Updates log prob. with log prob. of this Kraus occurring
        log_p_mle += np.log(rho_mle.tr())

        # Normalises the state
        rho_mle = rho_mle / rho_mle.tr()

    # Measures the system
    if measure_system_mle[0]:
        M_mle = measure_system_mle[1]
        # Defines probability that outcome 0 occurs
        p0_mle = (rho_mle * M_mle * M_mle.dag()).tr()
        log_p_mle += np.log(p0_mle)

    # Returns minus value as there is only a minimise argmax function
    return -log_p_mle


# Calculates the Fischer information of a sample based on the measurements that occurred
def fischer_contr(rho_0_fi, n_fi, lmbd_fi, phi_fi, meas_fi, measure_system_fi):
    f = 0
    for j in np.arange(n_fi):
        K_j, U_fi = kraus(0, lmbd_fi, phi_fi, [meas_fi[j]])
        # K_total added for system measurement
        K_total = K_j / (K_j * rho_0_fi).tr()
        # Initially calculates K_j_dot and finds associated renormalisation constant 1/Tr(K_j*rho)
        K_total_dot, U_fi = kraus_dot(lmbd_fi, [meas_fi[j]])
        K_total_dot = K_total_dot / (K_j * rho_0_fi).tr()
        # Then multiplies by the product K_n...K_(j-1) with their renormalisation constants
        for k in np.arange(j + 1, n_fi):
            K_k, U_fi = kraus(0, lmbd_fi, phi_fi, [meas_fi[k]])
            K_total_dot = K_k * K_total_dot / (K_k * rho_0_fi).tr()
            K_total = K_k * K_total / (K_k * rho_0_fi).tr()

        # Determines whether we're measuring the system
        if measure_system_fi[0]:
            msyst = measure_system_fi[1] * measure_system_fi[1].dag()
            # Adds the total contribution from derivative on jth Kraus operator to f
            f += ((msyst * K_total * rho_0_fi * K_total.dag()).tr())**(-1) \
                 * (2 * np.real((msyst * K_total_dot * rho_0_fi).tr()))
        else:
            # Adds the total contribution from derivative on jth Kraus operator to f
            f += (2 * np.real((K_total_dot * rho_0_fi).tr()))
    f = f**2
    return f


def Trajectory(id, setup):
    # Trajectory generation function that allows parallelization
    # Starts timer
    t0 = time.time()

    # Unpacks setup
    N = setup['N']
    n = setup['n']
    theta = setup['theta']
    lmbd = setup['lambda']
    phi = setup['phi']
    opt = setup['opt']
    strat = setup['Strategy']
    measure_system = setup['Measure system']

    # Initial state
    rho_0 = fock_dm(2, 0)
    rho = rho_0
    # Initial measurement choice
    M = [(fock(2, 0) + np.exp(1j * opt) * fock(2, 1)).unit(), (fock(2, 0) - np.exp(1j * opt) * fock(2, 1)).unit()]

    # Kraus and unitary operators for calculating A_1
    K, U = kraus(theta, lmbd, phi, M)
    # Derivative of Kraus and unitary operators at theta=0 for calculating A_1
    K_dot, U_dot = kraus_dot(lmbd, M)  # Used for calculating FI at theta=0
    # Initial A, needed for calculating future As
    A_1 = U_dot * tensor(rho, fock_dm(2, 0)) * U.dag() \
          - U * tensor(rho, fock_dm(2, 0)) * U_dot.dag()

    if id == 0:
        # M_angle stores the measurement angle of each measurement
        M_angle = np.zeros(n)
    else:
        M_angle = None

    # System Measurement Outcome
    syst_measurement = None

    # Stores measurements for calculating FI and MLE
    M_store = [None] * n

    # Initial pi_m
    pi_m = rho

    for j in np.arange(n):
        # print('Calculating the {} measurement for sample {}'.format(j + 1, i+1))
        # Finds adaptive choice of next measurement
        M, pi_m = measurement_choice(pi_m, U, U_dot, A_1, j, strat, opt)
        K, U = kraus(theta, lmbd, phi, M)

        # Defines probability that outcome 0 occurs
        p0 = (K[0] * rho * K[0].dag()).tr()
        # Choice of {0, 1} with prob {p(0), p(1)}
        x = np.random.choice([0, 1], p=[p0, 1 - p0])

        # Stores which measurement occurred for FI calculation and measurement angles plot
        M_store[j] = M[x]
        # print('Measurement choice {}'.format(j + 1))
        # print(M_store[j])

        # Updates the state by applying the measurement projection and normalising
        rho = K[x] * rho * K[x].dag()
        rho = rho / rho.tr()

        # Updates pi_m given measurement outcome
        if strat == 'adapt':
            pi_m = (tensor(qeye(2), M[x] * M[x].dag()) * pi_m).ptrace(0)

        # Extracts measurement angles of chain 0
        if id == 0:
            M_j = M_store[j].full()
            # Rotates measurement so M_j[0] is always positive
            M_j = M_j[0] / abs(M_j[0]) * M_j
            M_angle[j] = np.angle(M_j[1])

    # Measures the system
    if measure_system:
        M = system_Measurement(pi_m, measure_system, strat, opt)
        # Defines probability that outcome 0 occurs
        p0 = (rho * M[0] * M[0].dag()).tr()
        # Chooses a measurement with prob. defined above and stores it in syst_measurements
        syst_meas_outcome = np.random.choice([0, 1], p=[p0, 1 - p0])
        syst_measurement = M[syst_meas_outcome]

    # Uses the fischer_contr() function to add FI of this sample to the cumulative FI
    F = fischer_contr(rho_0, n, lmbd, phi, M_store, [False, syst_measurement])
    # Square of contributions for calculating variance
    F_sqrd = F ** 2

    # Same again for system measurement
    if measure_system:
        F_syst = fischer_contr(rho_0, n, lmbd, phi, M_store, [measure_system, M_store[-1]])
        F_syst_sqrd = F_syst ** 2
    else:
        F_syst = 0
        F_syst_sqrd = 0

    # Calculates the MLE for the sample
    # niter controls the number of attempts at estimation it tries
    # niter_success stops the process if it gets the same result the specified no. of times in a row
    # T - temperature parameter; should be comparable to step between minima
    # MLE
    bnds = ([(-0.5, 0.5)])  # Interval the argmax investigates
    result = minimize(mle, x0=np.array(0), bounds=bnds, method='Nelder-Mead',
                      args=(lmbd, phi, M_store, rho_0, strat, [measure_system, syst_measurement], opt))
    theta_est = result.x.item(0)
    # brute force on outliers
    if abs(theta_est) > 0.1:
        result, fval, grid, jout = brute(mle, ranges=bnds, Ns=200, full_output=True,
                                         args=(
                                         lmbd, phi, M_store, rho_0, strat, [measure_system, syst_measurement], opt))
        theta_est = result

    # Ends loop timer
    t1 = time.time()
    total_time = t1 - t0  # In seconds
    print('''Sample {} generated. It took {} minutes and {} seconds.
             The MLE estimate was {} and the sampling FI was {}'''.format(id + 1, np.floor(total_time / 60),
                                                                          total_time%60, theta_est, F))
    return F, F_sqrd, F_syst, theta_est, M_angle


#==========
## Setup ##
#==========
# Global parameter values
setup = {
    'N': 1,  # No. of samples to generate
    'n': 100,  # No. of ancillary systems
    'theta': 0,  # True value of theta for trajectory generation
    'lambda': 0.8,  # Transition parameter, 0<=lmbd<=1
    'phi': pi / 4,  # Phase parameter
    # Use (33/50)pi for case 1 or (14/49)pi for case 2. Found through exploration
    'opt': 0,  # Optimal regular measurement angle
    'Strategy': 'adapt',    # Measurement strategy, either 'reg' or 'adapt'
    # When True the code records the FI both with and without this final measurement
    'Measure system': True     # Final measurement on soi, set to False for regular measurements
}
#==========
## Setup ##
#==========

if __name__ == '__main__':
    # Parallel trajectories
    # Trajectory ids
    ids = np.arange(setup['N'])
    # Creates pool of processes
    pool = Pool()
    # Starts the process running
    results = pool.starmap(Trajectory, zip(ids, repeat(setup)))
    # Closes the processes
    pool.close()
    # Waits for all trajectories to be done
    pool.join()

    # Unpacks setup for calculations
    N = setup['N']
    n = setup['n']
    theta = setup['theta']
    lmbd = setup['lambda']
    phi = setup['phi']
    opt = setup['opt']
    strat = setup['Strategy']
    measure_system = setup['Measure system']

    # Unpacks the results
    F_chains = 0
    F_chains_sqrd = 0
    F_chains_syst = 0
    theta_est = np.zeros(N)
    for i in np.arange(len(results)):
        result = results[i]
        F_chains += result[0]
        F_chains_sqrd += result[1]
        F_chains_syst += result[2]
        theta_est[i] = result[3]
        if i == 0:
            M_angle = result[4]

    #==================
    # Approximate FIs #
    #==================
    # Calculates the FI as the average FI of the samples
    F_chains = (1 / N) * F_chains
    # The variance of FI of samples
    F_var = (1/N)*F_chains_sqrd - F_chains**2
    # The error associated with this variance
    error = sqrt(F_var/N)
    print('FI successfully calculated')
    print('Sample FI is {}'.format(F_chains))
    print('Error is {}'.format(error))

    # Calculates the FI including system measurement
    if measure_system:
        # The average FI of the samples
        F_chains_syst = (1 / N) * F_chains_syst
        print('Sample FI with system measurement is {}'.format(F_chains_syst))

    # Calculates FI from MLE estimates
    print('Argmax successfully completed')
    # FI inversely proportional to the sample variance
    # Using true parameter value to calculate sample variance
    sampleVar = (1/N) * np.sum((theta_est-theta)**2)
    sampleFI = 1/sampleVar
    print("The sample variance is {} and the empirical FI is {}".format(sampleVar, sampleFI))
    #==================
    # Approximate FIs #
    #==================

    #=============
    # Exact QFIs #
    #=============
    # calculate exact QFI
    QFI_exact = (1-(sqrt(1-lmbd))**n)**2/(1-sqrt(1-lmbd))**2 \
                + (1-lmbd)**(n-1)\
                + (n-1)*lmbd/(1-sqrt(1-lmbd))**2 \
                + (1-lmbd - (1-lmbd)**n)/(1-sqrt(1-lmbd))**2 \
                - 2*lmbd*(sqrt(1-lmbd)-(sqrt(1-lmbd))**n)/((1-sqrt(1-lmbd))**3) \
                + (1- (1-lmbd)**(n-1))/lmbd \
                -(1-lmbd-(1-lmbd)**n)/lmbd - 1 + n*(1-lmbd)**(n-1)\
                +n*(1- (1-lmbd)**(n-1))
    print('Exact QFI is {}'.format(4*QFI_exact))

    # calculate exact asymptotic QFI
    QFI_asymptotic =n*8*(1+ sqrt(1-lmbd)/(1-sqrt(1-lmbd)))
    print('Asymptotic QFI is {}'.format(QFI_asymptotic))
    #=============
    # Exact QFIs #
    #=============

    #===============
    # Figure plots #
    #===============
    # Measurement angle plot
    fig0 = plt.figure()
    ax0 = plt.subplot(1, 1, 1, projection='polar')
    # Corrects the angles to 0<theta<pi range
    M_angle_corrected = M_angle
    for i in np.arange(n):
        if M_angle_corrected[i] < 0:
            M_angle_corrected[i] = M_angle_corrected[i] + pi
        # Factor of two makes polar plot cover the whole circle
        M_angle_corrected[i] = 2 * M_angle_corrected[i]
    # Sets up radii for the polar plot
    r = np.linspace(1, 10, n)
    # Polar plot
    ax0.plot(M_angle_corrected, r, 'o', linewidth=0.1)
    # Arcs between points
    for i in np.arange(n-1):
        points = 100
        if abs(M_angle_corrected[i+1] - M_angle_corrected[i]) < pi:
            arc = np.linspace(M_angle_corrected[i], M_angle_corrected[i+1], points)
        else:
            if M_angle_corrected[i] < pi:
                arc0 = np.linspace(M_angle_corrected[i], 0, int(np.ceil(0.5*points)))
                arc1 = np.linspace(2*pi, M_angle_corrected[i+1], int(np.floor(0.5*points)))
                arc = np.append(arc0, arc1)
            else:
                arc0 = np.linspace(M_angle_corrected[i], 2*pi, int(np.ceil(0.5*points)))
                arc1 = np.linspace(0, M_angle_corrected[i+1], int(np.floor(0.5*points)))
                arc = np.append(arc0, arc1)
        ax0.plot(np.append(arc, M_angle_corrected[i+1]), np.append(r[i]*np.ones(points), r[i+1]), 'r--')
    ax0.set_xticks(pi/180*np.arange(0, 360, 30))
    ax0.set_xticklabels(0.5*np.arange(0, 360, 30))
    plt.show()

    # Plots histogram of MLE estimates
    plt.figure(2)
    plt.hist(theta_est)
    plt.xlabel(r'$\hat{\theta}$')
    plt.ylabel('Frequency')
    plt.show()
    #===============
    # Figure plots #
    #===============

    #=================
    # Saving Utility #
    #=================
    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[0]
    data_folder = adaptiveMeasurementSimulation.joinpath('data').joinpath('Model1')

    # Saves the figure
    save = False
    print('Saving angle figure: {}'.format(save))
    if save:
        plt.savefig(adaptiveMeasurementSimulation.joinpath('figures').joinpath('file.png'))

    # Saves the data
    save2 = False
    print('Saving FI: {}'.format(save2))
    # Toggles whether to save
    if save2:
        # Opens the file in append mode
        with open(data_folder.joinpath('fig1_adaptive.csv'), 'a', newline='') as file:
            data = [F_chains, N, n, error, lmbd, phi, strat, measure_system]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)

    # Same again with system measurement  FI instead
    save3 = False
    print('Saving FI with system measurement: {}'.format(save3))
    if save3:
        # Opens the file in append mode
        with open(data_folder.joinpath('fig1_adaptive_measure_system.csv'), 'a', newline='') as file:
            data = [F_chains_syst, N, n, None, lmbd, phi, strat, measure_system]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)

    # Saves the measurement angles
    save4 = False
    print('Saving measurement angles: {}'.format(save4))
    if save4:
        with open(data_folder.joinpath('fig2_measurement_angles.csv'), 'a', newline='') as file:
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(M_angle_corrected)

    # Saving the argmax
    save5 = False
    print('Saving the MLE estimates: {}'.format(save5))
    if save5:
        with open(data_folder.joinpath('fig3_adaptive.csv'), 'a', newline='') as file:
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(['I_MLE', 'I', 'N', 'n', 'lmbd', 'phi', 'Strategy', 'Measuring system'])
            z.writerow([sampleFI, F_chains, N, n, lmbd, phi, strat, measure_system])
            z.writerow(['Data']+theta_est.tolist())
    #=================
    # Saving Utility #
    #=================
