###################
# Kraus operators #
###################
import sys

# Standard imports
import numpy as np
from numpy import sqrt, pi
from qutip import *

# Absorber
from absorber import uV


# Kraus operators without the absorber
def k(tht, lmbd, phi, meas):
    # Inputs the unitary U; this is not a true unitary as only it's action on the |00>, |10> states are important.
    # At each step a new auxiliary system |0> interacts with our soi.
    U = Qobj([[np.cos(tht) * sqrt(1 - tht ** 2), 0, 1j * np.sin(tht) * sqrt(1 - lmbd), 0],
              [0, 0, sqrt(lmbd) * np.exp(1j * phi), 0],
              [1j * np.sin(tht) * sqrt(1 - tht ** 2), 0, np.cos(tht) * sqrt(1 - lmbd), 0],
              [tht, 0, 0, 0]],
             dims=[[2, 2], [2, 2]])

    # Checks whether the function was sent 1 or 2 measurements
    if len(meas) == 2:
        # WHen sent two measurements, calculates both corresponding Kraus operators as Tr_e(Io|0><meas|*U)
        K_0 = (tensor(qeye(2), fock(2, 0) * meas[0].dag()) * U).ptrace([0])
        K_1 = (tensor(qeye(2), fock(2, 0) * meas[1].dag()) * U).ptrace([0])

        # Kraus operators are returned in a list
        K = [K_0, K_1]

        # For checking they're proper Kraus operators
        # print('Kraus check:')
        # print(K[0].dag()*K[0] + K[1].dag()*K[1])
    else:
        # Kraus corresponding to a single measurement
        K = (tensor(qeye(2), fock(2, 0) * meas[0].dag()) * U).ptrace([0])
    return K, U


# Derivative of Kraus operator function without absorber, used for calculating QFI
def k_dot(tht, lmbd, phi, meas):
    # Inputs the unitary's derivative; calculated analytically elsewhere
    U_dot = Qobj([[-np.sin(tht) * sqrt(1 - tht ** 2) - tht * np.cos(tht) * ((1 - tht ** 2) ** (-1 / 2)), 0,
                   1j * np.cos(tht) * sqrt(1 - lmbd), 0],
                  [0, 0, 0, 0],
                  [1j * np.cos(tht) * sqrt(1 - tht ** 2) - 1j * tht * np.sin(tht) * ((1 - tht ** 2) ** (-1 / 2)), 0,
                   -np.sin(tht) * sqrt(1 - lmbd), 0],
                  [1, 0, 0, 0]],
                 dims=[[2, 2], [2, 2]])

    # Checks whether the function was sent 2 measurements or not
    if len(meas) == 2:
        # Calculates K_dot operators as Tr_e(Io|0><meas|*U_dot)
        K_0_dot = (tensor(qeye(2), fock(2, 0) * meas[0].dag()) * U_dot).ptrace([0])
        K_1_dot = (tensor(qeye(2), fock(2, 0) * meas[1].dag()) * U_dot).ptrace([0])

        # Once again returned in a list
        K_dot = [K_0_dot, K_1_dot]
    else:
        # Used when calculating the FI of a trajectory as measurements are already known
        K_dot = (tensor(qeye(2), fock(2, 0) * meas[0].dag()) * U_dot).ptrace([0])
    return K_dot, U_dot


def k_abs(tht, tht_r, lmbd, phi, meas):
    # Function for calculating the Kraus operators
    # Also returns the corresponding pseudo-unitary
    # Kraus operators are used to update the state
    # Inputs: param of interest, a rough estimate of this param, lambda, phi and a list containing the measurements
    # When sent one measurement, it calculates the corresponding single Kraus

    # Not a true unitary as only it's action on |000>, |100>, |010> and |110> states are important
    U = Qobj([[np.cos(tht)*sqrt(1-tht**2), 0, 1j*np.sin(tht)*sqrt(1-lmbd), 0],
              [0, 0, sqrt(lmbd)*np.exp(1j*phi), 0],
              [1j*np.sin(tht) * sqrt(1-tht**2), 0, np.cos(tht)*sqrt(1-lmbd), 0],
              [tht, 0, 0, 0]],
             dims=[[2, 2], [2, 2]])

    # SOI+adsorber unitary
    V, *_ = uV(tht_r, lmbd, phi)

    # Permutation operator used to make U act on syst+out instead of syst+absorber
    P23 = tensor(fock(2, 0), fock(2, 0)) * tensor(fock(2, 0), fock(2, 0)).dag() + \
          tensor(fock(2, 1), fock(2, 0)) * tensor(fock(2, 0), fock(2, 1)).dag() + \
          tensor(fock(2, 0), fock(2, 1)) * tensor(fock(2, 1), fock(2, 0)).dag() + \
          tensor(fock(2, 1), fock(2, 1)) * tensor(fock(2, 1), fock(2, 1)).dag()

    # Total unitary
    VU = tensor(qeye(2), V) * (tensor(qeye(2), P23)*tensor(U, qeye(2))*tensor(qeye(2), P23))

    # Checks whether the function was sent 2 measurements or not
    if len(meas) == 2:
        # WHen sent two measurements, calculates both corresponding Kraus operators as Tr_e(Io|0><meas|*VU)
        K_0 = (tensor(qeye(2), qeye(2), fock(2, 0)*meas[0].dag()) * VU).ptrace([0, 1])
        K_1 = (tensor(qeye(2), qeye(2), fock(2, 0)*meas[1].dag()) * VU).ptrace([0, 1])

        # Kraus operators are returned in a list
        K = [K_0, K_1]

        # For checking they're proper Kraus operators
        # print('Kraus check:')
        # print(K[0].dag()*K[0] + K[1].dag()*K[1])
        # print(K)
        # print(f'k_1 is {K_1}')
        # print(K_1.dag()*K_1)
    else:
        # Calculates a single Kraus operator as Tr_e(Io|0><meas|*VU)
        K = (tensor(qeye(2), qeye(2), fock(2, 0)*meas[0].dag()) * VU).ptrace([0, 1])
    return K, VU, V, U


def k_abs_dot(tht, tht_r, lmbd, phi, meas):
    # Function for the derivatives of the Kraus operators and unitary
    # Needed for calculation of the classical FI
    # When sent one measurement, it calculates the corresponding single Kraus derivative

    unitry_dot = Qobj([[-np.sin(tht) * sqrt(1 - tht ** 2) - tht * np.cos(tht) * (1 - tht ** 2) ** (-1 / 2), 0,
                        1j * np.cos(tht) * sqrt(1 - lmbd), 0],
                       [0, 0, 0, 0],
                       [1j * np.cos(tht) * sqrt(1 - tht ** 2) - 1j * tht * np.sin(tht) * (1 - tht ** 2) ** (-1 / 2), 0,
                        -np.sin(tht) * sqrt(1 - lmbd), 0],
                       [1, 0, 0, 0]],
                      dims=[[2, 2], [2, 2]])

    # SOI+adsorber unitary
    V, *_ = uV(tht_r, lmbd, phi)

    # Permutation operator makes U act on syst+out instead of syst+absorber
    P23 = tensor(fock(2, 0), fock(2, 0))*tensor(fock(2, 0), fock(2, 0)).dag() + \
        tensor(fock(2, 1), fock(2, 0)) * tensor(fock(2, 0), fock(2, 1)).dag() + \
        tensor(fock(2, 0), fock(2, 1)) * tensor(fock(2, 1), fock(2, 0)).dag() + \
        tensor(fock(2, 1), fock(2, 1)) * tensor(fock(2, 1), fock(2, 1)).dag()

    # Total
    VU_diff = tensor(qeye(2), V)*(tensor(qeye(2), P23)*tensor(unitry_dot, qeye(2))*tensor(qeye(2), P23))

    # Checks whether the function was sent 2 measurements or not
    if len(meas) == 2:
        # Calculates K_dot operators as Tr_e(Io|0><meas|*U_dot)
        K_0_dot = (tensor(qeye(2), qeye(2), fock(2, 0) * meas[0].dag()) * VU_diff).ptrace([0, 1])
        K_1_dot = (tensor(qeye(2), qeye(2), fock(2, 0) * meas[1].dag()) * VU_diff).ptrace([0, 1])
        # Once again returned in a list
        K_dot = [K_0_dot, K_1_dot]
    else:
        # Calculates K_dot operator as Tr_e(Io|0><meas|*U_dot)
        K_dot = (tensor(qeye(2), qeye(2), fock(2, 0) * meas[0].dag()) * VU_diff).ptrace([0, 1])

    return K_dot, VU_diff, V, unitry_dot


def true_ss(tht, tht_r, lmbd, phi, meas):
    # Calculates the true stationary state of VU in order to calculate the mpn

    # Kraus operators
    K, *_ = k_abs(tht, tht_r, lmbd, phi, meas)

    T_mat = np.zeros((16, 16), dtype=complex)
    for i in range(16):
        for j in range(16):
            # Runs through a set of basis vectors for vectorized form
            r_j = np.zeros(16)
            r_j[j] = 1
            r_j = Qobj(r_j.transpose(), type='operator-ket')
            # Converts to operator form
            R_j = np.zeros((4, 4))
            R_j[j // 4, j % 4] = 1
            # Applies transition operator
            tr = (K[0].data * R_j * (K[0].dag()).data + K[1].data * R_j * (K[1].dag()).data)
            # Converts back to vector
            Tj = np.hstack([tr[l, :] for l in range(4)])
            # Takes ith component for the ij component of the Transition matrix
            T_mat[i, j] = Tj[i]
    # Calculates its eigenvalues and eigenvectors; expect an evec with eval 1 corresponding to ss
    ev, vs = np.linalg.eig(T_mat)
    # print(ev)

    # Gets rid of small complex components caused by numerics
    evs = np.array([np.real_if_close(i) for i in ev])
    # print(evs)

    # Selects the eigenvalues with value 1
    criteria = np.arange(len(evs))[1-abs(evs) < 1e-10]
    # print(ev[criteria])
    # Alerts us when there are multiple stationary state
    if len(criteria) > 1:
        print(evs[criteria])
        raise Exception('Multiple stationary states')
    # Selects the stationary state
    r_ss = -vs[:, criteria[0]]

    # Converts that evec into operator form (the ss)
    rho_0 = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            rho_0[i, j] = r_ss[i * 4 + j]
    rho_0 = Qobj(rho_0, dims=[[2, 2], [2, 2]])

    # Normalizes the trace
    # print(f"Unormalized: {rho_0}")
    # print(f'Trace: {rho_0.tr()}')
    rho_0 = rho_0 / rho_0.tr()
    if not rho_0.check_herm():
        raise Exception("Stationary state isn't hermitian")
    return rho_0


if __name__ == '__main__':
    # Test parameters
    tht_test = 0.2
    lmbd_test = 0.8
    phi_test = pi/4
    # Standard counting measurement
    M = [fock(2, 0), fock(2, 1)]

    # Checks the functions run sucessfully
    Kraus, *_ = k(tht_test, lmbd_test, phi_test, M)
    # For checking they're proper Kraus operators
    print('Checking Kraus operators without absorber:')
    print(Kraus[0].dag()*Kraus[0] + Kraus[1].dag()*Kraus[1])

    # Checks the functions run sucessfully
    Kraus, *_ = k_abs(tht_test, tht_test+0.01, lmbd_test, phi_test, M)
    # For checking they're proper Kraus operators
    print('Checking Kraus operators with absorber:')
    print(Kraus[0].dag() * Kraus[0] + Kraus[1].dag() * Kraus[1])

    # Checks whether the Kraus operators with counting measurement have coefficients in the standard basis
    K, *_ = k(tht_test, lmbd_test, phi_test, M)
    K_dot, *_ = k_dot(tht_test, lmbd_test, phi_test, M)
    print('Checking coefficients of the Kraus operators')
    mat_0 = np.array([[M[0].dag() * K[0] * M[0], M[0].dag() * K[0] * M[1]],
                      [M[1].dag() * K[0] * M[0], M[1].dag() * K[0] * M[1]]])
    print(f'<e_a|K_0|e_b:\n{mat_0}')
    mat_1 = np.array([[M[0].dag() * K[1] * M[0], M[0].dag() * K[1] * M[1]],
                      [M[1].dag() * K[1] * M[0], M[1].dag() * K[1] * M[1]]])
    print(f'<e_a|K_1|e_b:\n{mat_1}')
    mat_3 = np.array([[M[0].dag() * K_dot[0] * M[0], M[0].dag() * K_dot[0] * M[1]],
                      [M[1].dag() * K_dot[0] * M[0], M[1].dag() * K_dot[0] * M[1]]])
    print(f'<e_a|K_dot_0|e_b:\n{mat_3}')
    mat_4 = np.array([[M[0].dag() * K_dot[1] * M[0], M[0].dag() * K_dot[1] * M[1]],
                      [M[1].dag() * K_dot[1] * M[0], M[1].dag() * K_dot[1] * M[1]]])
    print(f'<e_a|K_dot_1|e_b:\n{mat_4}')
