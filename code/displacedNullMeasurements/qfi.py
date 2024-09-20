#######
# QFI #
#######
# Calculated QFI of model based off standard formula for the QFI

# Standard imports
import numpy as np
from qutip import *

# Kraus operators
from kraus import k, k_dot
# Stationary state
from absorber import ss, id_t


def qfi_calc(tht, lmbd, phi):
    # Calculates the QFI using the formula from our paper

    # Initial measurement choice; arbitrary
    M = [(fock(2, 0) + fock(2, 1)).unit(), (fock(2, 0) - fock(2, 1)).unit()]

    # Kraus operators for calculating the QFI; don't include absorber
    K, U = k(tht, lmbd, phi, M)
    K_dot, U_dot = k_dot(tht, lmbd, phi, M)

    # Finds the system's stationary state
    r = ss(tht, lmbd, phi)
    rho_ss = (1/2) * (qeye(2) + r[0, 0]*sigmax() + r[1, 0]*sigmay() + r[2, 0]*sigmaz())
    # print(rho_ss)

    # Finds R - the Moore-Penrose inverse of Id-T
    Id_T = id_t(tht, lmbd, phi)
    R = np.linalg.pinv(Id_T)
    # print(R)

    # Finds the term that R is applied to when the gauge condition is satisfied
    A_sum = K_dot[0].dag()*K[0] + K_dot[1].dag()*K[1]
    A = Qobj((A_sum-A_sum.dag()) / (2*1j))

    # print("A")
    # print(A)
    # print(np.matrix([[(A*sigmax()).tr()],
    #                  [(A*sigmay()).tr()],
    #                  [(A*sigmaz()).tr()],
    #                  [A.tr()]]))
    # print("A check: {}".format((rho_ss*A).tr()))

    # Finds a - the mean of A, important since the gauge condition isn't naturally satisfied by A
    a = (rho_ss*A).tr()
    # print("a: {}".format(a))

    # Finds the new term to apply R to
    A_tilde = (K_dot[0] + 1j*a*K[0]).dag()*K[0] + (K_dot[1] + 1j*a*K[1]).dag()*K[1]
    A_tilde = Qobj((A_tilde - A_tilde.dag()) / (2 * 1j))
    # print('A_tilde')
    # print(A_tilde)
    # print("A_tilde check: {}".format((rho_ss*A_tilde).tr()))

    # Vectorizes A_tilde for applying R
    A_x = (A_tilde*sigmax()).tr()
    A_y = (A_tilde*sigmay()).tr()
    A_z = (A_tilde*sigmaz()).tr()
    A_I = A_tilde.tr()
    A_tilde = np.matrix([[A_x],
                         [A_y],
                         [A_z],
                         [A_I]])

    # Applies R
    RA = R*A_tilde
    # print(RA)

    # Converts this back from the vector from
    RA = (1/2)*(RA[0, 0]*sigmax() + RA[1, 0]*sigmay() + RA[2, 0]*sigmaz() + RA[3, 0]*qeye(2))

    # Initialises the QFI
    Q = 0
    # Adds contribution from each Kraus operator
    for i in np.arange(len(K)):
        # print((rho_ss*(K_dot[i] + 1j*a*K[i]).dag()*(K_dot[i] + 1j*a*K[i])).tr())
        # print(2 * (Qobj(np.imag(K[i] * rho_ss * (K_dot[i] + 1j*a*K[i]).dag())) * RA).tr())
        Q += 4 * ( (rho_ss*(K_dot[i] + 1j*a*K[i]).dag()*(K_dot[i] + 1j*a*K[i])).tr() +
                   2*( (Qobj(K[i]*rho_ss*(K_dot[i] + 1j*a*K[i]).dag()) -
                        Qobj(K[i]*rho_ss*(K_dot[i] + 1j*a*K[i]).dag()).dag()) / (2*1j) * RA).tr() )

    # Returns the QFI of our model
    return Q


if __name__ == '__main__':
    print(np.real_if_close(qfi_calc(0.2, 0.8, np.pi/4)))
