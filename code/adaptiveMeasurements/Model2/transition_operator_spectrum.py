# Alfred Godley
# Verifying that the transition operator T has only one stationary state
# Done through manually converting the operator into a matrix, then calculating its eigenvalues

from simulation_adaptive_full_parallel import kraus
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, sin, cos

# Variables from model
theta = 0.5
lmbd = 0.2
phi = 0.5
meas = [fock(2, 0), fock(2, 1)]

if __name__ == '__main__':
    # Calculates the model's kraus operators at theta_0=theta
    Ks, *_ = kraus(theta, theta, lmbd, phi, meas)
    k0 = Ks[0]
    k1 = Ks[1]

    # Uses these to calculate the transition operator in the Heisenberg picture
    T = spre(k0.dag())*spost(k0) + spre(k1.dag())*spost(k1)

    # Calculates the basis for operators on 2D systems
    operator_basis_2D = [1/sqrt(2)*qeye(2), 1/sqrt(2)*sigmax(), 1/sqrt(2)*sigmay(), 1/sqrt(2)*sigmaz()]
    # Tensor product of the above as the basis for 4D systems (in our case 2Dx2D)
    dim = len(operator_basis_2D)
    operator_basis_4D = [None]*dim**2
    for i in range(dim):
        for j in range(dim):
            operator_basis_4D[j*dim + i] = tensor(operator_basis_2D[i], operator_basis_2D[j])

    # Calculates the matrix form of T
    T_matrix = np.zeros((dim**2, dim**2), dtype=complex)
    for i in range(dim**2):
        A = operator_basis_4D[i]
        for j in range(dim**2):
            B = operator_basis_4D[j]
            T_matrix[i][j] = (dag(A).dag() * T(B)).tr()

    # Converts it into a quantum object
    T_matrix = Qobj(T_matrix)
    # Uses qutip's functions to find the eigenvalues and eigenstates
    evs, ess = T_matrix.eigenstates()
    print(evs, ess)
