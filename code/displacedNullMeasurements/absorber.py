############
# Absorber #
############
# Functions that deal with the absorber
# ss - stationary state
# t - transition operator
# id_t - identity minus transition operator
# uV - absorber unitary

# Standard imports
from qutip import *
from numpy import sin, cos, sqrt
import numpy as np


def ss(tht, lmb, phi):
    # Function that finds the stationary state of the system of interest (SOI) using the analytical formulas we
    # calculated. This stationary state is usually mixed, and we want to introduce a secondary unitary V acting on
    # the SOI and an additional adsorber system that makes the combined SOI+adsorber stationary state pure.

    # Commonly occurring square root term
    sqrts = sqrt(1-tht**2) * sqrt(1-lmb)

    # Deals with when the denom of A is 0; in which case r_z=1
    with np.errstate(divide='raise'):
        try:
            # Two constants defined in the calculation to simplify the expressions
            A = (1 - tht**2*lmb + sqrts*(-1 - tht*sqrt(lmb)*cos(phi)*(1-cos(2*tht)) + cos(2*tht)*(sqrts-1) ) ) \
                / (sqrts + tht*sqrt(lmb)*cos(phi) - 1)
            B = sin(2*tht)**2 * sqrts / (2*A)
            # Formula for rz component
            num = -(B + (cos(tht))**2) * (lmb - tht**2)
            denom = (B*(2-tht**2-lmb) + cos(2*tht) - cos(tht)**2*(tht**2+lmb) - 1)
            rz = num / denom

            # Formula for ry component
            ry = -sin(2*tht)*((2-tht**2-lmb)*rz + (lmb-tht**2)) / (2*A)

            # Formula for rx component
            rx = tht*sqrt(lmb)*sin(phi)*ry / (sqrts + tht*sqrt(lmb)*cos(phi) - 1)
        except ZeroDivisionError:
            raise

    # Grouping them together. The 4th component is necessary to account for constant terms when multiplying by the
    # matrix form of the transition operator T
    r = np.matrix([[rx],
                   [ry],
                   [rz],
                   [1]])
    return r


def t(tht, lmb, phi):
    # Finds the matrix form of the Schrodinger representation transition operator T using the analytical expressions we
    # calculated. For the stationary state vector r above, we should have T*r=r. This is used to calculate the unitary
    # V for the absorber.

    # All these terms are defined to simplify the expressions in T below
    # theta squared term
    tht_s = tht ** 2
    # Square root terms
    rt_lmb = np.sqrt(lmb)
    rt_tht_s = np.sqrt(1 - tht_s ** 2)
    rt_lmb_m = np.sqrt(1 - lmb)
    # Common combined sqrt terms
    a = rt_tht_s * rt_lmb_m
    b = tht * rt_lmb
    # Trig of phi
    cs_p = np.cos(phi)
    sn_p = np.sin(phi)
    # Trig of theta
    cs = np.cos(tht)
    cs2 = np.cos(2 * tht)
    sn = np.sin(tht)
    sn2 = np.sin(2 * tht)

    # Writing out t
    t = np.matrix([[a + b*cs_p, -b*sn_p, 0, 0],
                   [-b*sn_p, -b*cs_p + a*cs2, sn*cs*(2-tht_s-lmb), sn*cs*(lmb-tht_s)],
                   [0, -sn2*a, cs2 - cs**2*(tht_s + lmb), cs**2*(lmb - tht_s)],
                   [0, 0, 0, 1]])
    return t


def id_t(tht, lmb, phi):
    # The matrix form of Id-T, where T is the Heisenberg picture transition operator. The Moore-Penrose inverse of this
    # is used in the calculation of the QFI.

    # Terms defined to simplify expressions in code
    # theta squared term
    tht_s = tht ** 2
    # Square root terms
    rt_lmb = np.sqrt(lmb)
    rt_tht_s = np.sqrt(1 - tht_s ** 2)
    rt_lmb_m = np.sqrt(1 - lmb)
    # Common combined sqrt terms
    a = rt_tht_s * rt_lmb_m
    b = tht * rt_lmb
    # Trig of phi
    cs_p = np.cos(phi)
    sn_p = np.sin(phi)
    # Trig of theta
    cs = np.cos(tht)
    cs2 = np.cos(2 * tht)
    sn = np.sin(tht)
    sn2 = np.sin(2 * tht)

    # Writing out ID_T
    id_t = np.matrix([[1 - a - b*cs_p, b*sn_p, 0, 0],
                      [b*sn_p, 1 - cs2*a + b*cs_p, sn2*a, 0],
                      [0, -sn2*(1-(1/2)*(lmb+tht_s)), 1 - (cs2 - (1/2)*(tht_s + lmb)*(1 + cs2)), 0],
                      [0, -(1/2)*(lmb-tht_s)*sn2, -(1/2)*(lmb-tht_s)*(1+cs2), 0]])
    return id_t


def uV(tht, lmb, phi):
    # Calculates the unitary V which acts on the system of interest (SOI) and absorber system. V should reverse the
    # evolution of the SOI so that the combined SOI+absorber has a pure stationary state at theta=theta_0. V is
    # defined using two pairs of states on the absorber+input systems. Additionally, we need to specify some arbitrary
    # action on two additional degrees of freedom.
    #
    # The first pair, the phis or 'ph' states, come from purifying the ss of the SOI onto the SOI+adsorber system,
    # tensoring with the input state |0><0| and then rewriting this expression so that it splits the overall state into
    # a component with the SOI in the state |0> and a component with the ss in the state |1>. The ph states are the
    # parts tensored to this |0/1> respectively.
    #
    # Similarly, the psis or 'ps' states are the parts tensored to this |0/1> respectively after applying the evolution
    # unitary U to the combined SOI+adsorber+inputs and then repeating this splitting process again.
    #
    # We then define V as the projection from these ph states to the ps states.

    # Uses the ss function above to find the stationary state vector for the SOI
    r = ss(tht, lmb, phi)

    # Extracts the rx, ry and rz components
    rx = r[0, 0]
    ry = r[1, 0]
    rz = r[2, 0]

    # Calculated the norm of r, which is |r|<1 for mixed states
    # This term is important
    mod_r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    # Calculates the eigenstates of the ss for purification using analytical formulae
    # Also calculates these states norms, which is needed in the formulae for the states
    f_plus = Qobj([[rz + mod_r],
                   [rx + 1j*ry]]).unit()
    mod_f_plus = Qobj([[rz + mod_r],
                       [rx + 1j*ry]]).norm()
    # print(rz, mod_r, rx, ry)
    f_minus = Qobj([[rz - mod_r],
                    [rx + 1j*ry]]).unit()
    mod_f_minus = Qobj([[rz - mod_r],
                        [rx + 1j*ry]]).norm()

    # Calculates the corresponding eigenvalues
    lmb_plus = (1 / 2) * (1 + mod_r)
    lmb_minus = (1 / 2) * (1 - mod_r)

    # Calculates the purified stationary state
    sPsi = (sqrt(lmb_plus)*tensor(f_plus, f_plus) + sqrt(lmb_minus)*tensor(f_minus, f_minus)) * \
           (sqrt(lmb_plus)*tensor(f_plus, f_plus) + sqrt(lmb_minus)*tensor(f_minus, f_minus)).dag()

    # Terms introduced to simplify expressions
    a_plus = rz + mod_r
    a_minus = rz - mod_r
    b = rx + 1j * ry
    A = (1 / mod_f_plus ** 2) * np.sqrt(lmb_plus)
    B = (1 / mod_f_minus ** 2) * np.sqrt(lmb_minus)

    # ps states calculated using analytical formulae
    # Psi state 0
    ps_0 = ((A*a_plus**2 + B*a_minus**2) * tensor(basis(2, 0), basis(2, 0)) +
            (A*a_plus*b + B*a_minus*b)   * tensor(basis(2, 1), basis(2, 0))).unit()
    # Psi state 1
    ps_1 = ((A*a_plus*b + B*a_minus*b) * tensor(basis(2, 0), basis(2, 0)) +
             b**2*(A + B) * tensor(basis(2, 1), basis(2, 0))).unit()

    # Uses Gram-Schmidt to find component of ps_1 that is orthogonal to ps_0
    ps_1o = (ps_1 - (ps_0.dag() * ps_1).tr() / (ps_0.dag() * ps_0).tr() * ps_0).unit()

    ##############
    # Old method #
    ##############
    # # Calculates the other two ps states by finding the eigenstates of Id-(|ps_0><ps_0|+|ps_1><ps_1|)
    # # Projection operator for ps_0 and ps_1 states
    # proj_psi = ps_0 * ps_0.dag() + ps_1o * ps_1o.dag()
    # # Calculates the orthogonal projection
    # orth_proj_psi = qeye([2, 2]) - proj_psi
    # # Finds the eigenvalues and corresponding eigenstates
    # ev, psis = orth_proj_psi.eigenstates()
    # # The eigenvalues are ordered, so the two ev of 1 are at the end of the list. The same happens for the eigenstates.
    # ps_2 = psis[2]
    # ps_3 = psis[3]

    # New method for calculating the two ps states
    ps_2 = (Qobj(np.array([[1, 1, 1, 1]]), dims=[[1, 1], [2, 2]]).unit()).dag()
    ps_3 = (Qobj(np.array([[1, 1, 1, -1]]), dims=[[1, 1], [2, 2]]).unit()).dag()
    # Gram-Schmidt
    ps_2 = (ps_2
            - ((ps_0.dag() * ps_2).tr() / (ps_0.dag() * ps_0).tr() * ps_0)
            - ((ps_1o.dag() * ps_2).tr() / (ps_1o.dag() * ps_1o).tr() * ps_1o)).unit()
    ps_3 = (ps_3
            - ((ps_0.dag() * ps_3).tr() / (ps_0.dag() * ps_0).tr() * ps_0)
            - ((ps_1o.dag() * ps_3).tr() / (ps_1o.dag() * ps_1o).tr() * ps_1o)
            - ((ps_2.dag() * ps_3).tr() / (ps_2.dag() * ps_2).tr() * ps_2)).unit()

    # ph states calculated using analytical formulae
    # Phi state 0
    ph_0 = ((cos(tht) * sqrt(1 - tht**2) * (A*a_plus**2 + B*a_minus**2) + 1j*b*sin(tht)*sqrt(1 - lmb)
            * (A*a_plus + B*a_minus)) * tensor(basis(2, 0), basis(2, 0)) +
            b*sqrt(lmb) * np.exp(1j*phi) * (A*a_plus + B*a_minus) * tensor(basis(2, 0), basis(2, 1)) +
            (b*cos(tht) * sqrt(1-tht**2) * (A*a_plus + B*a_minus) + 1j*b**2*sin(tht) * sqrt(1-lmb)*(A + B))
            * tensor(basis(2, 1), basis(2, 0)) +
            b**2*sqrt(lmb) * np.exp(1j*phi)*(A + B) * tensor(basis(2, 1), basis(2, 1))).unit()
    # Phi state 1
    ph_1 = ((cos(tht) * sqrt(1 - lmb) * b * (A * a_plus + B * a_minus) + 1j * sin(tht) * sqrt(1 - tht ** 2) * (
            A * a_plus ** 2 + B * a_minus ** 2))
            * tensor(basis(2, 0), basis(2, 0)) +
            tht * (A * a_plus ** 2 + B * a_minus ** 2) * tensor(basis(2, 0), basis(2, 1)) +
            (b ** 2 * cos(tht) * sqrt(1 - lmb) * (A + B) + 1j * b * sin(tht) * sqrt(1 - tht ** 2) * (
            A * a_plus + B * a_minus))
            * tensor(basis(2, 1), basis(2, 0)) +
            b * tht * (A * a_plus + B * a_minus) * tensor(basis(2, 1), basis(2, 1))).unit()

    # Uses Gram-Schmidt to find component of ph_1 that is orthogonal to ph_0
    ph_1o = (ph_1 - (ph_0.dag() * ph_1).tr() / (ph_0.dag() * ph_0).tr() * ph_0).unit()

    ##############
    # Old method #
    ##############
    # # Calculates the other two ph states by finding the eigenstates of Id-(|ph_0><ph_0|+|ph_1><ph_1|)
    # # Projection operator for ps_0 and ps_1 states
    # proj_phi = ph_0*ph_0.dag() + ph_1o*ph_1o.dag()
    # # Orthogonal projection
    # orth_proj_ph = qeye([2, 2]) - proj_phi
    # # Finds the eigenvalues and corresponding eigenstates
    # ev, phis = orth_proj_ph.eigenstates()
    # # Select states corresponding to an ev of 1
    # ph_2 = phis[2]
    # ph_3 = phis[3]

    # New method for calculating arbitrary phis based on Gram-Schmidt
    ph_2 = (Qobj(np.array([[1, 1, 1, 1]]), dims=[[1, 1], [2, 2]]).unit()).dag()
    ph_3 = (Qobj(np.array([[1, 1, 1, -1]]), dims=[[1, 1], [2, 2]]).unit()).dag()
    # Gram-Schmidt
    ph_2 = (ph_2
            - ((ph_0.dag() * ph_2).tr() / (ph_0.dag() * ph_0).tr() * ph_0)
            - ((ph_1o.dag() * ph_2).tr() / (ph_1o.dag() * ph_1o).tr() * ph_1o)).unit()
    ph_3 = (ph_3
            - ((ph_0.dag() * ph_3).tr() / (ph_0.dag() * ph_0).tr() * ph_0)
            - ((ph_1o.dag() * ph_3).tr() / (ph_1o.dag() * ph_1o).tr() * ph_1o)
            - ((ph_2.dag() * ph_3).tr() / (ph_2.dag() * ph_2).tr() * ph_2)).unit()

    # Absorber Unitary V
    V = ps_0*ph_0.dag() + ps_1o*ph_1o.dag() + ps_2*ph_2.dag() + ps_3*ph_3.dag()
    return V, sPsi


if __name__ == '__main__':
    # Test parameters
    tht_test = 0.2
    phi_test = np.pi/4
    lmb_test = 0.8
    # SS vector
    print('r_ss:')
    r_ss = ss(tht_test, lmb_test, phi_test)
    print(r_ss)
    print('Mod r: {}'.format(sqrt(r_ss[0] ** 2 + r_ss[1] ** 2 + r_ss[2] ** 2)))
    # Transition operator
    print('T:')
    T = t(tht_test, lmb_test, phi_test)
    print(T)
    print('Check r is stationary:')
    print('T(r):')
    print(T * r_ss)
    print('T^2(r):')
    print(T ** 2 * r_ss)
    print('T^50(r):')
    print(T ** 50 * r_ss)
    print('T^100(r):')
    print(T ** 100 * r_ss)
    # Term that is Moore-Penrose inverted in the QFI calculation
    print('Id-T:')
    Id_T = id_t(tht_test, lmb_test, phi_test)
    print(Id_T)
    # Absorber unitary
    print('Absorber V')
    v, sPsi = uV(tht_test, lmb_test, phi_test)
    # Print statements for checking this is truly unitary
    print('Unitary V: {}'.format(v))
    print('V*V^dag:')
    print(v * v.dag())
    print('Stationary state:')
    print(sPsi)
    print('Stationary state ^2:')
    print(sPsi * sPsi)
