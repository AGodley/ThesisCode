############
# Patterns #
############

# Standard imports
import numpy as np
from qutip import *
from numpy.linalg import pinv

# For generating all sequences of 0s and 1s
import itertools


# Generates all binary combinations up to order+2
def possible_patterns(order):
    result = []

    # Finds all combinations of 0s and 1s up to order
    for r in range(1, order+1):
        combinations = itertools.product([0, 1], repeat=r)
        # Result is a list of tuples containing the patterns
        result.extend(combinations)

    # Initializes the dictionary that will contain the patterns
    dictionary = {}
    # Adds first two patterns
    dictionary['1'] = 0
    dictionary['11'] = 0

    # Converts each tuple into a string pattern, adds a 1 either side and adds it to a dictionary
    for tuple in result:
        pat = '1'
        for i in tuple:
            pat += str(i)
        pat += '1'
        dictionary[pat] = 0
    return dictionary


# Checks for patterns in the output list
def pattern_check(ones_list, order_patterns):
    # Creates the dictionary that stores the analysis
    analysis_x = possible_patterns(order_patterns)

    # # Finds the indices of all ones
    # ones_loc = []
    # for i in range(len(ones_list)):
    #     if ones_list[i] == 1:
    #         ones_loc.append(i)

    # Counts the 1s in all observed patterns
    weighted_sum_patterns = 0

    # Loop runs over each pattern in the dictionary above
    for key in analysis_x.keys():
        # This converts the string in the dict to the actual pattern found in the trajectory
        pattern = [int(num) for num in key]

        # Then adds a number of padding zeroes either side of the pattern
        pad = 10
        for i in range(pad):
            pattern.reverse()
            pattern.append(0)
            pattern.reverse()
            pattern.append(0)

        # Loops over the trajectory and identifies how many times the pattern occurs; this misses the tails of the list
        for i in range(len(ones_list) - len(pattern)):
            if ones_list[i:i+len(pattern)] == pattern:
                analysis_x[key] += 1
                weighted_sum_patterns += np.sum(pattern)/len(ones_list)

    return analysis_x, weighted_sum_patterns


def expected(stationary_state, ks, local_u, n_final):
    # Creates the dictionary that stores the analysis
    order_patterns = 6
    expected_x = possible_patterns(order_patterns)

    # Stores Fisher information calculation
    FI_patterns = 0

    # Calculates the mpn from a sum over expected patters
    expected_mpn = 0

    # Sum of mu terms squared
    sum_mus = 0

    # Loop runs over each pattern in the dictionary above
    for key in expected_x.keys():
        # This converts the string in the dict to the actual pattern found in the trajectory
        pattern = [int(num) for num in key]

        # Sets the mu to the stationary state for the formula
        mu_pattern = stationary_state

        # Creates a superoperator for the transition operator
        T = sprepost(ks[0], ks[0].dag()) + sprepost(ks[1], ks[1].dag())
        # Creates a superoperator for the other term in the formula
        J = sprepost(ks[0], ks[1].dag())

        # Runs through the pattern applying either T or J depending on the outcome 0 or 1 respectively
        for i in pattern:
            if i == 0:
                mu_pattern = T(mu_pattern)
            elif i == 1:
                mu_pattern = J(mu_pattern)
            else:
                raise "How'd you get here?"

        expected_x[key] = np.abs(mu_pattern.tr()) ** 2 * n_final
        # print(expected_x[key], mu_pattern.tr(), len(ones_list))

        # Cumulative sum of mus^2 updated
        sum_mus += 4 * np.abs(mu_pattern.tr()) ** 2 * np.sqrt(n_final)

        # Adds the number of photons to the total sum of photons in detected patterns
        expected_mpn += expected_x[key] / n_final * np.sum(pattern)

        # FI
        FI_patterns += 4 * expected_x[key] / n_final / (abs(local_u)) ** 2
    # Print statements
    # print(f'Analytical result for the m.p.n: {expected_mpn}')
    # print(f'Fisher information from patterns: {FI_patterns}')
    return expected_x, expected_mpn


def alternative(stationary_state_at_theta_rough, ks, ks_dot, local_u, n_final):
    # Alternative formula for Poisson rates |mu|^2
    # Creates the dictionary that stores |mu|^2
    order_patterns = 6
    alt_mu = possible_patterns(order_patterns)

    # Fisher information calculation
    alt_FI = 0

    # Dictionary for actual expected counts
    alt_expected = alt_mu.copy()

    # Loop runs over each pattern in the dictionary above
    for key in alt_expected.keys():
        # This converts the string in the dict to the actual pattern found in the trajectory
        pattern = [int(num) for num in key]

        # Sets the mu to the stationary state for the formula
        ss = stationary_state_at_theta_rough

        # Finds parts of each term that doesn't change with pattern
        inverse = Qobj(np.linalg.pinv(qeye([2, 2]) - ks[0]), dims=[[2, 2], [2, 2]])
        term_1 = ks[1] * inverse * ks_dot[0]
        term_2 = ks_dot[1]

        # Handles 1 pattern
        if key == '1':
            # Updates mus
            alt_mu[key] = np.abs((ss * term_1.dag()).tr() + (ss * term_2.dag()).tr()) ** 2
        # Handles other patterns
        else:
            # Adds product of Kraus' to this
            for i in pattern[1:]:
                term_1 = ks[i] * term_1
                term_2 = ks[i] * term_2

            # Updates mus
            alt_mu[key] = np.abs((ss * term_1.dag()).tr() + (ss * term_2.dag()).tr()) ** 2

        # Updates expected counts
        alt_expected[key] = alt_mu[key] * local_u ** 2 * n_final

        # FI
        alt_FI += 4 * alt_mu[key]
    # Print statements
    # print(f'Analytical result for the m.p.n: {expected_mpn}')
    # print(f'Fisher information from patterns: {FI_patterns}')
    return alt_mu, alt_FI, alt_expected


# Madalin's stochastic method
def stochastic_patterns(ones_list, p_end, order_pat):
    # Uses a stochastic method to search the output for patterns. This method can be imagined as a classical system
    # with two states. It sits in the ground state 0 for the bulk of the output. Whenever the output enters an excited
    # pattern (hits a 1), the classical system also enters its excited state 1. It then remains in this excited state
    # if the next bit is also a 1. If the next bit is a 0, then the classical system decays down to the ground state
    # with some probability p_end. This parameter should be carefully chosen so that the expected number steps required
    # to decay down to the ground state is comparable to the padding we use in the other method.

    # Creates the dictionary that stores the analysis
    # Algorithm will crash if the key for a pattern doesn't exist
    analysis_x = possible_patterns(order_pat)

    # Classical state for stochastic method
    c_state = 0
    # Variable that stores the current pattern
    pat = ''
    # Needed to recognise patterns such as 1001, records number of zeros between two 1s
    n_zero = 0
    # Counts the 1s in all observed patterns
    weighted_sum_patterns = 0

    # Loops through ones list
    for i in range(len(ones_list)):
        # Looks at current state of the classical chain
        if c_state == 0:    # Outside a pattern
            # Looks at next bit in output
            if ones_list[i] == 0:   # Still outside a pattern
                # Nothing needs to be done; c_state unchanged and move to next bit
                pass
            else:   # Start of a pattern
                # Updates the c_state
                c_state = 1
                # Initializes record of pattern
                pat = '1'
                # For locating the pattern
                # print(i)

        else:   # Inside a pattern
            # Looks at next bit in output
            if ones_list[i] == 0:   # May still be inside a pattern; need to do stochastic part to decide
                # Random Bernoulli trial to decide if this is the end of a pattern
                choice = np.random.choice(2, p=[p_end, 1-p_end])
                if choice == 0:     # Corresponds with ending pattern
                    # analysis_x only contains patterns up to some finite length;
                    # this deals with when a longer pattern occurs by ignoring it
                    try:
                        # Adds that pattern to the tally
                        analysis_x[pat] = analysis_x[pat] + 1
                        # Adds patterns to wsp
                        weighted_sum_patterns += np.sum([int(digit) for digit in pat])/len(ones_list)
                    except KeyError:    # Prints message, resets and ignores the pattern:
                        print(f'Unregistered pattern: {pat}')
                    # Swaps the classical state back to 0
                    c_state = 0
                    # Resets n_zero
                    n_zero = 0

                else:   # Corresponds with continuing pattern
                    # Increases n_zero
                    n_zero += 1

            else:   # Definitely inside a pattern; need to update
                # Adds on 0s and end 1
                pat = pat + '0'*n_zero + '1'
                # Resets n_zero
                n_zero = 0
    return analysis_x, weighted_sum_patterns


if __name__ == '__main__':
    # Should be 1:6, 11:3 , 101:1 , 111:1 ,1001:1, 1111:1
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pattern_check(x)
    # Generating all possible patterns up to length j+2
    j = 10
    result = possible_patterns(j)
    output = ','
    print(output.join([f'{int(i)}' for i in result.keys()]))

    # Checks regular algorithm
    result = pattern_check(x)
    print(result)

    # Checking stochastic algorithm
    result, wsp = stochastic_patterns(x, 0.4)
    output = ','
    print(result)
    print(wsp)
