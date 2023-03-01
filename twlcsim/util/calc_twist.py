"""
Calculate the twist of each chain.

Calculation of the twist of a discrete ring, as
given by method 1a in [K. Klenin, J. Langowski, Biopolymers, 54, 307 (2000)].

Input is the position of the discrete points of the chain,
the twist angles in the curve, and the size of the ring.

"""
import numpy as np


def calc_twist(twist_poly, num_polymers, num_beads):
    num_total = num_polymers * num_beads  # Total number of beads

    twist = np.zeros((num_polymers, 1), 'd')
    for i_poly in range(num_polymers):
        ind0 = num_beads * i_poly
        indf = ind0 + num_beads

        twist[i_poly] = np.sum(twist_poly[ind0:indf, None])

    twist /= (2 * np.pi)

    return twist


def shift_vector(a, shift_index, num_beads, i_poly):
    """
    Generate a step forward/back vector by shift_index steps.

    input:  vector a        Full vector (length num_beads * num_polymers x 3)
            shift_index     Index to shift the vector
            num_beads       Number of beads in each polymer
            i_poly          Index of the polymer to output the shift vector

    output: a_shift         Shifted vector (length num_bead x 3)

    """
    ind0 = num_beads * i_poly  # Determine the zero index for i_poly
    # Shift over shift_index to be between 0 and num_beads
    shift_index = shift_index % num_beads

    mid_index = ind0 + shift_index
    end_index = ind0 + num_beads

    a_shift = np.concatenate([a[mid_index:end_index, :], a[ind0:mid_index, :]])

    return a_shift
