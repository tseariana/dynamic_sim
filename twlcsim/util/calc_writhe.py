"""
Calculate the writhe of each chain.

Calculation of the writhe of a discrete ring, as
given by method 1a in [K. Klenin, J. Langowski, Biopolymers, 54, 307 (2000)].

Input is the position of the discrete points of the chain,
the twist angles in the curve, and the size of the ring.

"""
import numpy as np


def calc_writhe(r_poly, num_polymers, num_beads):
    num_total = num_polymers * num_beads  # Total number of beads

    writhe = np.zeros((num_polymers, 1), 'd')
    for i_poly in range(num_polymers):

        r_poly_i_bead = shift_vector(r_poly, 0, num_beads, i_poly)
        r_poly_i_bead_plus1 = shift_vector(r_poly_i_bead, 1, num_beads, 0)
        for j_bead in range(2, (num_beads - 1)):
            r_poly_j_bead = shift_vector(r_poly, j_bead, num_beads, i_poly)
            r_poly_j_bead_plus1 = shift_vector(r_poly_j_bead, 1, num_beads, 0)

            r13 = r_poly_i_bead - r_poly_j_bead
            r23 = r_poly_i_bead - r_poly_j_bead_plus1
            r14 = r_poly_i_bead_plus1 - r_poly_j_bead
            r24 = r_poly_i_bead_plus1 - r_poly_j_bead_plus1
            r12 = r_poly_j_bead_plus1 - r_poly_j_bead
            r34 = r_poly_i_bead_plus1 - r_poly_i_bead

            norm_1 = np.cross(r13, r14)
            norm_1 /= np.linalg.norm(norm_1, axis=1)[:, None]
            norm_2 = np.cross(r14, r24)
            norm_2 /= np.linalg.norm(norm_2, axis=1)[:, None]
            norm_3 = np.cross(r24, r23)
            norm_3 /= np.linalg.norm(norm_3, axis=1)[:, None]
            norm_4 = np.cross(r23, r13)
            norm_4 /= np.linalg.norm(norm_4, axis=1)[:, None]

            writhe_angle = (np.arcsin(np.sum(norm_1 * norm_2, axis=1))
                            + np.arcsin(np.sum(norm_2 * norm_3, axis=1))
                            + np.arcsin(np.sum(norm_3 * norm_4, axis=1))
                            + np.arcsin(np.sum(norm_1 * norm_4, axis=1))
                            )
            sign_mag = np.sign(np.sum(r13 * np.cross(r34, r12), axis=1))
            writhe[i_poly] += np.sum(writhe_angle * sign_mag)

    writhe /= (4 * np.pi)

    return writhe


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
