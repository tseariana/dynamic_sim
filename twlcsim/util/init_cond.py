"""
Setup the initial condition for the polymer chains.

The initial condition can be set using two options:

1. Random initialization
2. Initialization from file

"""
import numpy as np
from .chaingrowth import random_walk_gauss_chain
from .chaingrowth import gen_conf_rouse_active1
from .chaingrowth import gen_conf_rouse_active2
from .chaingrowth import confined_linear_chain
from .chaingrowth import confined_chain
from ._vector import fix_t_vectors

def init_cond(length, b, num_polymers, num_beads, force_active0, k_a, fl_only_tag, confinement_tag, a,
              from_file, loaded_file, input_dir):
    """Create num_polymer of num_beads each either from_file or randomly."""
    num_total = num_polymers * num_beads  # Total number of beads

    linking_number = 0
    length_bead = length / (num_beads-1)
    lp_dna = 53     # Persistence length of DNA in nm
    length_dim = length_bead * 0.34 / lp_dna        # Non-dimensionalized length per bead

    if from_file:
        print("Initialize from saved conformation")
        file =  np.loadtxt(input_dir + 'pos_file_' + loaded_file, delimiter=',', skiprows=2)
        r_poly= file[:,0:3]
        t1_poly=file[:,3:6]
        t2_poly= file[:,6:9]
        t3_poly= file[:,9:12]
        force_active= np.loadtxt(input_dir + 'fa_file_' + loaded_file, delimiter=',', skiprows=2)

        #r_poly = np.loadtxt(input_dir + "r_poly_" + loaded_file, delimiter=',')
        #t1_poly = np.loadtxt(input_dir + "t1_poly_" + loaded_file, delimiter=',')
        #t2_poly = np.loadtxt(input_dir + "t2_poly_" + loaded_file, delimiter=',')
        #t3_poly = np.loadtxt(input_dir + "t3_poly_" + loaded_file, delimiter=',')
        #force_active = np.loadtxt(input_dir + "force_active_" + loaded_file, delimiter=',')
    else:
        print("Initialize from random conformation")

        r_poly = np.zeros((num_total, 3), 'd')
        t1_poly = np.zeros((num_total, 3), 'd')
        t2_poly = np.zeros((num_total, 3), 'd')
        t3_poly = np.zeros((num_total, 3), 'd')
        force_active = np.zeros((num_total, 3), 'd')
        t1_poly[:, 0] = 1
        t2_poly[:, 1] = 1
        t3_poly[:, 2] = 1

        first_bead_indices= np.arange(0, num_total, num_beads)
        last_bead_indices= np.arange(num_beads-1, num_total, num_beads)
        fl_indices= np.append(first_bead_indices, last_bead_indices)

        # USING CHAINGROWTH.PY
        for n in range(0, num_polymers):
            if confinement_tag:
                print('Confined')
                r_poly_n, force_active_n = confined_chain(length, num_beads, a, fa=force_active0)
            else:
                r_poly_n, force_active_n = gen_conf_rouse_active1(length, num_beads, ka=k_a, fa=force_active0, b=b, num_modes=10000)
            r_poly[n*num_beads:(n+1)*num_beads, :] = r_poly_n
            force_active[n * num_beads:(n + 1) * num_beads, :] = force_active_n

        if force_active0==0:
            force_active = np.zeros((num_total, 3))

        if fl_only_tag:
            force_active_0 = np.zeros(force_active.shape)
            force_active_0[fl_indices, :] = force_active[fl_indices, :]
            force_active = force_active_0

        # Active force: First and last beads only
        #force_active_0= np.zeros(force_active.shape)
        #force_active_0[fl_indices,:]= force_active[fl_indices,:]
        #force_active= force_active_0

    # Ensure the t vectors are orthonormal
    t1_poly, t2_poly, t3_poly = fix_t_vectors(t1_poly, t2_poly, t3_poly)

    # Calculate the twist angle at each bead
    twist_poly = np.zeros((num_total, 1), 'd')
    for i_poly in range(num_polymers):
        ind0 = num_beads * i_poly
        indf = ind0 + num_beads

        t1_poly_plus1 = shift_vector(t1_poly, 1, num_beads, i_poly)
        t2_poly_plus1 = shift_vector(t2_poly, 1, num_beads, i_poly)
        t1_dot_t1plus1 = np.sum(t1_poly_plus1 * t1_poly[ind0:indf, :], axis=1)
        t1_dot_t2plus1 = np.sum(t2_poly_plus1 * t1_poly[ind0:indf, :], axis=1)
        t2_dot_t1plus1 = np.sum(t1_poly_plus1 * t2_poly[ind0:indf, :], axis=1)
        t2_dot_t2plus1 = np.sum(t2_poly_plus1 * t2_poly[ind0:indf, :], axis=1)
        twist_poly[ind0:indf, 0] = np.arctan2(t2_dot_t1plus1 - t1_dot_t2plus1, t1_dot_t1plus1 + t2_dot_t2plus1)

    return r_poly, t1_poly, t2_poly, t3_poly, twist_poly, force_active


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


def shift_array(a, shift_index, num_beads, i_poly):
    """
    Generate a step forward/back array by shift_index steps.

    input:  array a        Full array (length num_beads * num_polymers x 3)
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

    a_shift = np.concatenate([a[mid_index:end_index], a[ind0:mid_index]])

    return a_shift