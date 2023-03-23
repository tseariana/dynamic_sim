"""Routines for calculating the potential forces on the beads."""
import numpy as np


def calc_internal_force(r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers, num_beads,
                                 sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta):
    """Compute default forces (to be used by __main__)."""
#    force_elas, torque_1_elas, torque_2_elas, torque_3_elas = calc_force_elas(
#        r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers, num_beads,
#        sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta)

#    force_int = calc_force_int(
#        r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers, num_beads,
#        sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta)

    force_elas, torque_1_elas, torque_2_elas, torque_3_elas = calc_force_elas_only(
        r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers, num_beads,
        sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta)

    force_pot = force_elas
    torque_1_pot = torque_1_elas
    torque_2_pot = torque_2_elas
    torque_3_pot = torque_3_elas

    return force_pot, torque_1_pot, torque_2_pot, torque_3_pot

def calc_force_elas_only(r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers,
                    num_beads, sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta):

    num_total = num_beads * num_polymers  # Total number of beads

    force_elas_only= np.zeros((num_total, 3), 'd')  # Initialize the force
    torque_1_elas = np.zeros((num_total, 3), 'd')  # Initialize the torque 1
    torque_2_elas = np.zeros((num_total, 3), 'd')  # Initialize the torque 2
    torque_3_elas = np.zeros((num_total, 3), 'd')  # Initialize the torque 3

    ind0= np.arange(0, num_total, num_beads)
    indf= np.arange(num_beads-1, num_total, num_beads)
    ind_mid= np.setxor1d(np.arange(num_total), np.append(ind0,indf))

    force_elas_only[ind_mid,:]= eps_par*(-2*r_poly[ind_mid,:] + r_poly[(ind_mid+1),:] + r_poly[(ind_mid-1),:])
    force_elas_only[ind0,:] = eps_par * (r_poly[(ind0+1),:]-r_poly[ind0, :])
    force_elas_only[indf,:] = -eps_par * (r_poly[indf,:]-r_poly[(indf-1), :])

    return force_elas_only, torque_1_elas, torque_2_elas, torque_3_elas

def calc_force_elas(r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers,
                    num_beads, sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta):
    """Calculate the elastic forces within the polymer."""
    num_total = num_beads * num_polymers            # Total number of beads

    force_elas = np.zeros((num_total, 3), 'd')      # Initialize the force
    torque_1_elas = np.zeros((num_total, 3), 'd')     # Initialize the torque 1
    torque_2_elas = np.zeros((num_total, 3), 'd')     # Initialize the torque 2
    torque_3_elas = np.zeros((num_total, 3), 'd')     # Initialize the torque 3
    for i_poly in range(num_polymers):
        ind0 = num_beads * i_poly
        indf = ind0 + num_beads

        # Setup the geometrical quantities
        r_poly_plus1 = shift_vector(r_poly, 1, num_beads, i_poly)
        delta_r = r_poly_plus1 - r_poly[ind0:indf, :]
        delta_r_par = np.sum(delta_r * t3_poly[ind0:indf, :], axis=1)
        delta_r_perp = delta_r - delta_r_par[:, None] * t3_poly[ind0:indf, :]

#        t1_poly_plus1 = shift_vector(t1_poly, 1, num_beads, i_poly)
#        t2_poly_plus1 = shift_vector(t2_poly, 1, num_beads, i_poly)
#        t3_poly_plus1 = shift_vector(t3_poly, 1, num_beads, i_poly)
#        t1_poly_minus1 = shift_vector(t1_poly, -1, num_beads, i_poly)
#        t2_poly_minus1 = shift_vector(t2_poly, -1, num_beads, i_poly)
#        t3_poly_minus1 = shift_vector(t3_poly, -1, num_beads, i_poly)
#        t3_dot_t3plus1 = np.sum(t3_poly_plus1 * t3_poly[ind0:indf, :], axis=1)
#        t1_dot_t1plus1 = np.sum(t1_poly_plus1 * t1_poly[ind0:indf, :], axis=1)
#        t2_dot_t2plus1 = np.sum(t2_poly_plus1 * t2_poly[ind0:indf, :], axis=1)
#        t1_dot_t2plus1 = np.sum(t2_poly_plus1 * t1_poly[ind0:indf, :], axis=1)
#        t2_dot_t1plus1 = np.sum(t1_poly_plus1 * t2_poly[ind0:indf, :], axis=1)

        # Calculate the force vectors
        torque_bend = eps_bend * (t3_poly_plus1 - t3_poly[ind0:indf, :] - eta * delta_r_perp)
        torque_bend_minus1 = shift_vector(torque_bend, -1, num_beads, 0)
        torque_bend_par = np.sum(torque_bend * t3_poly[ind0:indf, :], axis=1)

        force_vector = (- eta * torque_bend
                        + eta * torque_bend_par[:, None] * t3_poly[ind0:indf, :]
                        + eps_par * (delta_r_par - gamma)[:, None] * t3_poly[ind0:indf, :]
                        + eps_perp * delta_r_perp)
        force_vector_minus1 = shift_vector(force_vector, -1, num_beads, 0)

        force_elas[(ind0+1):(indf-1), :] = force_vector[1:(num_beads-1), :] - force_vector_minus1[1:(num_beads-1), :]
        force_elas[ind0, :] = force_vector[0, :]
        force_elas[indf-1, :] = -force_vector_minus1[num_beads-1, :]

        # Calculate the torque vectors

        torque_3_elas[(ind0+1):(indf-1), :] = (torque_bend - torque_bend_minus1
                                               - eta * delta_r_par[:, None] * torque_bend
                                               - eta * torque_bend_par[:, None] * delta_r
                                               - eps_par * (delta_r_par - gamma)[:, None] * delta_r
                                               + eps_perp * delta_r_par[:, None] * delta_r_perp)[1:(num_beads-1), :]
        torque_3_elas[ind0, :] = (torque_bend
                                  - eta * delta_r_par[:, None] * torque_bend
                                  - eta * torque_bend_par[:, None] * delta_r
                                  - eps_par * (delta_r_par - gamma)[:, None] * delta_r
                                  + eps_perp * delta_r_par[:, None] * delta_r_perp)[0, :]

    return force_elas, torque_1_elas, torque_2_elas, torque_3_elas


def calc_force_int(r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers, num_beads,
        sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta):
    """Calculate the interaction forces within the polymer."""
    num_total = num_beads * num_polymers            # Total number of beads
    v_int_lj = 10.                                 # Strength of Lennard-Jones interaction
    sigma_int = 4 / 53        # Interaction distance for DNA
    delta = 3

    force_int = np.zeros((num_total, 3), 'd')      # Initialize the force
    for i_poly in range(num_polymers):
        ind0 = num_beads * i_poly
        indf = ind0 + num_beads

        r_poly_i_bead = shift_vector(r_poly, 0, num_beads, i_poly)
        for j_bead in range(delta, (num_beads - delta)):
            r_poly_j_bead = shift_vector(r_poly, j_bead, num_beads, i_poly)
            delta_r_ij = r_poly_i_bead - r_poly_j_bead
            delta_r_ij_mag = np.linalg.norm(delta_r_ij, axis=1)
            e_ij = delta_r_ij / delta_r_ij_mag[:, None]

            force_int_mag = (v_int_lj * (sigma_int**12 / delta_r_ij_mag**13 - sigma_int**6 / delta_r_ij_mag**7) *
                             np.heaviside(sigma_int - delta_r_ij_mag, 0.5))
            force_int[ind0:indf, :] += force_int_mag[:, None] * e_ij

    return force_int

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
