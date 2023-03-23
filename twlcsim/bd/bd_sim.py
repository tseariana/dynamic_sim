"""Routines for performing Brownian dynamics simulations."""

import numpy as np
from bd.calc_internal_force import calc_internal_force
from bd.find_parameters import find_parameters
from util._vector import fix_t_vectors
from bd.calc_boundary_force import calc_boundary_force
from bd.calc_active_force import calc_temp_corr_active_force

def bd_sim(r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers, num_beads,
           length, b, confinement_tag, a, force_active0, force_active, k_a, fl_only_tag, time_total):
    """Return new values of *r_poly* after *time_total* amounts of BD."""
    epsilon= 1e11
    time = 0.
    num_total = num_polymers * num_beads
    first_bead_indices= np.arange(0, num_total, num_beads)
    last_bead_indices= np.arange(num_beads-1, num_total, num_beads)
    fl_indices= np.append(first_bead_indices, last_bead_indices)

    # Setup the parameters for the simulation (based on discretization)
    length_bead = length / (num_beads-1)
    sim_type, eps_bend, eps_par, eps_perp, eps_twist, gamma, eta, \
        xi_r_drag_coef, xi_t3_drag_coef, xi_t12_drag_coef, dt_bd = find_parameters(length, b, num_beads)

    while time < time_total:
        # Determine differential time
        dt_bd = calc_differential_time(r_poly, confinement_tag, a, epsilon, xi_r_drag_coef)


        # Calculate boundary force
        if confinement_tag:
            force_boundary = calc_boundary_force(r_poly, num_polymers, num_beads, a, epsilon)
        else:
            force_boundary = np.zeros(r_poly.shape)


        # Calculate brownian force
        force_brown, torque_1_brown, torque_2_brown, torque_3_brown = calc_brownian_force(num_total, xi_r_drag_coef, xi_t12_drag_coef, xi_t3_drag_coef, dt_bd)


        # Calculate active force
        dt_bd_a, mag_active, w0, active_force_r_integration= calc_temp_corr_active_force(num_total, force_active0, k_a, dt_bd, force_active, fl_only_tag, fl_indices)


        # Calculate internal force
        force_pot, torque_1_pot, torque_2_pot, torque_3_pot = calc_internal_force(r_poly, t1_poly, t2_poly, t3_poly, twist_poly, num_polymers, num_beads, sim_type,
                                                                         eps_bend, eps_par, eps_perp, eps_twist,
                                                                         gamma, eta)


        # rate of change of the position and orientation variables
        drdt_poly = (force_brown + force_pot + force_boundary) / xi_r_drag_coef

        dt1dt_poly = ((np.sum((torque_1_pot + torque_1_brown) * t2_poly, axis=1) - np.sum((torque_2_pot + torque_2_brown) * t1_poly, axis=1))[:, None] * t2_poly / (2 * xi_t12_drag_coef)) + \
                     ((np.sum((torque_1_pot + torque_1_brown) * t3_poly, axis=1) - np.sum((torque_3_pot + torque_3_brown) * t1_poly, axis=1))[:, None] * t3_poly / (xi_t12_drag_coef + xi_t3_drag_coef))

        dt3dt_poly = ((np.sum((torque_3_pot + torque_3_brown) * t1_poly, axis=1) - np.sum((torque_1_pot + torque_1_brown) * t3_poly, axis=1))[:, None] * t1_poly / (xi_t12_drag_coef + xi_t3_drag_coef)) + \
                     ((np.sum((torque_3_pot + torque_3_brown) * t2_poly, axis=1) - np.sum((torque_2_pot + torque_2_brown) * t3_poly, axis=1))[:, None] * t2_poly / (xi_t12_drag_coef + xi_t3_drag_coef))


        # Update the position and orientation variables
        r_poly = r_poly + dt_bd * drdt_poly + active_force_r_integration/xi_r_drag_coef
        t1_poly = t1_poly + dt_bd * dt1dt_poly
        t3_poly = t3_poly + dt_bd * dt3dt_poly
        force_active = force_active*np.exp(-dt_bd_a) + mag_active*w0

        if fl_only_tag:
            force_active_0= np.zeros(force_active.shape)
            force_active_0[fl_indices,:]= force_active[fl_indices,:]
            force_active= force_active_0
        else:
            pass

        # Force pinning of first and last beads
        if confinement_tag:
            r_poly = pin_beads(r_poly, fl_indices, a)

        # Ensure the t vectors are orthonormal
        t1_poly, t2_poly, t3_poly = fix_t_vectors(t1_poly, t2_poly, t3_poly)

        # Calculate the updated twist angles within the polymer
        twist_poly = convert_t_to_twist(t1_poly, t2_poly, twist_poly, num_beads, num_polymers)

        time += dt_bd  # Update the integration time at end of time step

    return r_poly, t1_poly, t2_poly, t3_poly, twist_poly, time, force_active #, force_boundary


def calc_differential_time(r_poly, confinement_tag, a, epsilon, xi_r_drag_coef):
    """
    Calculates the differential time step. Necessary for confined chains to prevent differential instabilities as a result of large confinement forces.

    :param r_poly: position vector of all beads in simulation (num_total,3)
    :param confinement_tag: boolean-- whether polymers are confined in sphere
    :param a: radius spherical confinement

    :return: dt_bd: Differential time step
    """
    dt_bd = 0.005
    if confinement_tag:
        r = np.sqrt(np.sum(r_poly ** 2, axis=1))
        dis_fr_conf = r - a
        max= (1/(6*epsilon*dt_bd))**0.1
        if np.any(dis_fr_conf > max): # if any distance  is postive
            dis = np.max(dis_fr_conf[dis_fr_conf>max])
            dis_force_boundary = -6 * epsilon * dis ** 11
            dt_bd = (xi_r_drag_coef * dis / np.absolute(dis_force_boundary)) * 0.2
        else:
            pass

    return dt_bd


def calc_brownian_force(num_total, xi_r_drag_coef, xi_t12_drag_coef, xi_t3_drag_coef, dt_bd):
    """
    Calculate Brownian forces for each bead in simulation

    :param num_total: Number of beads in simulation
    :param xi_r_drag_coef: Drag coefficient - Position
    :param xi_t12_drag_coef: Drag coefficient - Rotational
    :param xi_t3_drag_coef: Drag coefficient - Rotational
    :param dt_bd: Differential time step

    :return: force_brown : Brownian force (num_total, 3)
    :return: torque_1_brown : Brownian force - Rotational (num_total, 3)
    :return: torque_2_brown : Brownian force - Rotational (num_total, 3)
    :return: torque_3_brown : Brownian force - Rotational (num_total, 3)
    """
    mag_force_brown = np.sqrt(2 * xi_r_drag_coef / dt_bd)       # Prefactor for Brownian forces
    mag_torque_12_brown = np.sqrt(2 * xi_t12_drag_coef / dt_bd)      # Prefactor for Brownian torques
    mag_torque_3_brown = np.sqrt(2 * xi_t3_drag_coef / dt_bd)      # Prefactor for Brownian torques

    # forces and torques on the beads
    force_brown = mag_force_brown * np.random.randn(num_total, 3)  # Brownian forces
    torque_1_brown = mag_torque_12_brown * np.random.randn(num_total, 3)  # Brownian torques
    torque_2_brown = mag_torque_12_brown * np.random.randn(num_total, 3)  # Brownian torques
    torque_3_brown = mag_torque_3_brown * np.random.randn(num_total, 3)  # Brownian torques

    return force_brown, torque_1_brown, torque_2_brown, torque_3_brown


def pin_beads(r_poly, fl_indices, a):
    """
    Returns r_poly but with first and last beads of chains pinned to spherical confinement

    :param r_poly: Positional vector of all beads in simulation (num_total, 3)
    :param fl_indices: Indices of first and last beads of polymers
    :param a: Radius of spherical confinement

    :return: r_poly_pinned: Positional vector of chain with first and last beads pinned to the spherical confinement
    """
    r_poly_pinned = np.copy(r_poly)
    fl_r_poly = r_poly_pinned[fl_indices, :]
    radial_dis_fl = np.sqrt(np.sum(fl_r_poly ** 2, axis=1))
    ratio = a / radial_dis_fl
    ratio = np.transpose(np.tile(ratio, (3, 1)))
    new_r_poly = fl_r_poly * ratio
    r_poly_pinned[fl_indices, :] = new_r_poly

    return r_poly_pinned



def convert_t_to_twist(t1_poly, t2_poly, twist_poly, num_beads, num_polymers):
    """
    Convert the material normals to the twist angles within the polymer

    :param t1_poly: Material normal 1 to the polymer chain
    :param t2_poly: Material normal 2 to the polymer chain
    :param twist_poly: Current value of the twist angle
    :param num_beads: Number of beads in each chain
    :param num_polymers: Number of polymer chains

    :return: twist_poly: Updated twist angle from t1 and t2
    """
    for i_poly in range(num_polymers):
        ind0 = num_beads * i_poly
        indf = ind0 + num_beads

        t1_poly_plus1 = shift_vector(t1_poly, 1, num_beads, i_poly)
        t2_poly_plus1 = shift_vector(t2_poly, 1, num_beads, i_poly)
        t1_dot_t1plus1 = np.sum(t1_poly_plus1 * t1_poly[ind0:indf, :], axis=1)
        t1_dot_t2plus1 = np.sum(t2_poly_plus1 * t1_poly[ind0:indf, :], axis=1)
        t2_dot_t1plus1 = np.sum(t1_poly_plus1 * t2_poly[ind0:indf, :], axis=1)
        t2_dot_t2plus1 = np.sum(t2_poly_plus1 * t2_poly[ind0:indf, :], axis=1)
        twist_next = np.arctan2(t2_dot_t1plus1 - t1_dot_t2plus1, t1_dot_t1plus1 + t2_dot_t2plus1)
        twist_poly[ind0:indf, 0] = (twist_next +
                                    2 * np.pi * np.round((twist_poly[ind0:indf, 0] - twist_next) / (2 * np.pi)))

    return twist_poly


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
