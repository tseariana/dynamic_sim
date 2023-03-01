"""Routines for performing Brownian dynamics simulations."""

import numpy as np
import time as tm
from bd.calc_force import calc_force
from bd.find_parameters import find_parameters
from util._vector import fix_t_vectors
from bd.calc_boundary_force import calc_boundary_force


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

    #mag_force_brown = np.sqrt(2 * xi_r_drag_coef / dt_bd)       # Prefactor for Brownian forces
    #mag_torque_12_brown = np.sqrt(2 * xi_t12_drag_coef / dt_bd)      # Prefactor for Brownian torques
    #mag_torque_3_brown = np.sqrt(2 * xi_t3_drag_coef / dt_bd)      # Prefactor for Brownian torques
    ##mag_active_brown= np.sqrt((2 * force_active0**2 * k_a) / dt_bd)    # (3* dt_bd)??? Prefactor for active forces

    while time < time_total:
        dt_bd = 0.005
        if confinement_tag:
            r = np.sqrt(np.sum(r_poly ** 2, axis=1))
            dis_fr_conf = r - a
            max= (1/(6*epsilon*dt_bd))**(1/10)
            if np.any(dis_fr_conf > max): # if any distance  is postive
                dis = np.max(dis_fr_conf[dis_fr_conf>max])
                dis_force_boundary = -6 * epsilon * dis ** 11
                dt_bd = (xi_r_drag_coef * dis / np.absolute(dis_force_boundary)) * 0.2
                # dt_bd= dt_bd * 10/((max_force)**.91)
            else:
                pass
            force_boundary = calc_boundary_force(r_poly, num_polymers, num_beads, a, epsilon)

        else:
            force_boundary = np.zeros(r_poly.shape)

        mag_force_brown = np.sqrt(2 * xi_r_drag_coef / dt_bd)       # Prefactor for Brownian forces
        mag_torque_12_brown = np.sqrt(2 * xi_t12_drag_coef / dt_bd)      # Prefactor for Brownian torques
        mag_torque_3_brown = np.sqrt(2 * xi_t3_drag_coef / dt_bd)      # Prefactor for Brownian torques
        #mag_active_brown= np.sqrt((2 * force_active0**2 * k_a) / dt_bd)    # (3* dt_bd)??? Prefactor for active forces

        dt_bd_a = dt_bd * k_a
        mag_active = np.sqrt(2 * force_active0 ** 2 * k_a)
        w0_2 = 1 / (2 * k_a) * (1 - np.exp(-2 * dt_bd_a))
        w1_2 = 1 / (2 * k_a ** 3) * (2 * dt_bd_a - 3 - np.exp(-2 * dt_bd_a) + 4 * np.exp(-dt_bd_a))
        w0w1 = 1 / (2 * k_a ** 2) * (1 - 2 * np.exp(-dt_bd_a) + np.exp(-2 * dt_bd_a))

        # forces and torques on the beads
        force_brown = mag_force_brown * np.random.randn(num_total, 3)  # Brownian forces
        torque_1_brown = mag_torque_12_brown * np.random.randn(num_total, 3)  # Brownian torques
        torque_2_brown = mag_torque_12_brown * np.random.randn(num_total, 3)  # Brownian torques
        torque_3_brown = mag_torque_3_brown * np.random.randn(num_total, 3)  # Brownian torques

        force_pot, torque_1_pot, torque_2_pot, torque_3_pot = calc_force(
            r_poly, t1_poly, t2_poly, t3_poly, twist_poly,
            num_polymers, num_beads, sim_type,
            eps_bend, eps_par, eps_perp, eps_twist,
            gamma, eta) # Potential force and torque


        # active force calculation
        w0 = np.sqrt(w0_2) * np.random.randn(num_total, 3)
        w1 = w0w1 / np.sqrt(w0_2) * np.random.randn(num_total, 3) + np.sqrt(np.absolute(w1_2 - (w0w1 ** 2 / w0_2))) * np.random.randn(num_total, 3)

        active_force_r_integration = (1 / k_a * (1 - np.exp(-dt_bd_a)) * force_active + mag_active * w1)

        if fl_only_tag:
            active_force_r_integration_0= np.zeros(active_force_r_integration.shape)
            active_force_r_integration_0[fl_indices,:] = active_force_r_integration[fl_indices,:]
            active_force_r_integration= active_force_r_integration_0
        else:
            pass

        # rate of change of the position and orientation variables
        drdt_poly = (force_brown + force_pot + force_boundary) / xi_r_drag_coef
        dt1dt_poly = (
            (np.sum((torque_1_pot + torque_1_brown) * t2_poly, axis=1)
             - np.sum((torque_2_pot + torque_2_brown) * t1_poly, axis=1))[:, None] *
            t2_poly / (2 * xi_t12_drag_coef)) + (
            (np.sum((torque_1_pot + torque_1_brown) * t3_poly, axis=1)
             - np.sum((torque_3_pot + torque_3_brown) * t1_poly, axis=1))[:, None] *
            t3_poly / (xi_t12_drag_coef + xi_t3_drag_coef))
        dt3dt_poly = (
            (np.sum((torque_3_pot + torque_3_brown) * t1_poly,
                    axis=1)
             - np.sum((torque_1_pot + torque_1_brown) * t3_poly, axis=1))[:, None] *
            t1_poly / (xi_t12_drag_coef + xi_t3_drag_coef)) + (
            (np.sum((torque_3_pot + torque_3_brown) * t2_poly, axis=1)
             - np.sum((torque_2_pot + torque_2_brown) * t3_poly, axis=1))[:, None] *
            t2_poly / (xi_t12_drag_coef + xi_t3_drag_coef))


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
            fl_r_poly = r_poly[fl_indices, :]
            r_fl_r_poly= np.sqrt(np.sum(fl_r_poly ** 2, axis=1))
            ratio = a / r_fl_r_poly
            ratio = np.transpose(np.tile(ratio, (3, 1)))
            new_r_poly = fl_r_poly * ratio
            r_poly[fl_indices, :] = new_r_poly
        else:
            pass

        # Ensure the t vectors are orthonormal
        t1_poly, t2_poly, t3_poly = fix_t_vectors(t1_poly, t2_poly, t3_poly)

        # Calculate the updated twist angles within the polymer
        twist_poly = convert_t_to_twist(t1_poly, t2_poly, twist_poly, num_beads, num_polymers)

        time += dt_bd  # Update the integration time at end of time step


    return r_poly, t1_poly, t2_poly, t3_poly, twist_poly, time, force_active #, force_boundary


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
